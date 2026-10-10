#include "runtime/gs/gs_worker.h"
#include "runtime/gs/gs_frontend.h"
#include "runtime/vu1_worker.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

namespace
{
    struct Worker
    {
        GS *gs = nullptr;
        std::thread thread;
        std::mutex mutex;
        std::condition_variable workReady;
        std::condition_variable workDone;
        std::deque<std::vector<uint8_t>> queue;
        std::vector<std::vector<uint8_t>> spare; // buffers to reuse
        size_t queuedBytes = 0;
        uint64_t posted = 0;
        uint64_t completed = 0;
        bool quit = false;
        std::atomic<bool> active{false};
        std::thread::id threadId;

        // The EE waits once this much is queued, so a slow host cannot make
        // the queue grow without bound.
        static constexpr size_t kMaxQueuedBytes = 32u * 1024u * 1024u;

        void loop()
        {
            // Everything queued is taken in one go, so the threads that post
            // meet this one at the lock once per run, not per packet.
            std::deque<std::vector<uint8_t>> taken;
            std::unique_lock<std::mutex> lock(mutex);
            for (;;)
            {
                workReady.wait(lock, [&]
                               { return quit || !queue.empty(); });
                if (queue.empty())
                    return;
                taken.swap(queue);
                lock.unlock();

                // PS2X_GS_STATS=1: how much of its time the worker spends drawing.
                static const bool wantStats = []
                {
                    const char *value = std::getenv("PS2X_GS_STATS");
                    return value != nullptr && value[0] != '\0' && value[0] != '0';
                }();
                const auto started = wantStats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

                size_t bytes = 0;
                for (const std::vector<uint8_t> &packet : taken)
                {
                    gs->processGIFPacket(packet.data(), static_cast<uint32_t>(packet.size()));
                    bytes += packet.size();
                }
                const uint64_t count = taken.size();

                if (wantStats)
                {
                    static double busySeconds = 0.0;
                    static uint64_t packets = 0;
                    static auto windowStart = started;
                    const auto finished = std::chrono::steady_clock::now();
                    busySeconds += std::chrono::duration<double>(finished - started).count();
                    packets += count;
                    const double window = std::chrono::duration<double>(finished - windowStart).count();
                    if (window >= 2.0)
                    {
                        std::fprintf(stderr, "[gs-worker] busy %.0f%% of the time, %.0f packets/s\n",
                                     100.0 * busySeconds / window, static_cast<double>(packets) / window);
                        busySeconds = 0.0;
                        packets = 0;
                        windowStart = finished;
                    }
                }

                lock.lock();
                completed += count;
                queuedBytes -= bytes;
                while (!taken.empty())
                {
                    if (spare.size() < 256u)
                        spare.push_back(std::move(taken.front()));
                    taken.pop_front();
                }
                workDone.notify_all();
            }
        }
    };

    Worker &worker()
    {
        static Worker instance;
        return instance;
    }
}

namespace
{
    std::atomic<void (*)()> g_interruptSentHook{nullptr};
}

namespace GsWorker
{
    void setInterruptSentHook(void (*hook)())
    {
        g_interruptSentHook.store(hook, std::memory_order_release);
    }

    void interruptSent()
    {
        if (void (*hook)() = g_interruptSentHook.load(std::memory_order_acquire))
            hook();
        else
            drain();
    }

    bool raisesInterrupt(const uint8_t *data, uint32_t sizeBytes)
    {
        const uint32_t qwords = sizeBytes / 16u;
        uint32_t q = 0;
        while (q < qwords)
        {
            uint64_t tagLo, tagHi;
            std::memcpy(&tagLo, data + q * 16u, sizeof(tagLo));
            std::memcpy(&tagHi, data + q * 16u + 8u, sizeof(tagHi));
            ++q;
            const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFFu);
            const uint32_t flg = static_cast<uint32_t>((tagLo >> 58) & 0x3u);
            uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xFu);
            if (nreg == 0u)
                nreg = 16u;
            if (flg == 1u)
            {
                q += (nloop * nreg + 1u) / 2u;
                continue;
            }
            if (flg >= 2u)
            {
                q += nloop;
                continue;
            }
            // PACKED: only A+D entries can address FINISH (0x61) or SIGNAL (0x60).
            uint32_t addressAndData = 0u;
            for (uint32_t r = 0; r < nreg; ++r)
                if (((tagHi >> (4u * r)) & 0xFu) == 0xEu)
                    addressAndData |= 1u << r;
            if (addressAndData == 0u)
            {
                q += nloop * nreg;
                continue;
            }
            for (uint32_t loop = 0; loop < nloop && q < qwords; ++loop)
            {
                for (uint32_t r = 0; r < nreg && q < qwords; ++r, ++q)
                {
                    if ((addressAndData >> r) & 1u)
                    {
                        const uint8_t address = data[q * 16u + 8u];
                        if (address == 0x60u || address == 0x61u)
                            return true;
                    }
                }
            }
        }
        return false;
    }

    bool mayRaiseInterrupt(const uint8_t *data, uint32_t sizeBytes)
    {
        // An A+D entry is a data doubleword followed by the register address.
        for (uint32_t offset = 8u; offset + 8u <= sizeBytes; offset += 8u)
        {
            uint64_t address;
            std::memcpy(&address, data + offset, sizeof(address));
            if (address == 0x60u || address == 0x61u)
                return true;
        }
        return false;
    }

    void start(GS &gs)
    {
        Worker &w = worker();
        if (w.active.load(std::memory_order_acquire))
            return;
        w.gs = &gs;
        const char *value = std::getenv("PS2X_GS_THREAD");
        if (value != nullptr && value[0] == '0')
            return;
        w.quit = false;
        w.thread = std::thread([&w]
                               { w.loop(); });
        w.threadId = w.thread.get_id();
        w.active.store(true, std::memory_order_release);
    }

    void stop()
    {
        Worker &w = worker();
        if (!w.active.exchange(false, std::memory_order_acq_rel))
            return;
        {
            std::lock_guard<std::mutex> lock(w.mutex);
            w.quit = true;
        }
        w.workReady.notify_all();
        if (w.thread.joinable())
            w.thread.join();
    }

    bool running()
    {
        return worker().active.load(std::memory_order_acquire);
    }

    void post(const uint8_t *data, uint32_t sizeBytes)
    {
        Worker &w = worker();
        if (!data || sizeBytes == 0u || !w.gs)
            return;
        if (!w.active.load(std::memory_order_acquire) || std::this_thread::get_id() == w.threadId)
        {
            w.gs->processGIFPacket(data, sizeBytes);
            return;
        }

        // The copy is made outside the lock.
        std::vector<uint8_t> packet;
        {
            std::unique_lock<std::mutex> lock(w.mutex);
            w.workDone.wait(lock, [&]
                            { return w.queuedBytes < Worker::kMaxQueuedBytes; });
            if (!w.spare.empty())
            {
                packet = std::move(w.spare.back());
                w.spare.pop_back();
            }
        }
        packet.assign(data, data + sizeBytes);
        bool wasEmpty = false;
        {
            std::lock_guard<std::mutex> lock(w.mutex);
            wasEmpty = w.queue.empty();
            w.queuedBytes += packet.size();
            w.queue.push_back(std::move(packet));
            ++w.posted;
        }
        // The worker only sleeps on an empty queue.
        if (wasEmpty)
            w.workReady.notify_one();

        // Sent straight from the EE (no VU1 worker in between): an interrupt
        // this packet raises is pending when the send returns.
        if (!Vu1Worker::onWorkerThread() && raisesInterrupt(data, sizeBytes))
            interruptSent();
    }

    bool drain()
    {
        Worker &w = worker();
        if (std::this_thread::get_id() == w.threadId || Vu1Worker::onWorkerThread())
            return false;
        // VU1 produces GS packets: it has to finish before the GS can.
        const bool waitedForVu1 = Vu1Worker::drain();
        if (!w.active.load(std::memory_order_acquire))
            return waitedForVu1;
        std::unique_lock<std::mutex> lock(w.mutex);
        if (w.completed == w.posted)
            return waitedForVu1;
        const uint64_t target = w.posted;
        w.workDone.wait(lock, [&]
                        { return w.completed >= target; });
        return true;
    }
}
