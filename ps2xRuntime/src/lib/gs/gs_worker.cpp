#include "runtime/gs/gs_worker.h"
#include "runtime/gs/gs_frontend.h"
#include "runtime/vu1_worker.h"

#include <atomic>
#include <condition_variable>
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
            std::unique_lock<std::mutex> lock(mutex);
            for (;;)
            {
                workReady.wait(lock, [&]
                               { return quit || !queue.empty(); });
                if (queue.empty())
                    return;
                std::vector<uint8_t> packet = std::move(queue.front());
                queue.pop_front();
                queuedBytes -= packet.size();
                lock.unlock();

                gs->processGIFPacket(packet.data(), static_cast<uint32_t>(packet.size()));

                lock.lock();
                ++completed;
                if (spare.size() < 64u)
                    spare.push_back(std::move(packet));
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

namespace GsWorker
{
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

        std::unique_lock<std::mutex> lock(w.mutex);
        w.workDone.wait(lock, [&]
                        { return w.queuedBytes < Worker::kMaxQueuedBytes; });
        std::vector<uint8_t> packet;
        if (!w.spare.empty())
        {
            packet = std::move(w.spare.back());
            w.spare.pop_back();
        }
        packet.assign(data, data + sizeBytes);
        w.queuedBytes += packet.size();
        w.queue.push_back(std::move(packet));
        ++w.posted;
        lock.unlock();
        w.workReady.notify_one();

        // Sent straight from the EE (no VU1 worker in between): an interrupt
        // this packet raises is pending when the send returns.
        if (!Vu1Worker::onWorkerThread() && raisesInterrupt(data, sizeBytes))
            drain();
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
