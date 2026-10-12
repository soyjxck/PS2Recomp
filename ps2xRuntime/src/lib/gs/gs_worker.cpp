#include "runtime/gs/gs_worker.h"
#include "runtime/gs/gs_frontend.h"
#include "runtime/ps2_memory.h"
#include "runtime/ps2_settings.h"
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
    // PS2X_GS_STATS: how long callers of drain() waited, VU1's part first,
    // and the same by where they waited.
    std::atomic<uint64_t> g_drainWaits{0}, g_drainVu1Nanoseconds{0}, g_drainGsNanoseconds{0};
    constexpr size_t kWaitKinds = static_cast<size_t>(GsWorker::Wait::Count);
    std::atomic<uint64_t> g_waitNanoseconds[kWaitKinds], g_waitCount[kWaitKinds];
    const char *const kWaitNames[kWaitKinds] = {"elsewhere", "an interrupt sent", "an interrupt due", "vblank", "idle",
                                                "CSR read", "sceGsSyncPath", "a VIF1 register", "VU1 memory"};

    bool statsWanted()
    {
        static const bool wanted = []
        {
            const char *value = std::getenv("PS2X_GS_STATS");
            return value != nullptr && value[0] != '\0' && value[0] != '0';
        }();
        return wanted;
    }

    void countWait(GsWorker::Wait why, std::chrono::steady_clock::time_point since)
    {
        const size_t kind = static_cast<size_t>(why) < kWaitKinds ? static_cast<size_t>(why) : 0u;
        g_waitCount[kind].fetch_add(1u, std::memory_order_relaxed);
        g_waitNanoseconds[kind].fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                              std::chrono::steady_clock::now() - since)
                                              .count()),
                                          std::memory_order_relaxed);
    }

    // "the EE waited at: vblank 31% (60/s, 5.2 ms each), ..."
    void printWaits(double window)
    {
        char line[512];
        int used = std::snprintf(line, sizeof(line), "[gs-waits] the EE waited at:");
        bool any = false;
        for (size_t kind = 0; kind < kWaitKinds; ++kind)
        {
            const uint64_t count = g_waitCount[kind].exchange(0u, std::memory_order_relaxed);
            const double seconds = static_cast<double>(g_waitNanoseconds[kind].exchange(0u, std::memory_order_relaxed)) * 1e-9;
            if (count == 0u || used >= static_cast<int>(sizeof(line)))
                continue;
            used += std::snprintf(line + used, sizeof(line) - static_cast<size_t>(used), "%s %s %.0f%% (%.0f/s, %.1f ms each)",
                                  any ? "," : "", kWaitNames[kind], 100.0 * seconds / window,
                                  static_cast<double>(count) / window, 1000.0 * seconds / static_cast<double>(count));
            any = true;
        }
        if (any)
            std::fprintf(stderr, "%s\n", line);
    }

    struct Worker
    {
        GS *gs = nullptr;
        std::thread thread;
        std::mutex mutex;
        std::condition_variable workReady;
        std::condition_variable workDone;
        struct Packet
        {
            std::vector<uint8_t> data;
            GsWorker::Display vblank; // tick not 0: no packet, a vblank of the EE's
            uint64_t interruptMark = 0; // not 0: no packet, the mark after one that raises an interrupt
        };
        std::deque<Packet> queue;
        std::vector<std::vector<uint8_t>> spare; // buffers to reuse
        GsWorker::Display vblankReached;
        uint64_t interruptMarkReached = 0;
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
            std::deque<Packet> taken;
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
                for (const Packet &packet : taken)
                {
                    if (packet.vblank.tick != 0u || packet.interruptMark != 0u)
                    {
                        {
                            std::lock_guard<std::mutex> guard(mutex);
                            if (packet.vblank.tick != 0u)
                                vblankReached = packet.vblank;
                            else
                                interruptMarkReached = packet.interruptMark;
                        }
                        workDone.notify_all();
                        continue;
                    }
                    gs->processGIFPacket(packet.data.data(), static_cast<uint32_t>(packet.data.size()));
                    bytes += packet.data.size();
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
                        std::fprintf(stderr, "[gs-worker] busy %.0f%% of the time, %.0f packets/s; the EE waited %.0f%% of the time for VU1 and %.0f%% for the GS, %.0f waits/s\n",
                                     100.0 * busySeconds / window, static_cast<double>(packets) / window,
                                     100.0 * static_cast<double>(g_drainVu1Nanoseconds.exchange(0)) * 1e-9 / window,
                                     100.0 * static_cast<double>(g_drainGsNanoseconds.exchange(0)) * 1e-9 / window,
                                     static_cast<double>(g_drainWaits.exchange(0)) / window);
                        printWaits(window);
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
                    if (spare.size() < 256u && !taken.front().data.empty())
                        spare.push_back(std::move(taken.front().data));
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
    std::atomic<uint32_t> g_lagVBlanks{0u}; // what the game asked for
    std::atomic<const GSRegisters *> g_registers{nullptr};
    std::atomic<uint64_t> g_interruptMarks{0u}; // marks posted (postInterruptMark)
}

void ps2x::setGsLag(uint32_t vblanks)
{
    g_lagVBlanks.store(vblanks, std::memory_order_relaxed);
}

namespace GsWorker
{
    void setInterruptSentHook(void (*hook)())
    {
        g_interruptSentHook.store(hook, std::memory_order_release);
    }

    void interruptSent()
    {
        // What reads CSR later waits for this packet and no further.
        if (lagVBlanks() != 0u)
            postInterruptMark(g_interruptMarks.fetch_add(1u, std::memory_order_acq_rel) + 1u);
        if (lagsInterrupts())
            return;
        if (void (*hook)() = g_interruptSentHook.load(std::memory_order_acquire))
            hook();
        else
            drain(Wait::Interrupt);
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

    void start(GS &gs, const GSRegisters *registers)
    {
        Worker &w = worker();
        if (w.active.load(std::memory_order_acquire))
            return;
        w.gs = &gs;
        g_registers.store(registers, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lock(w.mutex);
            w.vblankReached = {};
            w.interruptMarkReached = 0u;
            g_interruptMarks.store(0u, std::memory_order_release);
        }
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
        g_registers.store(nullptr, std::memory_order_release);
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
            w.queue.push_back(Worker::Packet{std::move(packet), {}, 0u});
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

    uint32_t lagVBlanks()
    {
        static const int64_t fromEnvironment = []() -> int64_t
        {
            const char *value = std::getenv("PS2X_GS_LAG");
            return value != nullptr && value[0] != '\0' ? static_cast<int64_t>(std::strtoul(value, nullptr, 10)) : -1;
        }();
        return fromEnvironment >= 0 ? static_cast<uint32_t>(fromEnvironment) : g_lagVBlanks.load(std::memory_order_relaxed);
    }

    bool lagsInterrupts()
    {
        if (lagVBlanks() == 0u)
            return false;
        const GSRegisters *registers = g_registers.load(std::memory_order_acquire);
        return registers != nullptr && ((registers->imr >> 8) & 0x3u) == 0x3u;
    }

    bool shownDisplay(Display &display)
    {
        if (lagVBlanks() == 0u)
            return false;
        Worker &w = worker();
        std::lock_guard<std::mutex> lock(w.mutex);
        display = w.vblankReached;
        return display.tick != 0u;
    }

    void postVBlank(const Display &display)
    {
        // Behind what VU1 has still to send.
        if (Vu1Worker::defers())
        {
            Vu1Worker::postVBlank(display);
            return;
        }
        Worker &w = worker();
        const bool queued = w.active.load(std::memory_order_acquire) && std::this_thread::get_id() != w.threadId;
        bool wasEmpty = false;
        {
            std::lock_guard<std::mutex> lock(w.mutex);
            if (queued)
            {
                wasEmpty = w.queue.empty();
                w.queue.push_back(Worker::Packet{{}, display, 0u});
                ++w.posted;
            }
            else
            {
                w.vblankReached = display;
            }
        }
        if (!queued)
            w.workDone.notify_all();
        else if (wasEmpty)
            w.workReady.notify_one();
    }

    void postInterruptMark(uint64_t mark)
    {
        // Behind what VU1 has still to send.
        if (Vu1Worker::defers())
        {
            Vu1Worker::postInterruptMark(mark);
            return;
        }
        Worker &w = worker();
        const bool queued = w.active.load(std::memory_order_acquire) && std::this_thread::get_id() != w.threadId;
        bool wasEmpty = false;
        {
            std::lock_guard<std::mutex> lock(w.mutex);
            if (queued)
            {
                wasEmpty = w.queue.empty();
                w.queue.push_back(Worker::Packet{{}, {}, mark});
                ++w.posted;
            }
            else
            {
                w.interruptMarkReached = mark;
            }
        }
        if (!queued)
            w.workDone.notify_all();
        else if (wasEmpty)
            w.workReady.notify_one();
    }

    bool settleInterrupts(Wait why)
    {
        if (lagVBlanks() == 0u)
            return drain(why);
        Worker &w = worker();
        if (std::this_thread::get_id() == w.threadId || Vu1Worker::onWorkerThread())
            return false;
        const uint64_t mark = g_interruptMarks.load(std::memory_order_acquire);
        const bool wantStats = statsWanted();
        const auto start = wantStats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        {
            std::unique_lock<std::mutex> lock(w.mutex);
            if (w.interruptMarkReached >= mark)
                return false;
            w.workDone.wait(lock, [&]
                            { return w.interruptMarkReached >= mark; });
        }
        if (wantStats)
            countWait(why, start);
        return true;
    }

    bool waitForVBlank(uint64_t tick, Wait why)
    {
        Worker &w = worker();
        if (std::this_thread::get_id() == w.threadId || Vu1Worker::onWorkerThread())
            return false;
        const bool wantStats = statsWanted();
        const auto start = wantStats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        {
            std::unique_lock<std::mutex> lock(w.mutex);
            if (w.vblankReached.tick >= tick)
                return false;
            w.workDone.wait(lock, [&]
                            { return w.vblankReached.tick >= tick; });
        }
        if (wantStats)
            countWait(why, start);
        return true;
    }

    bool drainVu1(Wait why)
    {
        if (!statsWanted())
            return Vu1Worker::drain();
        const auto start = std::chrono::steady_clock::now();
        const bool waited = Vu1Worker::drain();
        if (waited)
            countWait(why, start);
        return waited;
    }

    bool drain(Wait why)
    {
        Worker &w = worker();
        if (std::this_thread::get_id() == w.threadId || Vu1Worker::onWorkerThread())
            return false;
        const bool wantStats = statsWanted();
        const auto start = wantStats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        // VU1 produces GS packets: it has to finish before the GS can.
        const bool waitedForVu1 = Vu1Worker::drain();
        const auto afterVu1 = wantStats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (wantStats && waitedForVu1)
            g_drainVu1Nanoseconds.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(afterVu1 - start).count()));
        bool waited = waitedForVu1;
        if (w.active.load(std::memory_order_acquire))
        {
            std::unique_lock<std::mutex> lock(w.mutex);
            if (w.completed != w.posted)
            {
                const uint64_t target = w.posted;
                w.workDone.wait(lock, [&]
                                { return w.completed >= target; });
                waited = true;
                if (wantStats)
                {
                    g_drainWaits.fetch_add(1u);
                    g_drainGsNanoseconds.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - afterVu1).count()));
                }
            }
        }
        if (wantStats && waited)
            countWait(why, start);
        return waited;
    }
}
