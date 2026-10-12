#include "runtime/vu1_worker.h"
#include "runtime/ps2_memory.h"
#include "runtime/gs/gs_worker.h"
#include "../ps2_record_queue.h"

#include <atomic>
#include <cstring>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

namespace
{
    // What is queued: VIF1 data, a PATH3 GIF packet, or one of the two the GS
    // worker is to be handed in its turn (GsWorker::postVBlank,
    // postInterruptMark).
    enum Kind : uint32_t
    {
        kVif1,
        kPath3,
        kVBlank,
        kInterruptMark,
    };

    struct Worker
    {
        PS2Memory *memory = nullptr;
        std::thread thread;
        std::mutex mutex;
        std::condition_variable workReady;
        std::condition_variable workDone;
        RecordQueue queue;
        size_t queuedBytes = 0;
        uint64_t posted = 0;
        uint64_t completed = 0;
        bool quit = false;
        std::atomic<bool> active{false};
        std::thread::id threadId;

        static constexpr size_t kMaxQueuedBytes = 32u * 1024u * 1024u;

        void loop()
        {
            // Everything queued is taken in one go, as the GS worker takes
            // its own.
            RecordQueue taken;
            std::unique_lock<std::mutex> lock(mutex);
            for (;;)
            {
                workReady.wait(lock, [&]
                               { return quit || !queue.empty(); });
                if (queue.empty())
                    return;
                std::swap(taken, queue);
                lock.unlock();

                size_t bytes = 0;
                RecordQueue::Record record;
                for (RecordQueue::Reader reader = taken.read(); reader.next(record);)
                {
                    switch (record.kind)
                    {
                    case kVBlank:
                    {
                        GsWorker::Display display;
                        std::memcpy(&display, record.data, sizeof(display));
                        GsWorker::postVBlank(display);
                        break;
                    }
                    case kInterruptMark:
                        GsWorker::postInterruptMark(record.value);
                        break;
                    case kPath3:
                        memory->submitGifPacket(GifPathId::Path3, record.data, record.size, true);
                        bytes += record.size;
                        break;
                    default:
                        memory->processVIF1Data(record.data, record.size);
                        bytes += record.size;
                        break;
                    }
                }

                lock.lock();
                completed += taken.count;
                queuedBytes -= bytes;
                taken.clear();
                workDone.notify_all();
            }
        }

        void post(uint32_t kind, uint64_t value, const uint8_t *data, uint32_t sizeBytes)
        {
            bool wasEmpty = false;
            {
                std::unique_lock<std::mutex> lock(mutex);
                if (kind == kVif1 || kind == kPath3)
                {
                    workDone.wait(lock, [&]
                                  { return queuedBytes < kMaxQueuedBytes; });
                    queuedBytes += sizeBytes;
                }
                wasEmpty = queue.empty();
                queue.append(kind, value, data, sizeBytes);
                ++posted;
            }
            // The worker only sleeps on an empty queue.
            if (wasEmpty)
                workReady.notify_one();
        }
    };

    Worker &worker()
    {
        static Worker instance;
        return instance;
    }
}

namespace Vu1Worker
{
    void start(PS2Memory &memory)
    {
        Worker &w = worker();
        if (w.active.load(std::memory_order_acquire))
            return;
        w.memory = &memory;
        // On where there are cores for it (the EE, this, the GS thread and
        // its raster pool); PS2X_VU_THREAD=0 or 1 decides otherwise.
        bool wanted = std::thread::hardware_concurrency() >= 6u;
        if (const char *value = std::getenv("PS2X_VU_THREAD"); value != nullptr && value[0] != '\0')
            wanted = value[0] != '0';
        if (!wanted)
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

    bool onWorkerThread()
    {
        Worker &w = worker();
        return w.active.load(std::memory_order_acquire) && std::this_thread::get_id() == w.threadId;
    }

    bool defers()
    {
        Worker &w = worker();
        return w.active.load(std::memory_order_acquire) && std::this_thread::get_id() != w.threadId;
    }

    // A FINISH or SIGNAL in what the EE sends is pending when the send
    // returns, as it was when everything ran inline: the EE waits for the
    // pipeline to get that far.
    void postVif1(const uint8_t *data, uint32_t sizeBytes)
    {
        if (!data || sizeBytes == 0u)
            return;
        worker().post(kVif1, 0u, data, sizeBytes);
        if (GsWorker::mayRaiseInterrupt(data, sizeBytes))
            GsWorker::interruptSent();
    }

    void postPath3(const uint8_t *data, uint32_t sizeBytes)
    {
        if (!data || sizeBytes == 0u)
            return;
        worker().post(kPath3, 0u, data, sizeBytes);
        if (GsWorker::raisesInterrupt(data, sizeBytes))
            GsWorker::interruptSent();
    }

    void postVBlank(const GsWorker::Display &display)
    {
        worker().post(kVBlank, 0u, reinterpret_cast<const uint8_t *>(&display), sizeof(display));
    }

    void postInterruptMark(uint64_t mark)
    {
        worker().post(kInterruptMark, mark, nullptr, 0u);
    }

    bool drain()
    {
        Worker &w = worker();
        if (!w.active.load(std::memory_order_acquire) || std::this_thread::get_id() == w.threadId)
            return false;
        std::unique_lock<std::mutex> lock(w.mutex);
        if (w.completed == w.posted)
            return false;
        const uint64_t target = w.posted;
        w.workDone.wait(lock, [&]
                        { return w.completed >= target; });
        return true;
    }
}
