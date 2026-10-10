#include "runtime/vu1_worker.h"
#include "runtime/ps2_memory.h"
#include "runtime/gs/gs_worker.h"

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

namespace
{
    struct Item
    {
        bool path3 = false;
        std::vector<uint8_t> data;
    };

    struct Worker
    {
        PS2Memory *memory = nullptr;
        std::thread thread;
        std::mutex mutex;
        std::condition_variable workReady;
        std::condition_variable workDone;
        std::deque<Item> queue;
        std::vector<std::vector<uint8_t>> spare;
        size_t queuedBytes = 0;
        uint64_t posted = 0;
        uint64_t completed = 0;
        bool quit = false;
        std::atomic<bool> active{false};
        std::thread::id threadId;

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
                Item item = std::move(queue.front());
                queue.pop_front();
                queuedBytes -= item.data.size();
                lock.unlock();

                const uint32_t size = static_cast<uint32_t>(item.data.size());
                if (item.path3)
                    memory->submitGifPacket(GifPathId::Path3, item.data.data(), size, true);
                else
                    memory->processVIF1Data(item.data.data(), size);

                lock.lock();
                ++completed;
                if (spare.size() < 64u)
                    spare.push_back(std::move(item.data));
                workDone.notify_all();
            }
        }

        void post(bool path3, const uint8_t *data, uint32_t sizeBytes)
        {
            std::unique_lock<std::mutex> lock(mutex);
            workDone.wait(lock, [&]
                          { return queuedBytes < kMaxQueuedBytes; });
            Item item;
            item.path3 = path3;
            if (!spare.empty())
            {
                item.data = std::move(spare.back());
                spare.pop_back();
            }
            item.data.assign(data, data + sizeBytes);
            queuedBytes += item.data.size();
            queue.push_back(std::move(item));
            ++posted;
            lock.unlock();
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
        // Off unless asked for: see the header.
        const char *value = std::getenv("PS2X_VU_THREAD");
        if (value == nullptr || value[0] == '\0' || value[0] == '0')
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
        worker().post(false, data, sizeBytes);
        if (GsWorker::mayRaiseInterrupt(data, sizeBytes))
            GsWorker::drain();
    }

    void postPath3(const uint8_t *data, uint32_t sizeBytes)
    {
        if (!data || sizeBytes == 0u)
            return;
        worker().post(true, data, sizeBytes);
        if (GsWorker::raisesInterrupt(data, sizeBytes))
            GsWorker::drain();
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
