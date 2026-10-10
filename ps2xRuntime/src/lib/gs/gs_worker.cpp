#include "runtime/gs/gs_worker.h"
#include "runtime/gs/gs_frontend.h"

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
