#pragma once

// A joinable thread with a stack of a chosen size. std::thread takes the
// platform's default, 1 MB on Windows and 512 KB on macOS, and the EE
// executor runs a guest call as a host call: a game that recurses deeply (a
// script interpreter that calls itself for every call the script makes)
// needs far more host stack than it has guest stack. The size is address
// space that is reserved; memory is committed as the stack grows.

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <utility>

#if defined(_WIN32)
#include <process.h>

extern "C"
{
    __declspec(dllimport) unsigned long __stdcall WaitForSingleObject(void *handle, unsigned long milliseconds);
    __declspec(dllimport) int __stdcall CloseHandle(void *handle);
}
#else
#include <pthread.h>
#endif

class LargeStackThread
{
public:
    LargeStackThread(size_t stackBytes, std::function<void()> body)
    {
        auto *start = new std::function<void()>(std::move(body));
#if defined(_WIN32)
        constexpr unsigned kStackSizeIsAReservation = 0x00010000u;
        m_handle = reinterpret_cast<void *>(_beginthreadex(
            nullptr, static_cast<unsigned>(stackBytes), &LargeStackThread::entry, start, kStackSizeIsAReservation, nullptr));
        m_joinable = m_handle != nullptr;
#else
        pthread_attr_t attributes;
        pthread_attr_init(&attributes);
        pthread_attr_setstacksize(&attributes, stackBytes);
        m_joinable = pthread_create(&m_thread, &attributes, &LargeStackThread::entry, start) == 0;
        pthread_attr_destroy(&attributes);
#endif
        if (!m_joinable)
        {
            delete start;
            throw std::runtime_error("LargeStackThread: the thread could not be created");
        }
    }

    LargeStackThread(const LargeStackThread &) = delete;
    LargeStackThread &operator=(const LargeStackThread &) = delete;

    ~LargeStackThread()
    {
        join();
    }

    [[nodiscard]] bool joinable() const noexcept
    {
        return m_joinable;
    }

    void join()
    {
        if (!m_joinable)
        {
            return;
        }
        m_joinable = false;
#if defined(_WIN32)
        WaitForSingleObject(m_handle, 0xFFFFFFFFul);
        CloseHandle(m_handle);
        m_handle = nullptr;
#else
        pthread_join(m_thread, nullptr);
#endif
    }

private:
#if defined(_WIN32)
    static unsigned __stdcall entry(void *argument)
#else
    static void *entry(void *argument)
#endif
    {
        auto *start = static_cast<std::function<void()> *>(argument);
        (*start)();
        delete start;
#if defined(_WIN32)
        return 0u;
#else
        return nullptr;
#endif
    }

#if defined(_WIN32)
    void *m_handle = nullptr;
#else
    pthread_t m_thread{};
#endif
    bool m_joinable = false;
};
