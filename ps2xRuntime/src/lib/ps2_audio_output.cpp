#include "runtime/ps2_audio_output.h"
#include "ps2_host_backend.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace
{
    constexpr uint32_t kSampleRate = 48000u;
    constexpr uint64_t kQueueFrames = 1u << 15; // 0.68 s
    // What the queue is kept at: the IOP's frames come a picture's worth at a
    // time (800), so this is three of those.
    constexpr uint64_t kTargetFrames = 2400u;
    // Further behind than this the game is running ahead of the device (an
    // unpaced run): the device skips to the newest.
    constexpr uint64_t kLateFrames = 9600u;
    constexpr double kMostCorrection = 0.005;

    // One writer (the IOP's thread) and one reader (the device's): the writer
    // owns `written`, the reader `taken`, both count frames since the start.
    std::array<int16_t, kQueueFrames * 2u> g_queue;
    std::atomic<uint64_t> g_written{0};
    std::atomic<uint64_t> g_taken{0};
    std::atomic<bool> g_started{false};

    // The reader's own.
    bool g_playing = false;
    double g_position = 0.0; // of the next frame out, past `taken`
    double g_level = static_cast<double>(kTargetFrames);

#if !defined(PLATFORM_VITA)
    AudioStream g_stream{};
#endif
    // PS2X_AUDIO_STATS=1: every ten seconds of the game's sound, how the
    // device kept up with it.
    std::atomic<uint32_t> g_ranDry{0};
    std::atomic<uint32_t> g_skipped{0};
    std::atomic<uint32_t> g_levelNow{0};
    bool g_stats = false;

    // The dump is written by the IOP's thread and closed by the window's.
    std::mutex g_dumpMutex;
    std::atomic<bool> g_dumping{false};
    std::FILE *g_dump = nullptr;
    uint32_t g_dumpBytes = 0;
    // The sizes in the dump's header until it is closed: a run that is cut
    // off leaves a file that still reads to its end.
    constexpr uint32_t kOpenDumpBytes = 0x7FFFF000u;

    inline int16_t queued(uint64_t frame, unsigned channel)
    {
        return g_queue[(frame & (kQueueFrames - 1u)) * 2u + channel];
    }

    void deviceWants(void *buffer, unsigned int frames)
    {
        int16_t *out = static_cast<int16_t *>(buffer);
        const uint64_t written = g_written.load(std::memory_order_acquire);
        uint64_t taken = g_taken.load(std::memory_order_relaxed);
        uint64_t waiting = written - taken;
        if (waiting > kLateFrames)
        {
            taken = written - kTargetFrames;
            waiting = kTargetFrames;
            g_position = 0.0;
            g_level = static_cast<double>(kTargetFrames);
            g_skipped.fetch_add(1u, std::memory_order_relaxed);
        }
        if (!g_playing)
        {
            // Nothing until there is the usual amount to play from, so that
            // a game that has fallen behind is heard in pieces as long as
            // they can be made.
            if (waiting < kTargetFrames)
            {
                std::memset(out, 0, static_cast<size_t>(frames) * 2u * sizeof(int16_t));
                return;
            }
            g_playing = true;
            g_position = 0.0;
            g_level = static_cast<double>(waiting);
        }

        g_level += (static_cast<double>(waiting) - g_level) * 0.02;
        g_levelNow.store(static_cast<uint32_t>(g_level), std::memory_order_relaxed);
        const double off = std::clamp((g_level - static_cast<double>(kTargetFrames)) / static_cast<double>(kTargetFrames), -1.0, 1.0);
        const double step = 1.0 + off * kMostCorrection;

        unsigned int made = 0;
        for (; made < frames; ++made)
        {
            const uint64_t whole = static_cast<uint64_t>(g_position);
            if (whole + 1u >= waiting)
                break;
            const double part = g_position - static_cast<double>(whole);
            for (unsigned channel = 0; channel < 2u; ++channel)
            {
                const double a = queued(taken + whole, channel), b = queued(taken + whole + 1u, channel);
                out[made * 2u + channel] = static_cast<int16_t>(a + (b - a) * part);
            }
            g_position += step;
        }
        const uint64_t used = std::min(static_cast<uint64_t>(g_position), waiting);
        g_position -= static_cast<double>(used);
        g_taken.store(taken + used, std::memory_order_release);
        if (made < frames)
        {
            // Run dry: the rest is silence, and it waits to fill again.
            std::memset(out + made * 2u, 0, static_cast<size_t>(frames - made) * 2u * sizeof(int16_t));
            g_playing = false;
            g_ranDry.fetch_add(1u, std::memory_order_relaxed);
        }
    }

    void put32(uint8_t *at, uint32_t value)
    {
        at[0] = static_cast<uint8_t>(value);
        at[1] = static_cast<uint8_t>(value >> 8);
        at[2] = static_cast<uint8_t>(value >> 16);
        at[3] = static_cast<uint8_t>(value >> 24);
    }

    void writeDumpHeader(uint32_t bytes)
    {
        uint8_t header[44] = {'R', 'I', 'F', 'F', 0, 0, 0, 0, 'W', 'A', 'V', 'E', 'f', 'm', 't', ' ',
                              16, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              4, 0, 16, 0, 'd', 'a', 't', 'a', 0, 0, 0, 0};
        put32(header + 4, 36u + bytes);
        put32(header + 24, kSampleRate);
        put32(header + 28, kSampleRate * 4u);
        put32(header + 40, bytes);
        std::fseek(g_dump, 0, SEEK_SET);
        std::fwrite(header, 1, sizeof(header), g_dump);
    }
}

namespace ps2x::audio
{
    void startOutput()
    {
        if (const char *stats = std::getenv("PS2X_AUDIO_STATS"))
            g_stats = std::atoi(stats) != 0;
        if (const char *path = std::getenv("PS2X_AUDIO_DUMP"); path && *path && !g_dumping.load())
        {
            std::lock_guard<std::mutex> lock(g_dumpMutex);
            g_dump = std::fopen(path, "wb");
            if (g_dump)
            {
                writeDumpHeader(kOpenDumpBytes);
                g_dumping.store(true);
            }
        }
#if !defined(PLATFORM_VITA)
        // An unpaced run (PS2X_TURBO) makes sound faster than it can be
        // played: it is not, and the dump has all of it.
        const char *turbo = std::getenv("PS2X_TURBO");
        if (g_started.load() || !IsAudioDeviceReady() || (turbo && turbo[0] != '\0' && turbo[0] != '0'))
            return;
        g_stream = LoadAudioStream(kSampleRate, 16, 2);
        if (!IsAudioStreamValid(g_stream))
            return;
        SetAudioStreamCallback(g_stream, deviceWants);
        PlayAudioStream(g_stream);
        g_started.store(true);
#endif
    }

    void stopOutput()
    {
#if !defined(PLATFORM_VITA)
        if (g_started.exchange(false))
        {
            StopAudioStream(g_stream);
            UnloadAudioStream(g_stream);
            g_stream = {};
        }
#endif
        if (g_dumping.exchange(false))
        {
            std::lock_guard<std::mutex> lock(g_dumpMutex);
            writeDumpHeader(g_dumpBytes);
            std::fclose(g_dump);
            g_dump = nullptr;
        }
    }

    void pushOutput(const int16_t *samples, size_t frames)
    {
        if (g_dumping.load(std::memory_order_relaxed))
        {
            std::lock_guard<std::mutex> lock(g_dumpMutex);
            if (g_dump)
                g_dumpBytes += static_cast<uint32_t>(std::fwrite(samples, 1, frames * 4u, g_dump));
        }
        uint64_t written = g_written.load(std::memory_order_relaxed);
        for (size_t frame = 0; frame < frames; ++frame, ++written)
        {
            const size_t at = static_cast<size_t>(written & (kQueueFrames - 1u)) * 2u;
            g_queue[at] = samples[frame * 2u];
            g_queue[at + 1u] = samples[frame * 2u + 1u];
        }
        g_written.store(written, std::memory_order_release);
        if (g_stats && written / (kSampleRate * 10u) != (written - frames) / (kSampleRate * 10u))
        {
            std::fprintf(stderr, "[audio] %llu s made, %lld frames waiting (about %u), ran dry %u times, skipped ahead %u times\n",
                         static_cast<unsigned long long>(written / kSampleRate),
                         static_cast<long long>(written - g_taken.load(std::memory_order_relaxed)),
                         g_levelNow.load(std::memory_order_relaxed),
                         g_ranDry.exchange(0u, std::memory_order_relaxed),
                         g_skipped.exchange(0u, std::memory_order_relaxed));
        }
    }
}
