#include "ps2_runtime.h"
#include "runtime/gs/gs_worker.h"
#include "runtime/vu1_worker.h"
#include "ps2_log.h"
#include "ps2_stubs.h"
#include "ps2_syscalls.h"
#include "game_overrides.h"
#include "ps2_runtime_macros.h"
#include "runtime/gs/gs_frontend.h"
#include "runtime/ee_scheduler.h"
#include "runtime/ps2_audio_output.h"
#include "runtime/ps2_display_mode.h"
#include "runtime/ps2_settings.h"
#include "runtime/ps2_settings_menu.h"
#include "runtime/gs/gs_gpu_renderer.h"
#include "ThreadNaming.h"
#include "LargeStackThread.h"
#include "Kernel/Stubs/Audio.h"
#include "Kernel/Stubs/GS.h"
#include "Kernel/Stubs/MPEG.h"
#include "ps2_host_backend.h"
#include "ps2_iop_host.h"
#include "ps2x/iop/iop_subsystem.h"

#include <iostream>
#include <stdexcept>
#include <fstream>
#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <limits>
#include <chrono>
#include <atomic>
#include <thread>
#include <unordered_map>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace ps2_stubs
{
    void resetSifState();
}

#define ELF_MAGIC 0x464C457F // "\x7FELF" in little endian
#define ET_EXEC 2            // Executable file
#define EM_MIPS 8            // MIPS architecture
#define PT_LOAD 1            // Loadable segment

static constexpr int FB_WIDTH = 640;
static constexpr int FB_HEIGHT = 512;
static constexpr int DEFAULT_DISPLAY_HEIGHT = 448;
static constexpr uint32_t DEFAULT_FB_SIZE = FB_WIDTH * FB_HEIGHT * 4;
static constexpr uint32_t DEFAULT_FB_ADDR = (PS2_RAM_SIZE - DEFAULT_FB_SIZE - 0x10000u);
#if defined(PLATFORM_VITA)
static constexpr int HOST_WINDOW_WIDTH = 960;
static constexpr int HOST_WINDOW_HEIGHT = 544;
#else
static constexpr int HOST_WINDOW_WIDTH = FB_WIDTH;
static constexpr int HOST_WINDOW_HEIGHT = DEFAULT_DISPLAY_HEIGHT;
#endif
// How the game's picture is put on the host's screen, from the settings
// file's [display] section or the environment:
//   aspect     (PS2X_ASPECT)      4:3 as on a television (the default), 16:9
//                                 and so on, or "native": one host pixel
//                                 for each of the game's, which is 640:448.
//   window     (PS2X_WINDOW)      the window's size, as 1280x720.
//   fullscreen (PS2X_FULLSCREEN)  off: a window. borderless: a window
//                                 without a frame over the whole screen.
//                                 exclusive: the screen taken over in its
//                                 own mode. on is borderless. F11 switches
//                                 while the game runs (ps2_display_mode.h).
//   filter     (PS2X_FILTER)      linear (smooth, the default) or nearest.
struct HostDisplay
{
    float aspect = 4.0f / 3.0f; // width over height as shown; 0 for the picture's own
    int windowWidth = 0;
    int windowHeight = 0;
    ps2x::DisplayMode mode = ps2x::DisplayMode::Window;
    bool smooth = true;
};

static const HostDisplay &hostDisplay()
{
    static const HostDisplay display = []
    {
        HostDisplay d;
        float first = 0.0f, second = 0.0f;
        if (const char *aspect = ps2x::setting("PS2X_ASPECT", "display.aspect"))
        {
            if (std::strcmp(aspect, "native") == 0)
                d.aspect = 0.0f;
            else if (ps2x::settingPair("PS2X_ASPECT", "display.aspect", first, second))
                d.aspect = first / second;
        }
        if (ps2x::settingPair("PS2X_WINDOW", "display.window", first, second) && first >= 64.0f && second >= 64.0f)
        {
            d.windowWidth = static_cast<int>(first);
            d.windowHeight = static_cast<int>(second);
        }
        else
        {
            // Twice the game's height or so, and as wide as the picture is.
            d.windowHeight = d.aspect > 0.0f ? 720 : HOST_WINDOW_HEIGHT;
            d.windowWidth = d.aspect > 0.0f ? static_cast<int>(720.0f * d.aspect + 0.5f) : HOST_WINDOW_WIDTH;
        }
        d.mode = ps2x::displayModeFrom(ps2x::setting("PS2X_FULLSCREEN", "display.fullscreen"), ps2x::DisplayMode::Window);
        if (const char *filter = ps2x::setting("PS2X_FILTER", "display.filter"))
            d.smooth = std::strcmp(filter, "nearest") != 0;
        return d;
    }();
    return display;
}

const char *ps2x::displayModeValue(DisplayMode mode)
{
    switch (mode)
    {
    case DisplayMode::Borderless:
        return "borderless";
    case DisplayMode::Exclusive:
        return "exclusive";
    default:
        return "off";
    }
}

ps2x::DisplayMode ps2x::displayModeFrom(const char *text, DisplayMode otherwise)
{
    if (!text)
        return otherwise;
    std::string value;
    for (const char *at = text; *at != '\0'; ++at)
        if (!std::isspace(static_cast<unsigned char>(*at)))
            value.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(*at))));
    if (value == "off" || value == "no" || value == "false" || value == "0" || value == "window")
        return DisplayMode::Window;
    // "on" is from when the window had one other state, and that was this.
    if (value == "borderless" || value == "on" || value == "yes" || value == "true" || value == "1")
        return DisplayMode::Borderless;
    if (value == "exclusive" || value == "fullscreen")
        return DisplayMode::Exclusive;
    return otherwise;
}

namespace
{
    // Which of the two F11 goes to, and the window as it was before it
    // became the screen (raylib gives the screen's mode the window's size,
    // so the window is made the monitor's size first).
    ps2x::DisplayMode g_wholeScreenMode = ps2x::DisplayMode::Borderless;
    int g_windowedWidth = 0, g_windowedHeight = 0;
}

ps2x::DisplayMode ps2x::displayMode()
{
#if !defined(PLATFORM_VITA)
    if (IsWindowFullscreen())
        return DisplayMode::Exclusive;
    if (IsWindowState(FLAG_BORDERLESS_WINDOWED_MODE))
        return DisplayMode::Borderless;
#endif
    return DisplayMode::Window;
}

void ps2x::setDisplayMode(DisplayMode mode)
{
#if !defined(PLATFORM_VITA)
    if (mode != DisplayMode::Window)
        g_wholeScreenMode = mode;
    const DisplayMode now = displayMode();
    if (mode == now)
        return;

    // Each of the other two is entered from a window.
    if (now == DisplayMode::Borderless)
    {
        ToggleBorderlessWindowed();
    }
    else if (now == DisplayMode::Exclusive)
    {
        ToggleFullscreen();
        if (g_windowedWidth > 0 && g_windowedHeight > 0)
            SetWindowSize(g_windowedWidth, g_windowedHeight);
    }

    if (mode == DisplayMode::Borderless)
    {
        ToggleBorderlessWindowed();
    }
    else if (mode == DisplayMode::Exclusive)
    {
        const int monitor = GetCurrentMonitor();
        g_windowedWidth = GetScreenWidth();
        g_windowedHeight = GetScreenHeight();
#if defined(__APPLE__)
        // macOS: a window cannot be given the monitor's size first. The
        // desktop keeps it under the menu bar, and the size it comes back
        // with is then the mode raylib asks for. Nor is the desktop's own
        // mode on offer: a "Retina" desktop counts in points, the modes
        // are in the panel's pixels, and anything asked in points gets
        // the smallest of them. So the screen is taken over first, and
        // then asked for the mode of its own pixels; in full screen the
        // size goes to the monitor, and raylib keeps what it is told.
        const Vector2 scale = GetWindowScaleDPI();
        const int width = static_cast<int>(static_cast<float>(GetMonitorWidth(monitor)) * scale.x + 0.5f);
        const int height = static_cast<int>(static_cast<float>(GetMonitorHeight(monitor)) * scale.y + 0.5f);
        ToggleFullscreen();
        SetWindowSize(width, height);
        // What it was given, should that not be what was asked.
        SetWindowSize(GetMonitorWidth(monitor), GetMonitorHeight(monitor));
#else
        SetWindowSize(GetMonitorWidth(monitor), GetMonitorHeight(monitor));
        ToggleFullscreen();
#endif
    }
#else
    (void)mode;
#endif
}

void ps2x::toggleDisplayMode()
{
    setDisplayMode(displayMode() == DisplayMode::Window ? g_wholeScreenMode : DisplayMode::Window);
}

static std::atomic<float> g_presentationAspect{0.0f};

void ps2x::setPresentationAspect(float widthOverHeight)
{
    g_presentationAspect.store(widthOverHeight > 0.0f ? widthOverHeight : 0.0f, std::memory_order_relaxed);
}

// -1 for what [display] filter says.
static std::atomic<int> g_displayFilter{-1};

void ps2x::setDisplayFilter(bool smooth)
{
    g_displayFilter.store(smooth ? 1 : 0, std::memory_order_relaxed);
}

static bool displaySmooth()
{
    const int asked = g_displayFilter.load(std::memory_order_relaxed);
    return asked < 0 ? hostDisplay().smooth : asked != 0;
}

struct ElfHeader
{
    uint32_t magic;
    uint8_t elf_class;
    uint8_t endianness;
    uint8_t version;
    uint8_t os_abi;
    uint8_t abi_version;
    uint8_t padding[7];
    uint16_t type;
    uint16_t machine;
    uint32_t version2;
    uint32_t entry;
    uint32_t phoff;
    uint32_t shoff;
    uint32_t flags;
    uint16_t ehsize;
    uint16_t phentsize;
    uint16_t phnum;
    uint16_t shentsize;
    uint16_t shnum;
    uint16_t shstrndx;
};

struct ProgramHeader
{
    uint32_t type;
    uint32_t offset;
    uint32_t vaddr;
    uint32_t paddr;
    uint32_t filesz;
    uint32_t memsz;
    uint32_t flags;
    uint32_t align;
};

namespace
{
    constexpr uint32_t kGuestHeapDefaultBase = 0x00100000u;
    constexpr uint32_t kGuestHeapDefaultAlignment = 16u;
    constexpr uint32_t kGuestHeapSafetyPad = 0x1000u;
    constexpr uint32_t kGuestHeapHardLimit = 0x01F00000u;

    constexpr uint32_t COP0_CAUSE_EXCCODE_MASK = 0x0000007Cu;
    constexpr uint32_t COP0_CAUSE_BD = 0x80000000u;
    constexpr uint32_t COP0_STATUS_EXL = 0x00000002u;
    constexpr uint32_t COP0_STATUS_BEV = 0x00400000u;
    constexpr uint32_t EXCEPTION_VECTOR_GENERAL = 0x80000080u;
    constexpr uint32_t EXCEPTION_VECTOR_TLB_REFILL = 0x80000000u;
    constexpr uint32_t EXCEPTION_VECTOR_BOOT = 0xBFC00200u;

    struct DispatchHistory
    {
        std::array<uint32_t, 64> pcs{};
        uint32_t next = 0u;
        bool wrapped = false;
    };

    // What a thread running guest code keeps to itself. One object, so that
    // a guest call, which needs both, finds the thread's storage once (on
    // macOS each look is a call).
    struct GuestThreadState
    {
        DispatchHistory history;
        // Set from the moment guest code starts handing control back to the
        // scheduler (a checkpoint that is due, or a jump out of the function)
        // until the scheduler enters guest code again. While it is set,
        // ctx->pc is where to resume and says nothing about which call has
        // returned.
        bool unwinding = false;
    };
    thread_local GuestThreadState t_guest;

    // PS2X_CALL_HISTORY=1: keep the last few thousand calls, jumps and
    // returns to a call site, with the stack pointer and return address at
    // each, and print them when the guest ends up at an address that is not
    // code. For a return through a clobbered stack slot this shows which
    // call came back with the stack where it was not.
    struct CallRecord
    {
        uint32_t target, source, sp, ra;
        char kind; // 'c' call, 'j' jump, 'r' back at the call site, 'x' left another way
    };
    struct CallHistory
    {
        std::array<CallRecord, 4096> records{};
        uint32_t next = 0u;
        uint64_t count = 0u;
    };
    thread_local CallHistory g_callHistory;
    const bool g_callHistoryWanted = []
    {
        const char *value = std::getenv("PS2X_CALL_HISTORY");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();

    inline void pushCall(char kind, uint32_t target, uint32_t source, const R5900Context *ctx)
    {
        CallHistory &h = g_callHistory;
        h.records[h.next] = {target, source, static_cast<uint32_t>(_mm_extract_epi32(ctx->r[29], 0)),
                             static_cast<uint32_t>(_mm_extract_epi32(ctx->r[31], 0)), kind};
        h.next = (h.next + 1u) % static_cast<uint32_t>(h.records.size());
        ++h.count;
    }

    void printCallHistory(size_t last)
    {
        const CallHistory &h = g_callHistory;
        const size_t size = h.records.size();
        const size_t available = static_cast<size_t>(std::min<uint64_t>(h.count, size));
        last = std::min(last, available);
        int depth = 0;
        for (size_t i = 0; i < last; ++i)
        {
            const CallRecord &r = h.records[(h.next + size - last + i) % size];
            if (r.kind == 'r' || r.kind == 'x')
                depth = std::max(depth - 1, 0);
            std::fprintf(stderr, "[calls] %*s%c %08x from %08x sp=%08x ra=%08x\n", depth * 2, "", r.kind, r.target, r.source, r.sp, r.ra);
            if (r.kind == 'c')
                ++depth;
        }
    }

    bool computeFileCrc32(const std::string &path, uint32_t &crcOut)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open())
        {
            return false;
        }

        static const std::array<uint32_t, 256> table = []
        {
            std::array<uint32_t, 256> values{};
            for (uint32_t i = 0; i < values.size(); ++i)
            {
                uint32_t value = i;
                for (uint32_t bit = 0; bit < 8; ++bit)
                {
                    value = (value & 1u) ? (0xEDB88320u ^ (value >> 1u)) : (value >> 1u);
                }
                values[i] = value;
            }
            return values;
        }();

        uint32_t crc = 0xFFFFFFFFu;
        std::array<uint8_t, 16 * 1024> buffer{};
        while (file.good())
        {
            file.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
            const std::streamsize count = file.gcount();
            for (std::streamsize i = 0; i < count; ++i)
            {
                crc = table[(crc ^ buffer[static_cast<size_t>(i)]) & 0xFFu] ^ (crc >> 8u);
            }
        }
        if (file.bad())
        {
            return false;
        }
        crcOut = ~crc;
        return true;
    }

    inline void pushDispatchPc(DispatchHistory &h, uint32_t pc)
    {
        h.pcs[h.next] = pc;
        h.next = (h.next + 1u) % static_cast<uint32_t>(h.pcs.size());
        if (h.next == 0u)
        {
            h.wrapped = true;
        }
    }

    std::string formatDispatchHistory()
    {
        const DispatchHistory &h = t_guest.history;
        const uint32_t count = h.wrapped ? static_cast<uint32_t>(h.pcs.size()) : h.next;
        if (count == 0u)
        {
            return "(empty)";
        }

        std::ostringstream oss;
        bool first = true;
        for (uint32_t i = 0u; i < count; ++i)
        {
            const uint32_t idx = (h.next + h.pcs.size() - count + i) % static_cast<uint32_t>(h.pcs.size());
            if (!first)
            {
                oss << " -> ";
            }
            first = false;
            oss << "0x" << std::hex << h.pcs[idx];
        }
        return oss.str();
    }

    uint32_t selectExceptionVector(const R5900Context *ctx, bool tlbRefill)
    {
        if (ctx->cop0_status & COP0_STATUS_BEV)
        {
            return EXCEPTION_VECTOR_BOOT;
        }
        return tlbRefill ? EXCEPTION_VECTOR_TLB_REFILL : EXCEPTION_VECTOR_GENERAL;
    }

    void seedVu0IdleSuccess(R5900Context *ctx)
    {
        if (!ctx)
        {
            return;
        }

        ctx->vu0_clip_flags = 0;
        ctx->vu0_clip_flags2 = 0;
        ctx->vu0_mac_flags = 0;
        ctx->vu0_status = 0;
        ctx->vu0_q = 1.0f;
        ctx->vu0_r = _mm_castsi128_ps(_mm_set1_epi32(0x3F800000));
        ctx->vu0_vpu_stat = 0;
        ctx->vu0_vpu_stat2 = 0;
    }

    void copyVu0ContextToState(const R5900Context *ctx, VU1State &state)
    {
        std::memset(&state, 0, sizeof(state));

        for (uint32_t i = 0; i < 32u; ++i)
        {
            _mm_storeu_ps(state.vf[i], ctx->vu0_vf[i]);
        }
        for (uint32_t i = 0; i < 16u; ++i)
        {
            state.vi[i] = static_cast<int16_t>(ctx->vi[i]);
        }

        _mm_storeu_ps(state.acc, ctx->vu0_acc);
        state.q = ctx->vu0_q;
        state.p = ctx->vu0_p;
        state.i = ctx->vu0_i;
        alignas(16) uint32_t rWords[4]{};
        _mm_storeu_si128(reinterpret_cast<__m128i *>(rWords), _mm_castps_si128(ctx->vu0_r));
        state.r = 0x3F800000u | (rWords[0] & 0x007FFFFFu);
        state.pc = ctx->vu0_pc;
        state.mac = ctx->vu0_mac_flags;
        state.clip = ctx->vu0_clip_flags;
        state.status = ctx->vu0_status;
        state.itop = ctx->vu0_itop;
        state.dBitEnabled = (ctx->vu0_fbrst & (1u << 2)) != 0u;
        state.tBitEnabled = (ctx->vu0_fbrst & (1u << 3)) != 0u;

        state.vf[0][0] = 0.0f;
        state.vf[0][1] = 0.0f;
        state.vf[0][2] = 0.0f;
        state.vf[0][3] = 1.0f;
        state.vi[0] = 0;
    }

    void copyVu0StateToContext(const VU1State &state, R5900Context *ctx)
    {
        for (uint32_t i = 0; i < 32u; ++i)
        {
            ctx->vu0_vf[i] = _mm_loadu_ps(state.vf[i]);
        }
        for (uint32_t i = 0; i < 16u; ++i)
        {
            ctx->vi[i] = static_cast<uint16_t>(state.vi[i]);
        }

        ctx->vu0_acc = _mm_loadu_ps(state.acc);
        ctx->vu0_q = state.q;
        ctx->vu0_p = state.p;
        ctx->vu0_i = state.i;
        ctx->vu0_r = _mm_castsi128_ps(_mm_set1_epi32(static_cast<int32_t>(state.r)));
        ctx->vu0_mac_flags = state.mac;
        ctx->vu0_clip_flags = state.clip;
        ctx->vu0_clip_flags2 = state.clip;
        ctx->vu0_status = static_cast<uint16_t>(state.status);
        ctx->vu0_itop = state.itop;
        ctx->vu0_pc = state.pc;
        ctx->vu0_tpc = state.pc;
        ctx->vu0_vpu_stat = (ctx->vu0_vpu_stat & 0xFF00u) | (state.stoppedByD ? (1u << 1) : 0u) | (state.stoppedByT ? (1u << 2) : 0u);
        ctx->vu0_vpu_stat2 = 0;

        ctx->vu0_vf[0] = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);
        ctx->vi[0] = 0;
    }

    void raiseCop0Exception(R5900Context *ctx, uint32_t exceptionCode, bool tlbRefill = false)
    {
        if (ctx->in_delay_slot)
        {
            ctx->cop0_epc = ctx->branch_pc;
            ctx->cop0_cause = (ctx->cop0_cause & ~COP0_CAUSE_EXCCODE_MASK) |
                              ((exceptionCode << 2) & COP0_CAUSE_EXCCODE_MASK) |
                              COP0_CAUSE_BD;
        }
        else
        {
            ctx->cop0_epc = ctx->pc;
            ctx->cop0_cause = (ctx->cop0_cause & ~(COP0_CAUSE_EXCCODE_MASK | COP0_CAUSE_BD)) |
                              ((exceptionCode << 2) & COP0_CAUSE_EXCCODE_MASK);
        }

        ctx->cop0_status |= COP0_STATUS_EXL;
        ctx->pc = selectExceptionVector(ctx, tlbRefill);
        ctx->in_delay_slot = false;
    }

    std::filesystem::path normalizeAbsolutePath(const std::filesystem::path &path)
    {
        if (path.empty())
        {
            return {};
        }

#if defined(PLATFORM_VITA)
        const std::string generic = path.generic_string();
        const std::size_t colon = generic.find(':');
        if (colon != std::string::npos && colon != 0u)
        {
            const std::size_t slash = generic.find_first_of("/\\");
            if (slash == std::string::npos || colon < slash)
            {
                return path.lexically_normal();
            }
        }
#endif

        std::error_code ec;
        const std::filesystem::path absolute = std::filesystem::absolute(path, ec);
        if (ec)
        {
            return path.lexically_normal();
        }
        return absolute.lexically_normal();
    }

    PS2Runtime::IoPaths &runtimeIoPaths()
    {
        static PS2Runtime::IoPaths paths = []()
        {
            PS2Runtime::IoPaths defaults;
            std::error_code ec;
            const std::filesystem::path cwd = std::filesystem::current_path(ec);
            defaults.elfDirectory = ec ? std::filesystem::path(".") : cwd.lexically_normal();
            defaults.hostRoot = defaults.elfDirectory;
            defaults.cdRoot = defaults.elfDirectory;
            defaults.mcRoot = defaults.elfDirectory / "mc0";
            return defaults;
        }();

        return paths;
    }

    std::string readGuestPrintableString(const uint8_t *rdram, uint32_t addr, size_t maxLen)
    {
        std::string out;
        if (!rdram || maxLen == 0)
        {
            return out;
        }

        out.reserve(std::min<size_t>(maxLen, 64));
        for (size_t i = 0; i < maxLen; ++i)
        {
            const char ch = static_cast<char>(rdram[(addr + static_cast<uint32_t>(i)) & PS2_RAM_MASK]);
            if (ch == '\0')
            {
                break;
            }
            if (ch >= 0x20 && ch < 0x7F)
            {
                out.push_back(ch);
            }
            else
            {
                out.push_back('.');
            }
        }
        return out;
    }
}

// Set while a traced VU1 run is in flight; ps2_vu1_core.cpp logs XGKICK packets.
int g_ps2xVu1TraceKicks = 0;

uint32_t g_ps2xWatchBase = 0;
uint32_t g_ps2xWatchSize = 0;

namespace
{
    int g_ps2xWatchReports = 0;
    // With a fourth field, only stores of that value (its low 32 bits) count:
    // for a slot that is written all the time, such as one on the stack.
    bool g_ps2xWatchOneValue = false;
    uint32_t g_ps2xWatchValue = 0;

    // Arms PS2X_WATCH=<hex address>[:bytes[:reports[:hex value]]] before any
    // guest code runs.
    const bool g_ps2xWatchArmed = []
    {
        const char *value = std::getenv("PS2X_WATCH");
        if (!value || !value[0])
            return false;
        char *end = nullptr;
        g_ps2xWatchBase = static_cast<uint32_t>(std::strtoul(value, &end, 16)) & PS2_RAM_MASK;
        uint32_t size = 16u;
        g_ps2xWatchReports = 64;
        if (end && *end == ':')
        {
            size = static_cast<uint32_t>(std::strtoul(end + 1, &end, 0));
            if (end && *end == ':')
            {
                g_ps2xWatchReports = static_cast<int>(std::strtol(end + 1, &end, 0));
                if (end && *end == ':')
                {
                    g_ps2xWatchValue = static_cast<uint32_t>(std::strtoul(end + 1, nullptr, 16));
                    g_ps2xWatchOneValue = true;
                }
            }
        }
        g_ps2xWatchSize = size != 0u ? size : 16u;
        return true;
    }();
}

void ps2ReportWatchedWrite(uint32_t guestAddr, uint32_t size, uint64_t valueLo, uint64_t valueHi,
                           const char *op, const R5900Context *ctx)
{
    if (g_ps2xWatchOneValue && static_cast<uint32_t>(valueLo) != g_ps2xWatchValue)
        return;
    if (g_ps2xWatchReports <= 0)
    {
        g_ps2xWatchSize = 0u;
        return;
    }
    --g_ps2xWatchReports;
    std::fprintf(stderr, "[watch] %s 0x%08x (%u bytes) = 0x%016llx%016llx  pc=0x%08x ra=0x%08x\n",
                 op ? op : "write", guestAddr, size, static_cast<unsigned long long>(valueHi),
                 static_cast<unsigned long long>(valueLo), ctx ? ctx->pc : 0u,
                 ctx ? static_cast<uint32_t>(_mm_extract_epi32(ctx->r[31], 0)) : 0u);
}
extern int g_ps2xGifTraceRemaining; // ps2_memory.cpp
extern int g_ps2xVif1TraceRemaining; // ps2_vif1_interpreter.cpp
void ps2xInstallGsFinishDelay(EeScheduler &scheduler); // EeScheduler.cpp

namespace
{
    constexpr double kTestVBlanksPerSecond = 59.94; // NTSC

    // PS2X_VU1_TRACE=<secs>[:count[:skip]]  log VU1 runs starting at that
    // emulated time: the data memory the game uploaded, the registers after
    // the run and every packet it kicked. `skip` runs are passed over first.
    // PS2X_VU1_TRACE_KICKS=1 logs only the kicked packets, in full, as hex
    // (one frame's worth of geometry for offline analysis).
    struct Vu1TracePlan
    {
        uint64_t fromTick = 0;
        uint32_t remaining = 0;
        uint32_t skip = 0;
        bool kicksOnly = false;
    };

    Vu1TracePlan &vu1TracePlan()
    {
        static Vu1TracePlan plan = []
        {
            Vu1TracePlan p;
            if (const char *value = std::getenv("PS2X_VU1_TRACE"))
            {
                p.fromTick = static_cast<uint64_t>(std::atof(value) * kTestVBlanksPerSecond);
                p.remaining = 4u;
                if (const char *kicks = std::getenv("PS2X_VU1_TRACE_KICKS"))
                    p.kicksOnly = kicks[0] != '\0' && kicks[0] != '0';
                if (const char *colon = std::strchr(value, ':'))
                {
                    p.remaining = static_cast<uint32_t>(std::atoi(colon + 1));
                    if (const char *colon2 = std::strchr(colon + 1, ':'))
                        p.skip = static_cast<uint32_t>(std::atoi(colon2 + 1));
                }
            }
            return p;
        }();
        return plan;
    }

    // While a test run fast-forwards (see GS::skipDrawForFastForward) nothing
    // is rasterised, so the geometry VU1 would produce is not needed either,
    // and interpreting it is most of the cost of a 3D scene. Drawing resumes
    // some frames before each dump, which gives VU1 time to pick its state
    // back up. PS2X_FF_KEEP_VU1=1 runs it regardless.
    bool skipVu1ForFastForward(const GS &gs)
    {
        static const bool keep = []
        {
            const char *value = std::getenv("PS2X_FF_KEEP_VU1");
            return value != nullptr && value[0] != '\0' && value[0] != '0';
        }();
        return !keep && gs.skipDrawForFastForward();
    }

    bool vu1TraceBegin(uint64_t tick, const char *kind, uint32_t startPC, uint32_t top, uint32_t itop,
                       const uint8_t *data, const uint8_t *code)
    {
        Vu1TracePlan &plan = vu1TracePlan();
        if (plan.remaining == 0u || plan.fromTick == 0u || tick < plan.fromTick)
            return false;

        // PS2X_VU1_DUMP_CODE=<prefix>: the microcode in VU1 memory, written to
        // <prefix>.<n>.bin each time a traced run finds it changed (games
        // upload different programs to the same addresses).
        static uint64_t lastCodeHash = 0;
        static uint32_t codeImages = 0;
        if (code)
        {
            if (const char *prefix = std::getenv("PS2X_VU1_DUMP_CODE"))
            {
                uint64_t hash = 1469598103934665603ull;
                for (uint32_t i = 0; i < PS2_VU1_CODE_SIZE; ++i)
                    hash = (hash ^ code[i]) * 1099511628211ull;
                if (hash != lastCodeHash)
                {
                    lastCodeHash = hash;
                    const std::string path = std::string(prefix) + "." + std::to_string(codeImages) + ".bin";
                    if (FILE *file = std::fopen(path.c_str(), "wb"))
                    {
                        std::fwrite(code, 1, PS2_VU1_CODE_SIZE, file);
                        std::fclose(file);
                    }
                    std::fprintf(stderr, "[vu1] code image %u -> %s\n", codeImages, path.c_str());
                    ++codeImages;
                }
            }
        }
        if (plan.skip != 0u)
        {
            --plan.skip;
            return false;
        }
        --plan.remaining;
        std::fprintf(stderr, "[vu1] %s t=%.2fs tick=%llu pc=0x%x top=%u itop=%u\n", kind,
                     static_cast<double>(tick) / kTestVBlanksPerSecond, static_cast<unsigned long long>(tick), startPC, top, itop);
        if (plan.kicksOnly)
        {
            g_ps2xVu1TraceKicks = 2;
            return true;
        }
        if (data)
        {
            bool gap = false;
            for (uint32_t q = 0; q < PS2_VU1_DATA_SIZE / 16u; ++q)
            {
                uint32_t w[4];
                float f[4];
                std::memcpy(w, data + q * 16u, sizeof(w));
                std::memcpy(f, w, sizeof(f));
                if ((w[0] | w[1] | w[2] | w[3]) == 0u)
                {
                    gap = true;
                    continue;
                }
                if (gap)
                    std::fprintf(stderr, "[vu1]   ...\n");
                gap = false;
                std::fprintf(stderr, "[vu1]   in %4u: %08x %08x %08x %08x  % .5g % .5g % .5g % .5g\n",
                             q, w[0], w[1], w[2], w[3], f[0], f[1], f[2], f[3]);
            }
        }
        g_ps2xVu1TraceKicks = 1;
        return true;
    }

    void vu1TraceEnd(const VU1State &state)
    {
        const bool kicksOnly = g_ps2xVu1TraceKicks == 2;
        g_ps2xVu1TraceKicks = 0;
        if (kicksOnly)
            return;
        std::fprintf(stderr, "[vu1] done pc=0x%x cycles=%llu ebit=%d q=%g p=%g\n", state.pc,
                     static_cast<unsigned long long>(state.cycles), state.ebit ? 1 : 0, state.q, state.p);
        for (uint32_t r = 0; r < 32u; ++r)
            std::fprintf(stderr, "[vu1]   vf%02u % .6g % .6g % .6g % .6g\n", r,
                         state.vf[r][0], state.vf[r][1], state.vf[r][2], state.vf[r][3]);
        std::fprintf(stderr, "[vu1]   vi");
        for (uint32_t r = 0; r < 16u; ++r)
            std::fprintf(stderr, " %d", state.vi[r]);
        std::fprintf(stderr, "\n");
    }

    uint64_t envEmulatedTicks(const char *name)
    {
        const char *value = std::getenv(name);
        return value ? static_cast<uint64_t>(std::atof(value) * kTestVBlanksPerSecond) : 0ull;
    }

    // Unattended test runs, all in emulated time (seconds since boot, counted
    // in vblanks) so they do not depend on how fast the host runs:
    //   PS2X_FRAME_DUMP_DIR=<dir>     write presented frames as PNGs there
    //   PS2X_FRAME_DUMP_EVERY=<secs>  ...periodically
    //   PS2X_FRAME_DUMP_AT=<t1,t2>    ...and/or at specific times
    //   PS2X_EXIT_AT=<secs>           dump a last frame and exit
    //   PS2X_PERF_LOG=<secs>          print emulated-vs-wall speed periodically
    struct FrameDumpPlan
    {
        std::string dir;
        uint64_t everyTicks = 0;
        std::vector<uint64_t> atTicks;

        bool dueAt(uint64_t tick) const
        {
            if (dir.empty() || tick == 0u)
                return false;
            if (everyTicks != 0u && tick % everyTicks == 0u)
                return true;
            return std::binary_search(atTicks.begin(), atTicks.end(), tick);
        }
    };

    const FrameDumpPlan &frameDumpPlan()
    {
        static const FrameDumpPlan plan = []
        {
            FrameDumpPlan p;
            if (const char *dir = std::getenv("PS2X_FRAME_DUMP_DIR"))
                p.dir = dir;
            p.everyTicks = envEmulatedTicks("PS2X_FRAME_DUMP_EVERY");
            if (const char *list = std::getenv("PS2X_FRAME_DUMP_AT"))
            {
                std::stringstream stream(list);
                std::string item;
                while (std::getline(stream, item, ','))
                    if (!item.empty())
                        p.atTicks.push_back(static_cast<uint64_t>(std::atof(item.c_str()) * kTestVBlanksPerSecond));
            }
            if (const uint64_t exitTick = envEmulatedTicks("PS2X_EXIT_AT"))
                p.atTicks.push_back(exitTick);
            std::sort(p.atTicks.begin(), p.atTicks.end());
            return p;
        }();
        return plan;
    }

    // The last tick whose frame the window thread has saved.
    std::atomic<uint64_t> g_frameDumpSavedTick{0};

    // Whether the frame on display at this tick is one to save. Asked once
    // a tick: ticks at or before it are done with afterwards, saved or not.
    bool frameDumpDue(uint64_t tick)
    {
        const FrameDumpPlan &plan = frameDumpPlan();
        static uint64_t nextEveryTick = plan.everyTicks;
        static size_t nextAt = 0;

        if (plan.dir.empty())
            return false;

        bool due = false;
        if (plan.everyTicks != 0u && tick >= nextEveryTick)
        {
            due = true;
            while (nextEveryTick <= tick)
                nextEveryTick += plan.everyTicks;
        }
        while (nextAt < plan.atTicks.size() && tick >= plan.atTicks[nextAt])
        {
            due = true;
            ++nextAt;
        }
        return due;
    }

    void maybeDumpPresentedFrame(const uint8_t *rgba, uint32_t strideBytes, uint32_t width, uint32_t height, uint64_t tick)
    {
        const FrameDumpPlan &plan = frameDumpPlan();
        if (!frameDumpDue(tick))
            return;
        if (!rgba || width == 0u || height == 0u)
        {
            g_frameDumpSavedTick.store(tick, std::memory_order_release);
            return;
        }

        std::vector<uint8_t> packed(static_cast<size_t>(width) * height * 4u);
        for (uint32_t y = 0; y < height; ++y)
            std::memcpy(packed.data() + static_cast<size_t>(y) * width * 4u,
                        rgba + static_cast<size_t>(y) * strideBytes, static_cast<size_t>(width) * 4u);
        Image image{};
        image.data = packed.data();
        image.width = static_cast<int>(width);
        image.height = static_cast<int>(height);
        image.mipmaps = 1;
        image.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;
        char name[64];
        std::snprintf(name, sizeof(name), "/f_%07.2f.png", static_cast<double>(tick) / kTestVBlanksPerSecond);
        ExportImage(image, (plan.dir + name).c_str());
        g_frameDumpSavedTick.store(tick, std::memory_order_release);
    }
}

// Called by the EE at each vblank. A frame dump is of the picture on display
// at its tick, whatever the host's speed: the emulation stops at a tick that
// is to be dumped until the window thread has saved it (or stopped drawing).
void ps2xSetGsBackgroundDrawing(const uint8_t *vram); // gs/gs_cpu_backend.cpp

void ps2xHoldForFrameDump(uint64_t tick)
{
    if (!frameDumpPlan().dueAt(tick))
        return;
    GsWorker::drain(); // a GS that may lag: the picture of this vblank, not of an earlier one
    const auto giveUp = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (g_frameDumpSavedTick.load(std::memory_order_acquire) < tick && std::chrono::steady_clock::now() < giveUp)
        std::this_thread::sleep_for(std::chrono::microseconds(200));
}

// The texture the picture is uploaded to, at the size of the picture.
static void resizeFrameTexture(Texture2D &tex, int width, int height)
{
    if (tex.width == width && tex.height == height)
        return;
    UnloadTexture(tex);
    Image blank = GenImageColor(width, height, BLANK);
    tex = LoadTextureFromImage(blank);
    UnloadImage(blank);
    SetTextureFilter(tex, displaySmooth() ? TEXTURE_FILTER_BILINEAR : TEXTURE_FILTER_POINT);
}

// The picture when the graphics card has kept it (GSGpu::LatestShown): its
// texture, which the window draws in place of the one frames are uploaded to.
static Texture2D g_cardTexture{};
static bool g_cardPicture = false;

static void UploadFrame(Texture2D &tex, PS2Runtime *rt, uint32_t &outWidth, uint32_t &outHeight)
{
    static uint64_t s_lastPresentationTick = std::numeric_limits<uint64_t>::max();
    static bool s_hasLatchedInitialFrame = false;
    static uint32_t s_lastDisplayFbp = std::numeric_limits<uint32_t>::max();
    static uint32_t s_lastSourceFbp = std::numeric_limits<uint32_t>::max();
    static bool s_lastPreferred = false;
    static uint32_t s_lastWidth = 0u;
    static uint32_t s_lastHeight = 0u;
    static bool s_hasUploadedFrame = false;
    static std::vector<uint8_t> s_scratch;
    static std::vector<uint8_t> s_uploadBuffer(DEFAULT_FB_SIZE, 0u);

    // A GS that may lag (GsWorker::lagVBlanks) has a new picture when it
    // passes a vblank, shown with the display registers of that vblank.
    uint64_t currentTick = rt->eeScheduler().currentVSyncTick();
    GsWorker::Display shown;
    const bool lagging = GsWorker::shownDisplay(shown);
    if (lagging)
        currentTick = shown.tick;
    const bool needsLatch = !s_hasLatchedInitialFrame || currentTick != s_lastPresentationTick;
    if (needsLatch)
    {
        rt->gs().latchHostPresentationFrame(lagging ? &shown : nullptr);
        s_lastPresentationTick = currentTick;
        s_hasLatchedInitialFrame = true;
    }
    else if (s_hasUploadedFrame)
    {
        outWidth = (s_lastWidth != 0u) ? s_lastWidth : FB_WIDTH;
        outHeight = (s_lastHeight != 0u) ? s_lastHeight : DEFAULT_DISPLAY_HEIGHT;
        return;
    }

    // Did the card show a picture for this latch? Then that is it.
    {
        static uint64_t s_shownSerial = 0u;
        GSGpu::Shown shown;
        g_cardPicture = needsLatch && GSGpu::LatestShown(shown) && shown.serial != s_shownSerial;
        if (g_cardPicture)
        {
            s_shownSerial = shown.serial;
            if (g_cardTexture.id != shown.texture || g_cardTexture.width != static_cast<int>(shown.width) ||
                g_cardTexture.height != static_cast<int>(shown.height))
            {
                g_cardTexture.id = shown.texture;
                g_cardTexture.width = static_cast<int>(shown.width);
                g_cardTexture.height = static_cast<int>(shown.height);
                g_cardTexture.mipmaps = 1;
                g_cardTexture.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;
                SetTextureFilter(g_cardTexture, displaySmooth() ? TEXTURE_FILTER_BILINEAR : TEXTURE_FILTER_POINT);
            }
            if (frameDumpDue(currentTick))
            {
                Image image = LoadImageFromTexture(g_cardTexture);
                if (image.data)
                {
                    char name[64];
                    std::snprintf(name, sizeof(name), "/f_%07.2f.png", static_cast<double>(currentTick) / kTestVBlanksPerSecond);
                    ExportImage(image, (frameDumpPlan().dir + name).c_str());
                    UnloadImage(image);
                }
                g_frameDumpSavedTick.store(currentTick, std::memory_order_release);
            }
            outWidth = shown.width;
            outHeight = shown.height;
            s_lastWidth = outWidth;
            s_lastHeight = outHeight;
            s_hasUploadedFrame = true;
            return;
        }
    }

    s_scratch.clear();
    uint32_t width = 0u;
    uint32_t height = 0u;
    uint32_t displayFbp = 0u;
    uint32_t sourceFbp = 0u;
    bool usedPreferredDisplaySource = false;
    if (!rt->gs().copyLatchedHostPresentationFrame(s_scratch,
                                                   width,
                                                   height,
                                                   &displayFbp,
                                                   &sourceFbp,
                                                   &usedPreferredDisplaySource))
    {
        resizeFrameTexture(tex, FB_WIDTH, FB_HEIGHT);
        Image blank = GenImageColor(FB_WIDTH, FB_HEIGHT, MAGENTA);
        UpdateTexture(tex, blank.data);
        UnloadImage(blank);
        maybeDumpPresentedFrame(nullptr, 0u, 0u, 0u, currentTick); // nothing to save for this tick
        outWidth = FB_WIDTH;
        outHeight = DEFAULT_DISPLAY_HEIGHT;
        s_lastWidth = outWidth;
        s_lastHeight = outHeight;
        s_hasUploadedFrame = true;
        return;
    }

    PS2_IF_AGRESSIVE_LOGS({
        static uint32_t s_uploadDebugCount = 0u;
        if (s_uploadDebugCount < 128u ||
            displayFbp != s_lastDisplayFbp ||
            sourceFbp != s_lastSourceFbp ||
            usedPreferredDisplaySource != s_lastPreferred ||
            width != s_lastWidth ||
            height != s_lastHeight)
        {
            std::cout << "[frame:upload] idx=" << s_uploadDebugCount
                      << " tick=" << currentTick
                      << " displayFbp=" << displayFbp
                      << " sourceFbp=" << sourceFbp
                      << " size=" << width << "x" << height
                      << " preferred=" << static_cast<uint32_t>(usedPreferredDisplaySource ? 1u : 0u)
                      << std::endl;
        }
        ++s_uploadDebugCount;
    });
    s_lastDisplayFbp = displayFbp;
    s_lastSourceFbp = sourceFbp;
    s_lastPreferred = usedPreferredDisplaySource;
    s_lastWidth = width;
    s_lastHeight = height;

    // A picture larger than the game's own (a higher internal resolution)
    // has a texture of its own size, and comes packed.
    const bool large = width > static_cast<uint32_t>(FB_WIDTH) || height > static_cast<uint32_t>(FB_HEIGHT);
    resizeFrameTexture(tex, large ? static_cast<int>(width) : FB_WIDTH, large ? static_cast<int>(height) : FB_HEIGHT);
    if (large && s_scratch.size() >= static_cast<size_t>(width) * height * 4u)
    {
        UpdateTexture(tex, s_scratch.data());
        maybeDumpPresentedFrame(s_scratch.data(), width * 4u, width, height, currentTick);
        outWidth = width;
        outHeight = height;
        s_hasUploadedFrame = true;
        return;
    }

    std::fill(s_uploadBuffer.begin(), s_uploadBuffer.end(), 0u);
    if (!s_scratch.empty() && width != 0u && height != 0u)
    {
        const uint32_t copyWidth = std::min<uint32_t>(width, FB_WIDTH);
        const uint32_t copyHeight = std::min<uint32_t>(height, FB_HEIGHT);
        const size_t srcRowBytes = static_cast<size_t>(width) * 4u;
        const size_t dstRowBytes = static_cast<size_t>(FB_WIDTH) * 4u;
        const size_t copyRowBytes = static_cast<size_t>(copyWidth) * 4u;
        for (uint32_t y = 0; y < copyHeight; ++y)
        {
            const size_t srcOffset = static_cast<size_t>(y) * srcRowBytes;
            const size_t dstOffset = static_cast<size_t>(y) * dstRowBytes;
            if (srcOffset + copyRowBytes > s_scratch.size() ||
                dstOffset + copyRowBytes > s_uploadBuffer.size())
            {
                break;
            }
            std::memcpy(s_uploadBuffer.data() + dstOffset, s_scratch.data() + srcOffset, copyRowBytes);
        }
    }

    UpdateTexture(tex, s_uploadBuffer.data());
    maybeDumpPresentedFrame(s_uploadBuffer.data(), FB_WIDTH * 4u,
                            std::min<uint32_t>(width, FB_WIDTH), std::min<uint32_t>(height, FB_HEIGHT), currentTick);
    outWidth = width;
    outHeight = height;
    s_hasUploadedFrame = true;
}

PS2Runtime::PS2Runtime()
{
    m_iopHost = std::make_unique<PS2IopHostAdapter>(*this);
    m_iopSubsystem = std::make_unique<ps2x::iop::IopSubsystem>(*m_iopHost);

    m_eeScheduler = std::make_unique<EeScheduler>(*this);

    // Assign rather than memset: R5900Context's constructor zeroes itself and
    // then applies the COP0 reset values, which a memset here would discard.
    m_cpuContext = R5900Context{};

    // R0 is always zero in MIPS
    m_cpuContext.r[0] = _mm_set1_epi32(0);
    m_cpuContext.vu0_vf[0] = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);
    m_cpuContext.vu0_q = 1.0f;
    m_cpuContext.vu0_r = _mm_castsi128_ps(_mm_set1_epi32(0x3F800000));

    // Stack pointer (SP) and global pointer (GP) will be set by the loaded ELF

    m_loadedModules.clear();
    m_guestHeapBlocks.clear();
    m_guestHeapBase = kGuestHeapDefaultBase;
    m_guestHeapEnd = kGuestHeapDefaultBase;
    m_guestHeapLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
    m_guestHeapSuggestedBase = kGuestHeapDefaultBase;
    m_guestHeapConfigured = false;
    m_asyncCallbackStackFloor = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
    m_asyncCallbackStackTop = PS2_RAM_SIZE;
}

void PS2Runtime::setDebugUiCallbacks(DebugUiCallback initCallback,
                                     DebugUiCallback drawCallback,
                                     DebugUiCallback shutdownCallback,
                                     void *userData)
{
    if (m_debugUiInitialized && m_debugUiShutdownCallback)
    {
        m_debugUiShutdownCallback(*this, m_debugUiUserData);
        m_debugUiInitialized = false;
    }

    m_debugUiInitCallback = initCallback;
    m_debugUiDrawCallback = drawCallback;
    m_debugUiShutdownCallback = shutdownCallback;
    m_debugUiUserData = userData;
}

PS2Runtime::~PS2Runtime()
{
    try
    {
        requestStop();
        Vu1Worker::stop();
        GsWorker::stop();
        ps2xSetGsBackgroundDrawing(nullptr);
        m_iopSubsystem.reset();
        m_iopHost.reset();
#if defined(PLATFORM_VITA)
        m_audioBackend.stopAll();
        m_audioBackend.setAudioReady(false);
#else
        ps2x::audio::stopOutput();
        if (IsAudioDeviceReady())
        {
            CloseAudioDevice();
            m_audioBackend.setAudioReady(false);
        }
#endif
        if (m_debugUiInitialized && m_debugUiShutdownCallback)
        {
            m_debugUiShutdownCallback(*this, m_debugUiUserData);
            m_debugUiInitialized = false;
        }

        if (IsWindowReady())
        {
            CloseWindow();
        }

        m_loadedModules.clear();
    }
    catch (const std::exception &e)
    {
        std::cerr << "[~PS2Runtime] cleanup exception: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "[~PS2Runtime] cleanup exception: unknown" << std::endl;
    }
}

ps2x::iop::ModuleLoadResult PS2Runtime::loadIopModule(std::string_view path, const void *arguments, uint32_t argumentSize)
{
    auto scope = m_iopHost->enterCall(nullptr, m_memory.getRDRAM());
    return m_iopSubsystem->loadModule(path, arguments, argumentSize);
}

ps2x::iop::ModuleLoadResult PS2Runtime::loadIopModuleBuffer(uint32_t guestAddress, const void *arguments, uint32_t argumentSize)
{
    auto scope = m_iopHost->enterCall(nullptr, m_memory.getRDRAM());
    return m_iopSubsystem->loadModuleBuffer(guestAddress, arguments, argumentSize);
}

bool PS2Runtime::stopIopModule(int32_t moduleId, int32_t *result)
{
    auto scope = m_iopHost->enterCall(nullptr, m_memory.getRDRAM());
    return m_iopSubsystem->stopModule(moduleId, result);
}

ps2x::iop::RpcAbi PS2Runtime::selectIopRpcAbi(const ps2x::iop::RpcAbiRequest &request) const
{
    return m_iopSubsystem->selectRpcAbi(request);
}

bool PS2Runtime::canBindIopRpc(uint32_t sid) const noexcept
{
    return m_iopSubsystem->canBindRpc(sid);
}

ps2x::iop::RpcResult PS2Runtime::handleIopRpc(uint8_t *rdram, R5900Context *ctx, ps2x::iop::RpcRequest request)
{
    auto scope = m_iopHost->enterCall(ctx, rdram);
    request.callToken = scope.token();
    return m_iopSubsystem->handleRpc(request);
}

void PS2Runtime::notifyIopSifTransfer(uint8_t *rdram, const ps2x::iop::SifTransfer &transfer)
{
    auto scope = m_iopHost->enterCall(nullptr, rdram);
    m_iopSubsystem->onSifTransfer(transfer);
}

void PS2Runtime::advanceIopEeCycles(uint64_t eeCycles) noexcept
{
    m_iopSubsystem->runEeCycles(eeCycles);
}

void PS2Runtime::resetIop()
{
    m_iopSubsystem->reset();
}

ps2x::iop::DebugSnapshot PS2Runtime::iopDebugSnapshot() const
{
    return m_iopSubsystem->debugSnapshot();
}

uint32_t PS2Runtime::allocateIopMemory(uint32_t size, uint32_t alignment)
{
    return m_iopSubsystem ? m_iopSubsystem->allocateMemory(size, alignment) : 0u;
}

bool PS2Runtime::freeIopMemory(uint32_t address)
{
    return m_iopSubsystem && m_iopSubsystem->freeMemory(address);
}

bool PS2Runtime::readIopMemory(uint32_t address, void *destination, size_t size) const
{
    return m_iopSubsystem && m_iopSubsystem->readMemory(address, destination, size);
}

bool PS2Runtime::writeIopMemory(uint32_t address, const void *source, size_t size)
{
    return m_iopSubsystem && m_iopSubsystem->writeMemory(address, source, size);
}

bool PS2Runtime::zeroIopMemory(uint32_t address, size_t size)
{
    return m_iopSubsystem && m_iopSubsystem->zeroMemory(address, size);
}

bool PS2Runtime::isIopMemoryRange(uint32_t address, size_t size) const
{
    return m_iopSubsystem && m_iopSubsystem->isMemoryRange(address, size);
}

bool PS2Runtime::syncCoreSubsystems()
{
    uint8_t *const rdram = m_memory.getRDRAM();
    uint8_t *const gsVram = m_memory.getGSVRAM();
    if (!rdram || !gsVram)
    {
        return false;
    }

    if (m_boundRdram == rdram && m_boundGSVram == gsVram)
    {
        return true;
    }

    m_gs.init(gsVram, static_cast<uint32_t>(PS2_GS_VRAM_SIZE), &m_memory.gs());
    ps2xSetGsBackgroundDrawing(gsVram);
    GsWorker::start(m_gs, &m_memory.gs());
    Vu1Worker::start(m_memory);
    if (m_eeScheduler)
        ps2xInstallGsFinishDelay(*m_eeScheduler);
    m_gifArbiter.setProcessPacketFn([](const uint8_t *data, uint32_t size)
                                    { GsWorker::post(data, size); });
    m_memory.setGifArbiter(&m_gifArbiter);
    m_memory.setVu1MscalCallback([this](uint32_t startPC, uint32_t top, uint32_t itop)
                                 {
                                     if (skipVu1ForFastForward(m_gs))
                                         return;
                                     // On the VU1 worker the EE's registers belong to another
                                     // thread: the D/T stop enables are taken as off and the
                                     // stop bits are not reported back.
                                     const bool onWorker = Vu1Worker::onWorkerThread();
                                     R5900Context *cpuContext = (!onWorker && m_eeScheduler) ? m_eeScheduler->currentContext() : nullptr;
                                     if (!cpuContext)
                                     {
                                         cpuContext = &m_cpuContext;
                                     }
                                     m_vu1.state().dBitEnabled =
                                         !onWorker && (cpuContext->vu0_fbrst & (1u << 10)) != 0u;
                                     m_vu1.state().tBitEnabled =
                                         !onWorker && (cpuContext->vu0_fbrst & (1u << 11)) != 0u;
                                     const bool traced = vu1TraceBegin(m_eeScheduler ? m_eeScheduler->currentVSyncTick() : 0u,
                                                                       "MSCAL", startPC, top, itop, m_memory.getVU1Data(), m_memory.getVU1Code());
                                     m_vu1.execute(m_memory.getVU1Code(), PS2_VU1_CODE_SIZE,
                                                   m_memory.getVU1Data(), PS2_VU1_DATA_SIZE,
                                                   m_gs, &m_memory, startPC, top, itop, 65536);
                                     if (traced)
                                         vu1TraceEnd(m_vu1.state());
                                     if (!onWorker)
                                     {
                                         cpuContext->vu0_vpu_stat =
                                             (cpuContext->vu0_vpu_stat & ~0x0600u) |
                                             (m_vu1.state().stoppedByD ? 0x0200u : 0u) |
                                             (m_vu1.state().stoppedByT ? 0x0400u : 0u);
                                     } });
    m_memory.setVu1MscntCallback([this](uint32_t top, uint32_t itop)
                                 {
                                     if (skipVu1ForFastForward(m_gs))
                                         return;
                                     // On the VU1 worker the EE's registers belong to another
                                     // thread: the D/T stop enables are taken as off and the
                                     // stop bits are not reported back.
                                     const bool onWorker = Vu1Worker::onWorkerThread();
                                     R5900Context *cpuContext = (!onWorker && m_eeScheduler) ? m_eeScheduler->currentContext() : nullptr;
                                     if (!cpuContext)
                                     {
                                         cpuContext = &m_cpuContext;
                                     }
                                     m_vu1.state().dBitEnabled =
                                         !onWorker && (cpuContext->vu0_fbrst & (1u << 10)) != 0u;
                                     m_vu1.state().tBitEnabled =
                                         !onWorker && (cpuContext->vu0_fbrst & (1u << 11)) != 0u;
                                     const bool traced = vu1TraceBegin(m_eeScheduler ? m_eeScheduler->currentVSyncTick() : 0u,
                                                                       "MSCNT", m_vu1.state().pc, top, itop, m_memory.getVU1Data(), m_memory.getVU1Code());
                                     m_vu1.resume(m_memory.getVU1Code(), PS2_VU1_CODE_SIZE,
                                                  m_memory.getVU1Data(), PS2_VU1_DATA_SIZE,
                                                  m_gs, &m_memory, top, itop, 65536);
                                     if (traced)
                                         vu1TraceEnd(m_vu1.state());
                                     if (!onWorker)
                                     {
                                         cpuContext->vu0_vpu_stat =
                                             (cpuContext->vu0_vpu_stat & ~0x0600u) |
                                             (m_vu1.state().stoppedByD ? 0x0200u : 0u) |
                                             (m_vu1.state().stoppedByT ? 0x0400u : 0u);
                                     } });
    resetIop();
    m_vu0.reset();
    m_vu1.reset();

    m_boundRdram = rdram;
    m_boundGSVram = gsVram;
    return true;
}

bool PS2Runtime::initialize(const char *title)
{
    try
    {
        if (!m_memory.initialize())
        {
            std::cerr << "Failed to initialize PS2 memory" << std::endl;
            return false;
        }

        if (!syncCoreSubsystems())
        {
            std::cerr << "Failed to bind runtime core subsystems" << std::endl;
            return false;
        }
#if defined(PLATFORM_VITA)
        InitWindow(HOST_WINDOW_WIDTH, HOST_WINDOW_HEIGHT, title); // raylib vita does not support audio
#else
        // ALWAYS_RUN: raylib otherwise waits in PollInputEvents for as long
        // as the window is minimised, and this loop with it (no picture, no
        // pad, no frame dumps, no PS2X_EXIT_AT) while the game's own thread
        // plays on with whatever the pad last held.
        SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_WINDOW_ALWAYS_RUN);
        InitWindow(hostDisplay().windowWidth, hostDisplay().windowHeight, title);
        ps2x::setDisplayMode(hostDisplay().mode);
        // A second GL context beside the window's, for a GS drawn by the
        // graphics card (display.renderer = gpu or hybrid).
        if (const char *renderer = ps2x::setting("PS2X_GS_RENDERER", "display.renderer");
            renderer && (std::strcmp(renderer, "gpu") == 0 || std::strcmp(renderer, "hybrid") == 0))
            GSGpu::PrepareContext();
        InitAudioDevice();
        m_audioBackend.setAudioReady(IsAudioDeviceReady());
        ps2x::audio::startOutput();
        // [audio] volume (PS2X_VOLUME): 0 to 100, of everything that is played.
        if (const char *volume = ps2x::setting("PS2X_VOLUME", "audio.volume"))
            SetMasterVolume(std::clamp(static_cast<float>(std::atof(volume)), 0.0f, 100.0f) / 100.0f);
#endif
        SetTargetFPS(60);
        if (m_debugUiInitCallback)
        {
            m_debugUiInitCallback(*this, m_debugUiUserData);
            m_debugUiInitialized = true;
        }

        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Failed to initialize PS2 runtime: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "Failed to initialize PS2 runtime: unknown exception" << std::endl;
    }

    return false;
}

bool PS2Runtime::loadELF(const std::string &elfPath)
{
    configureIoPathsFromElf(elfPath);

    std::ifstream file(elfPath, std::ios::binary);
    if (!file)
    {
        std::cerr << "Failed to open ELF file: " << elfPath << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    const std::streamoff fileSize = file.tellg();
    if (fileSize < static_cast<std::streamoff>(sizeof(ElfHeader)))
    {
        std::cerr << "ELF file is too small: " << elfPath << std::endl;
        return false;
    }
    file.seekg(0, std::ios::beg);

    ElfHeader header{};
    if (!file.read(reinterpret_cast<char *>(&header), sizeof(header)))
    {
        std::cerr << "Failed to read ELF header from: " << elfPath << std::endl;
        return false;
    }

    if (header.magic != ELF_MAGIC)
    {
        std::cerr << "Invalid ELF magic number" << std::endl;
        return false;
    }

    if (header.elf_class != 1u || header.endianness != 1u)
    {
        std::cerr << "Unsupported ELF format (expected 32-bit little-endian)." << std::endl;
        return false;
    }

    if (header.machine != EM_MIPS || header.type != ET_EXEC)
    {
        std::cerr << "Not a MIPS executable ELF file" << std::endl;
        return false;
    }

    if (header.phnum != 0u && header.phentsize < sizeof(ProgramHeader))
    {
        std::cerr << "Unsupported ELF program-header entry size: " << header.phentsize << std::endl;
        return false;
    }

    const uint64_t programHeaderTableEnd =
        static_cast<uint64_t>(header.phoff) +
        static_cast<uint64_t>(header.phnum) * static_cast<uint64_t>(header.phentsize);
    if (programHeaderTableEnd > static_cast<uint64_t>(fileSize))
    {
        std::cerr << "ELF program-header table is out of range." << std::endl;
        return false;
    }

    m_cpuContext.pc = header.entry;
    m_debugPc.store(m_cpuContext.pc, std::memory_order_relaxed);

    uint32_t maxLoadedRdramEnd = kGuestHeapDefaultBase;
    uint32_t moduleBase = std::numeric_limits<uint32_t>::max();
    uint32_t moduleEnd = 0u;
    bool loadedAnySegment = false;

    for (uint16_t i = 0; i < header.phnum; i++)
    {
        const uint64_t phOffset =
            static_cast<uint64_t>(header.phoff) +
            static_cast<uint64_t>(i) * static_cast<uint64_t>(header.phentsize);
        if (phOffset + sizeof(ProgramHeader) > static_cast<uint64_t>(fileSize))
        {
            std::cerr << "ELF program header " << i << " is out of range." << std::endl;
            return false;
        }

        ProgramHeader ph{};
        file.seekg(static_cast<std::streamoff>(phOffset), std::ios::beg);
        if (!file.read(reinterpret_cast<char *>(&ph), sizeof(ph)))
        {
            std::cerr << "Failed to read ELF program header " << i << std::endl;
            return false;
        }

        if (ph.type != PT_LOAD || ph.memsz == 0u)
        {
            continue;
        }

        if (ph.filesz > ph.memsz)
        {
            std::cerr << "ELF segment " << i << " has filesz > memsz." << std::endl;
            return false;
        }

        const uint64_t segmentFileEnd = static_cast<uint64_t>(ph.offset) + static_cast<uint64_t>(ph.filesz);
        if (segmentFileEnd > static_cast<uint64_t>(fileSize))
        {
            std::cerr << "ELF segment " << i << " exceeds file bounds." << std::endl;
            return false;
        }

        const bool scratch =
            ph.vaddr >= PS2_SCRATCHPAD_BASE &&
            ph.vaddr < (PS2_SCRATCHPAD_BASE + PS2_SCRATCHPAD_SIZE);

        uint32_t physAddr = 0u;
        try
        {
            physAddr = m_memory.translateAddress(ph.vaddr);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Failed to translate ELF segment " << i
                      << " virtual address 0x" << std::hex << ph.vaddr
                      << std::dec << ": " << e.what() << std::endl;
            return false;
        }
        const uint64_t regionSize = scratch ? static_cast<uint64_t>(PS2_SCRATCHPAD_SIZE)
                                            : static_cast<uint64_t>(PS2_RAM_SIZE);
        const uint64_t segmentMemEnd = static_cast<uint64_t>(physAddr) + static_cast<uint64_t>(ph.memsz);
        if (segmentMemEnd > regionSize)
        {
            std::cerr << "ELF segment " << i << " exceeds "
                      << (scratch ? "scratchpad" : "RDRAM")
                      << " bounds (vaddr=0x" << std::hex << ph.vaddr
                      << " memsz=0x" << ph.memsz << std::dec << ")." << std::endl;
            return false;
        }

        uint8_t *destBase = scratch ? m_memory.getScratchpad() : m_memory.getRDRAM();
        if (!destBase)
        {
            std::cerr << "ELF segment " << i << " has no destination memory backing." << std::endl;
            return false;
        }

        uint8_t *dest = destBase + physAddr;
        if (ph.filesz > 0u)
        {
            file.seekg(static_cast<std::streamoff>(ph.offset), std::ios::beg);
            if (!file.read(reinterpret_cast<char *>(dest), ph.filesz))
            {
                std::cerr << "Failed to read ELF segment " << i << " payload." << std::endl;
                return false;
            }
        }

        if (ph.memsz > ph.filesz)
        {
            std::memset(dest + ph.filesz, 0, ph.memsz - ph.filesz);
        }

        RUNTIME_LOG("Loading segment: 0x" << std::hex << ph.vaddr
                                          << " - 0x" << (static_cast<uint64_t>(ph.vaddr) + static_cast<uint64_t>(ph.memsz))
                                          << " (filesz: 0x" << ph.filesz
                                          << ", memsz: 0x" << ph.memsz << ")"
                                          << std::dec << std::endl);

        if (!scratch)
        {
            maxLoadedRdramEnd = std::max(maxLoadedRdramEnd, static_cast<uint32_t>(segmentMemEnd));
        }

        if (ph.flags & 0x1u) // PF_X
        {
            const uint64_t execEnd = static_cast<uint64_t>(ph.vaddr) + static_cast<uint64_t>(ph.filesz);
            if (execEnd <= std::numeric_limits<uint32_t>::max())
            {
                m_memory.registerCodeRegion(ph.vaddr, static_cast<uint32_t>(execEnd));
            }
        }

        loadedAnySegment = true;
        moduleBase = std::min(moduleBase, ph.vaddr);
        const uint64_t segmentVirtualEnd = static_cast<uint64_t>(ph.vaddr) + static_cast<uint64_t>(ph.memsz);
        const uint32_t clampedVirtualEnd =
            (segmentVirtualEnd > std::numeric_limits<uint32_t>::max())
                ? std::numeric_limits<uint32_t>::max()
                : static_cast<uint32_t>(segmentVirtualEnd);
        moduleEnd = std::max(moduleEnd, clampedVirtualEnd);
    }

    if (!loadedAnySegment)
    {
        std::cerr << "ELF contains no loadable PT_LOAD segments." << std::endl;
        return false;
    }

    if (maxLoadedRdramEnd > PS2_RAM_SIZE)
    {
        maxLoadedRdramEnd = PS2_RAM_SIZE;
    }

    const uint32_t paddedEnd = (maxLoadedRdramEnd > (PS2_RAM_SIZE - kGuestHeapSafetyPad))
                                   ? PS2_RAM_SIZE
                                   : (maxLoadedRdramEnd + kGuestHeapSafetyPad);
    const uint32_t suggestedHeapBase = alignGuestHeapValue(paddedEnd, kGuestHeapDefaultAlignment);
    {
        std::lock_guard<std::mutex> lock(m_guestHeapMutex);
        if (!m_guestHeapConfigured)
        {
            const uint32_t hardLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
            m_guestHeapSuggestedBase = std::min(suggestedHeapBase, hardLimit);
            m_guestHeapBase = m_guestHeapSuggestedBase;
            m_guestHeapEnd = m_guestHeapSuggestedBase;
            m_guestHeapLimit = hardLimit;
        }
    }
    {
        std::lock_guard<std::mutex> lock(m_asyncCallbackStackMutex);
        const uint32_t hardLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
        m_asyncCallbackStackFloor = std::min(std::max(hardLimit, suggestedHeapBase), PS2_RAM_SIZE);
        m_asyncCallbackStackTop = PS2_RAM_SIZE;
    }

    LoadedModule module;
    module.name = elfPath.substr(elfPath.find_last_of("/\\") + 1);
    module.baseAddress = (moduleBase == std::numeric_limits<uint32_t>::max()) ? 0x00100000u : moduleBase;
    module.size = (moduleEnd > module.baseAddress) ? static_cast<size_t>(moduleEnd - module.baseAddress) : 0u;
    module.active = true;

    m_loadedModules.push_back(module);

    uint32_t elfCrc32 = 0u;
    const bool elfCrc32Valid = computeFileCrc32(elfPath, elfCrc32);
    if (!elfCrc32Valid)
    {
        std::cerr << "[ps2xIOP] failed to compute ELF CRC32 for '" << elfPath << "'" << std::endl;
    }
    ps2x::iop::GameIdentity identity;
    identity.elfName = module.name;
    identity.entryPoint = m_cpuContext.pc;
    identity.crc32 = elfCrc32;
    std::string romError;
    if (!m_romDevice.configure(identity, &romError))
    {
        std::cerr << "[ROM0] failed to configure profile: " << romError << std::endl;
        return false;
    }

    m_iopSubsystem->reset();

    ps2_game_overrides::applyMatching(*this,
                                      elfPath,
                                      m_cpuContext.pc,
                                      elfCrc32,
                                      elfCrc32Valid);

    // The ELF may carry the overlay the game starts with.
    refreshOverlays();

    RUNTIME_LOG("ELF file loaded successfully. Entry point: 0x" << std::hex << m_cpuContext.pc << std::dec);
    return true;
}

const PS2Runtime::IoPaths &PS2Runtime::getIoPaths()
{
    return runtimeIoPaths();
}

void PS2Runtime::setIoPaths(const IoPaths &paths)
{
    IoPaths normalized = paths;
    normalized.elfPath = normalizeAbsolutePath(normalized.elfPath);
    normalized.elfDirectory = normalizeAbsolutePath(normalized.elfDirectory);
    normalized.hostRoot = normalizeAbsolutePath(normalized.hostRoot);
    normalized.cdRoot = normalizeAbsolutePath(normalized.cdRoot);
    normalized.mcRoot = normalizeAbsolutePath(normalized.mcRoot);
    normalized.cdImage = normalizeAbsolutePath(normalized.cdImage);

    if (normalized.elfDirectory.empty() && !normalized.elfPath.empty())
    {
        normalized.elfDirectory = normalized.elfPath.parent_path();
    }

    if (normalized.hostRoot.empty())
    {
        normalized.hostRoot = normalized.elfDirectory;
    }
    if (normalized.cdRoot.empty())
    {
        normalized.cdRoot = normalized.elfDirectory;
    }
    if (normalized.mcRoot.empty())
    {
        normalized.mcRoot = normalized.elfDirectory / "mc0";
    }
    // PS2X_MC_ROOT=<dir>: keep the memory card somewhere else, so that test
    // runs leave the player's saves alone. Slot 2 is "<dir>_slot1".
    if (const char *mcRoot = std::getenv("PS2X_MC_ROOT"); mcRoot != nullptr && mcRoot[0] != '\0')
    {
        normalized.mcRoot = normalizeAbsolutePath(std::filesystem::path(mcRoot));
    }
    // PS2X_CD_IMAGE=<iso>: the disc as an image. Files are looked up in its
    // directories and read at their own sectors (both layers of a DVD9);
    // what was extracted next to the ELF is still found there.
    if (const char *cdImage = std::getenv("PS2X_CD_IMAGE"); cdImage != nullptr && cdImage[0] != '\0')
    {
        normalized.cdImage = normalizeAbsolutePath(std::filesystem::path(cdImage));
    }

    runtimeIoPaths() = normalized;
}

void PS2Runtime::configureIoPathsFromElf(const std::string &elfPath)
{
    IoPaths paths = runtimeIoPaths();
    paths.elfPath = normalizeAbsolutePath(std::filesystem::path(elfPath));
    if (!paths.elfPath.empty())
    {
        paths.elfDirectory = paths.elfPath.parent_path();
    }

    if (!paths.elfDirectory.empty())
    {
        paths.hostRoot = paths.elfDirectory;
        paths.cdRoot = paths.elfDirectory;
        paths.mcRoot = paths.elfDirectory / "mc0";
    }

    setIoPaths(paths);
}

namespace
{
    bool generatedFunctionTableSlot(uint32_t address, uint32_t &slot)
    {
        if ((address & 3u) != 0u || g_ps2RecompiledFunctionTableSlotCount == 0u)
        {
            return false;
        }

        if (address < g_ps2RecompiledFunctionTableBase || address >= g_ps2RecompiledFunctionTableEnd)
        {
            return false;
        }

        const uint32_t offset = address - g_ps2RecompiledFunctionTableBase;
        slot = offset >> 2;
        return slot < g_ps2RecompiledFunctionTableSlotCount;
    }
}

bool PS2Runtime::replaceFunction(uint32_t address, RecompiledFunction func)
{
    uint32_t slot = 0u;
    if (!generatedFunctionTableSlot(address, slot))
    {
        std::cerr << "[function-table] cannot replace guest PC 0x" << std::hex << address
                  << ": outside generated dense table [0x" << g_ps2RecompiledFunctionTableBase
                  << ", 0x" << g_ps2RecompiledFunctionTableEnd << ")"
                  << std::dec << std::endl;
        return false;
    }

    g_ps2RecompiledFunctionTable[slot] = func;
    return true;
}

bool PS2Runtime::registerFunction(uint32_t address, RecompiledFunction func)
{
    return replaceFunction(address, func);
}

bool PS2Runtime::hasFunction(uint32_t address) const
{
    uint32_t slot = 0u;
    return generatedFunctionTableSlot(address, slot) && g_ps2RecompiledFunctionTable[slot] != nullptr;
}

namespace
{
    struct RegisteredOverlay
    {
        PS2Runtime::OverlayImage image;
        bool active = false;
    };

    // Filled from the static initialisers of generated code, which run in no
    // particular order: the list has to exist by the first of them.
    std::vector<RegisteredOverlay> &registeredOverlays()
    {
        static std::vector<RegisteredOverlay> overlays;
        return overlays;
    }

    uint64_t fnv1a64(const uint8_t *data, size_t size, uint64_t hash = 0xCBF29CE484222325ull)
    {
        for (size_t i = 0; i < size; ++i)
        {
            hash = (hash ^ data[i]) * 0x100000001B3ull;
        }
        return hash;
    }

    bool overlayRangeInRam(const PS2Runtime::OverlayCodeRange &range)
    {
        const uint32_t start = range.start & PS2_RAM_MASK;
        return range.end > range.start && static_cast<uint64_t>(start) + (range.end - range.start) <= PS2_RAM_SIZE;
    }

    bool overlayProbeMatches(const PS2Runtime::OverlayImage &image, const uint8_t *rdram)
    {
        if (image.codeRangeCount == 0u || !overlayRangeInRam(image.code[0]))
        {
            return false;
        }
        const uint32_t size = std::min(image.code[0].end - image.code[0].start, PS2Runtime::kOverlayProbeBytes);
        return fnv1a64(rdram + (image.code[0].start & PS2_RAM_MASK), size) == image.probeHash;
    }

    bool overlayCodeMatches(const PS2Runtime::OverlayImage &image, const uint8_t *rdram)
    {
        uint64_t hash = 0xCBF29CE484222325ull;
        for (size_t i = 0; i < image.codeRangeCount; ++i)
        {
            if (!overlayRangeInRam(image.code[i]))
            {
                return false;
            }
            hash = fnv1a64(rdram + (image.code[i].start & PS2_RAM_MASK), image.code[i].end - image.code[i].start, hash);
        }
        return hash == image.codeHash;
    }
}

void PS2Runtime::registerOverlay(const OverlayImage &image)
{
    registeredOverlays().push_back(RegisteredOverlay{image, false});
}

void PS2Runtime::refreshOverlays()
{
    std::vector<RegisteredOverlay> &overlays = registeredOverlays();
    const uint8_t *rdram = m_memory.getRDRAM();
    if (overlays.empty() || rdram == nullptr)
    {
        return;
    }

    // Nothing has been loaded while the images in the table still start as
    // they did and no other one does: the usual case, and a cheap one. Once
    // that fails every image that could be there is hashed in full.
    std::vector<bool> probes(overlays.size());
    bool changed = false;
    for (size_t i = 0; i < overlays.size(); ++i)
    {
        probes[i] = overlayProbeMatches(overlays[i].image, rdram);
        changed = changed || probes[i] != overlays[i].active;
    }
    if (!changed)
    {
        return;
    }

    // Those that went are taken out before the ones that came are put in:
    // they share addresses.
    std::vector<RegisteredOverlay *> arrived;
    for (size_t i = 0; i < overlays.size(); ++i)
    {
        RegisteredOverlay &overlay = overlays[i];
        const bool present = probes[i] && overlayCodeMatches(overlay.image, rdram);
        if (overlay.active && !present)
        {
            for (size_t f = 0; f < overlay.image.functionCount; ++f)
            {
                replaceFunction(overlay.image.functions[f].address, nullptr);
            }
            overlay.active = false;
            std::cout << "[overlay] " << overlay.image.name << " is no longer in memory" << std::endl;
        }
        else if (!overlay.active && present)
        {
            arrived.push_back(&overlay);
        }
    }

    for (RegisteredOverlay *overlay : arrived)
    {
        for (size_t i = 0; i < overlay->image.functionCount; ++i)
        {
            replaceFunction(overlay->image.functions[i].address, overlay->image.functions[i].function);
        }
        overlay->active = true;
        std::cout << "[overlay] " << overlay->image.name << ": " << overlay->image.functionCount
                  << " entries at 0x" << std::hex << overlay->image.code[0].start << std::dec << std::endl;
    }
}

std::string PS2Runtime::describeActiveOverlays() const
{
    std::string names;
    for (const RegisteredOverlay &overlay : registeredOverlays())
    {
        if (overlay.active)
        {
            names += names.empty() ? "" : ",";
            names += overlay.image.name;
        }
    }
    return names.empty() ? "none" : names;
}

const char *describeGuestBranchKind(PS2Runtime::GuestBranchKind kind)
{
    switch (kind)
    {
    case PS2Runtime::GuestBranchKind::DirectJump:
        return "DirectJump";
    case PS2Runtime::GuestBranchKind::DirectCall:
        return "DirectCall";
    case PS2Runtime::GuestBranchKind::IndirectJump:
        return "IndirectJump";
    case PS2Runtime::GuestBranchKind::IndirectCall:
        return "IndirectCall";
    case PS2Runtime::GuestBranchKind::Return:
        return "Return";
    default:
        return "Unknown";
    }
}

PS2Runtime::RecompiledFunction PS2Runtime::lookupFunction(uint32_t address)
{
    GuestThreadState &guest = t_guest;
    pushDispatchPc(guest.history, address);
    // Guest code is about to be entered: whatever was unwinding has arrived.
    guest.unwinding = false;

    uint32_t slot = 0u;
    if (generatedFunctionTableSlot(address, slot))
    {
        RecompiledFunction fn = g_ps2RecompiledFunctionTable[slot];
        if (fn != nullptr)
        {
            return fn;
        }
    }

    std::cerr << "Error: No exact recompiled function for guest PC 0x" << std::hex << address
              << " tableBase=0x" << g_ps2RecompiledFunctionTableBase
              << " tableEnd=0x" << g_ps2RecompiledFunctionTableEnd
              << " codeRegion=" << (m_memory.isCodeAddress(address) ? "yes" : "no")
              << " trace=" << formatDispatchHistory()
              << std::dec << std::endl;

    static RecompiledFunction missingFunction = [](uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t badPc = ctx->pc;
        runtime->reportMissingFunction(rdram,
                                       ctx,
                                       badPc,
                                       0u,
                                       PS2Runtime::GuestBranchKind::IndirectJump,
                                       "dispatch");
    };

    return missingFunction;
}

void PS2Runtime::setMissingFunctionPolicy(MissingFunctionPolicy policy)
{
    m_missingFunctionPolicy.store(static_cast<uint32_t>(policy), std::memory_order_release);
}

PS2Runtime::MissingFunctionPolicy PS2Runtime::missingFunctionPolicy() const
{
    return static_cast<MissingFunctionPolicy>(m_missingFunctionPolicy.load(std::memory_order_acquire));
}

void PS2Runtime::resetMissingFunctionReportOnce()
{
    m_missingFunctionReported.store(false, std::memory_order_release);
}

void PS2Runtime::reportMissingFunction(uint8_t *rdram,
                                       R5900Context *ctx,
                                       uint32_t targetPc,
                                       uint32_t sourcePc,
                                       GuestBranchKind kind,
                                       const char *debugName)
{
    const MissingFunctionPolicy policy = missingFunctionPolicy();
    const bool firstReport = !m_missingFunctionReported.exchange(true, std::memory_order_acq_rel);
    if (firstReport && g_callHistoryWanted)
        printCallHistory(400u);

    const uint32_t pc = ctx->pc;
    const uint32_t ra = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[31], 0));
    const uint32_t sp = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[29], 0));
    const uint32_t gp = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[28], 0));
    const uint32_t a0 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[4], 0));
    const uint32_t a1 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[5], 0));
    const uint32_t a2 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[6], 0));
    const uint32_t a3 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[7], 0));
    const uint32_t s0 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[16], 0));
    const uint32_t s1 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[17], 0));
    const uint32_t v0 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[2], 0));
    const uint32_t v1 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[3], 0));

    auto readGuestU32At = [rdram](uint32_t addr, uint32_t &out) -> bool
    {
        // TODO this !rdram exist only because of test fix those test later
        if (!rdram || addr > PS2_RAM_SIZE - sizeof(uint32_t))
        {
            out = 0u;
            return false;
        }

        std::memcpy(&out, rdram + addr, sizeof(uint32_t));
        return true;
    };

    auto readGuestU32Offset = [&readGuestU32At](uint32_t base, uint32_t offset, uint32_t &out) -> bool
    {
        if (base > PS2_RAM_SIZE - sizeof(uint32_t) || offset > PS2_RAM_SIZE - sizeof(uint32_t) - base)
        {
            out = 0u;
            return false;
        }

        return readGuestU32At(base + offset, out);
    };

    uint32_t a0Word0 = 0u;
    uint32_t a0Word4 = 0u;
    uint32_t a0Word8 = 0u;
    uint32_t a0WordC = 0u;
    const bool a0Readable =
        readGuestU32Offset(a0, 0x00u, a0Word0) &&
        readGuestU32Offset(a0, 0x04u, a0Word4) &&
        readGuestU32Offset(a0, 0x08u, a0Word8) &&
        readGuestU32Offset(a0, 0x0cu, a0WordC);

    uint32_t s0Word0 = 0u;
    uint32_t s0Word4 = 0u;
    uint32_t s0Word8 = 0u;
    uint32_t s0WordC = 0u;
    const bool s0Readable =
        readGuestU32Offset(s0, 0x00u, s0Word0) &&
        readGuestU32Offset(s0, 0x04u, s0Word4) &&
        readGuestU32Offset(s0, 0x08u, s0Word8) &&
        readGuestU32Offset(s0, 0x0cu, s0WordC);

    uint32_t recordWord0 = 0u;
    uint32_t recordWord4 = 0u;
    uint32_t recordWord8 = 0u;
    uint32_t recordWordC = 0u;
    const bool recordReadable =
        s0Readable && s0Word4 != 0u &&
        readGuestU32Offset(s0Word4, 0x00u, recordWord0) &&
        readGuestU32Offset(s0Word4, 0x04u, recordWord4) &&
        readGuestU32Offset(s0Word4, 0x08u, recordWord8) &&
        readGuestU32Offset(s0Word4, 0x0cu, recordWordC);

    uint32_t vtableSlot0 = 0u;
    uint32_t vtableSlot4 = 0u;
    uint32_t vtableSlot8 = 0u;
    uint32_t vtableSlotC = 0u;
    const bool vtableReadable =
        a0Readable && a0Word0 != 0u &&
        readGuestU32Offset(a0Word0, 0x00u, vtableSlot0) &&
        readGuestU32Offset(a0Word0, 0x04u, vtableSlot4) &&
        readGuestU32Offset(a0Word0, 0x08u, vtableSlot8) &&
        readGuestU32Offset(a0Word0, 0x0cu, vtableSlotC);

    if (firstReport)
    {
        std::ostringstream oss;
        oss << "[guest-branch:missing-target] kind=" << describeGuestBranchKind(kind)
            << " op=" << (debugName ? debugName : "<unknown>")
            << " source=0x" << std::hex << sourcePc
            << " target=0x" << targetPc
            << " pc=0x" << pc
            << " ra=0x" << ra
            << " sp=0x" << sp
            << " gp=0x" << gp
            << " a0=0x" << a0
            << " a1=0x" << a1
            << " a2=0x" << a2
            << " a3=0x" << a3
            << " s0=0x" << s0
            << " s1=0x" << s1
            << " v0=0x" << v0
            << " v1=0x" << v1
            << " a0Readable=" << (a0Readable ? "yes" : "no")
            << " a0[0]=0x" << a0Word0
            << " a0[4]=0x" << a0Word4
            << " a0[8]=0x" << a0Word8
            << " a0[c]=0x" << a0WordC
            << " s0Readable=" << (s0Readable ? "yes" : "no")
            << " s0[0]=0x" << s0Word0
            << " s0[4]=0x" << s0Word4
            << " s0[8]=0x" << s0Word8
            << " s0[c]=0x" << s0WordC
            << " recordReadable=" << (recordReadable ? "yes" : "no")
            << " record[0]=0x" << recordWord0
            << " record[4]=0x" << recordWord4
            << " record[8]=0x" << recordWord8
            << " record[c]=0x" << recordWordC
            << " vtableReadable=" << (vtableReadable ? "yes" : "no")
            << " vtbl[0]=0x" << vtableSlot0
            << " vtbl[4]=0x" << vtableSlot4
            << " vtbl[8]=0x" << vtableSlot8
            << " vtbl[c]=0x" << vtableSlotC
            << " codeRegion=" << (m_memory.isCodeAddress(targetPc) ? "yes" : "no")
            << " policy=" << static_cast<uint32_t>(policy)
            << " trace=" << formatDispatchHistory()
            << std::dec;

        static std::mutex s_missingFunctionLogMutex;
        {
            std::lock_guard<std::mutex> lock(s_missingFunctionLogMutex);
            std::cerr << oss.str() << std::endl;
        }
    }

    if (firstReport && policy == MissingFunctionPolicy::BreakOnce)
    {
#if defined(_MSC_VER)
        __debugbreak();
#endif // TODO others breakpoints
    }

    if (ctx)
    {
        ctx->pc = targetPc;
    }

    if (policy == MissingFunctionPolicy::Stop)
    {
        requestStop();
    }
}

bool PS2Runtime::dispatchGuestBranch(uint8_t *rdram,
                                     R5900Context *ctx,
                                     uint32_t targetPc,
                                     uint32_t sourcePc,
                                     uint32_t fallthroughPc,
                                     GuestBranchKind kind,
                                     const char *debugName)
{
    ctx->pc = targetPc;
    const bool isCall = (kind == GuestBranchKind::DirectCall || kind == GuestBranchKind::IndirectCall);
    if (g_callHistoryWanted) [[unlikely]]
        pushCall(isCall ? 'c' : 'j', targetPc, sourcePc, ctx);

    // Every inter-function transfer is also a deterministic EE safe point.
    // Backward edges inside generated functions use eeCheckpointDue(), while
    // this charge bounds straight-line call chains that have no local loop.
    if (m_eeScheduler && m_eeScheduler->checkpointDue(EeScheduler::kGuestDispatchCycles))
    {
        t_guest.unwinding = true;
        return false;
    }

    if (!isCall)
    {
        if (!hasFunction(targetPc))
        {
            reportMissingFunction(rdram, ctx, targetPc, sourcePc, kind, debugName);
        }

        ctx->pc = targetPc;
        // A jump leaves this function for the scheduler to continue at the
        // target; a return (strict diagnostics only) goes to the caller.
        if (kind != GuestBranchKind::Return)
        {
            t_guest.unwinding = true;
        }
        return false;
    }

    // The table is looked in once for a call, and the thread's own state
    // found once (see GuestThreadState).
    uint32_t slot = 0u;
    RecompiledFunction targetFn = generatedFunctionTableSlot(targetPc, slot) ? g_ps2RecompiledFunctionTable[slot] : nullptr;
    if (targetFn == nullptr)
    {
        reportMissingFunction(rdram, ctx, targetPc, sourcePc, kind, debugName);

        const MissingFunctionPolicy policy = missingFunctionPolicy();

        if (policy == MissingFunctionPolicy::SkipCallDebug && isCall)
        {
            ctx->pc = fallthroughPc;
            return true;
        }

        if (policy == MissingFunctionPolicy::ContinueToTarget)
        {
            ctx->pc = targetPc;
            // if you need the app to keep open to open debug pannel change this to false
            return false;
        }

        return false;
    }

    GuestThreadState &guest = t_guest;
    pushDispatchPc(guest.history, targetPc);
    // Guest code is about to be entered: whatever was unwinding has arrived.
    guest.unwinding = false;
    const uint32_t entryPc = ctx->pc;
    targetFn(rdram, ctx, this);
    if (g_callHistoryWanted) [[unlikely]]
        pushCall((ctx->pc == entryPc || ctx->pc == fallthroughPc) ? 'r' : 'x', ctx->pc, targetPc, ctx);

    if (isStopRequested() || ctx->pc == 0u)
    {
        return false;
    }

    // Something below is on its way back to the scheduler. ctx->pc is where
    // it resumes, and may well equal this call's target or return address
    // without this call having returned: a recursive function interrupted as
    // it calls itself comes back up through its own outer call with pc at
    // its own entry, and taking that for "the callee finished" (as below)
    // carried on in the outer caller with the inner call's stack.
    if (guest.unwinding)
    {
        return false;
    }

    // A callee that never touched pc is a host function (an HLE stub, an
    // override): it has returned.
    if (ctx->pc == entryPc)
    {
        ctx->pc = fallthroughPc;
    }

    return ctx->pc == fallthroughPc;
}

void PS2Runtime::SignalException(R5900Context *ctx, PS2Exception exception)
{
    if (exception == EXCEPTION_INTEGER_OVERFLOW)
    {
        HandleIntegerOverflow(ctx);
        return;
    }

    raiseCop0Exception(ctx, static_cast<uint32_t>(exception),
                       exception == EXCEPTION_TLB_REFILL);
}

void PS2Runtime::executeVU0Microprogram(uint8_t *rdram, R5900Context *ctx, uint32_t address)
{
    (void)rdram;

    uint8_t *const vu0Code = m_memory.getVU0Code();
    uint8_t *const vu0Data = m_memory.getVU0Data();
    const uint32_t startPC = address & ~0x7u;

    if (!vu0Code || !vu0Data || startPC + 8u > PS2_VU0_CODE_SIZE)
    {
        seedVu0IdleSuccess(ctx);
        return;
    }

    // PS2X_VU0_TRACE=<secs>[:count]: log the microprograms the EE calls on
    // VU0 from that emulated time on (64 of them unless told): where each
    // starts and who called it ($ra), the registers that are not zero going
    // in, and the ones the program changed coming out. A program called at
    // an address that is not one of its routines shows here first.
    static const uint64_t vu0TraceTick = envEmulatedTicks("PS2X_VU0_TRACE");
    static int vu0TraceLeft = []
    {
        const char *value = std::getenv("PS2X_VU0_TRACE");
        const char *colon = value ? std::strchr(value, ':') : nullptr;
        return colon ? std::atoi(colon + 1) : 64;
    }();
    const bool vu0Trace = vu0TraceTick != 0u && vu0TraceLeft > 0 && m_eeScheduler &&
                          m_eeScheduler->currentVSyncTick() >= vu0TraceTick;
    __m128 vu0TraceBefore[32];
    auto vu0TraceRegs = [&](const char *when, bool changedOnly)
    {
        std::fprintf(stderr, "[vu0] %s pc=0x%x ra=0x%08x vi", when, startPC,
                     static_cast<uint32_t>(_mm_extract_epi32(ctx->r[31], 0)));
        for (uint32_t r = 0; r < 16u; ++r)
            std::fprintf(stderr, " %u", ctx->vi[r]);
        std::fprintf(stderr, "\n");
        for (uint32_t r = 1; r < 32u; ++r)
        {
            alignas(16) float f[4];
            _mm_store_ps(f, ctx->vu0_vf[r]);
            const bool show = changedOnly
                                  ? std::memcmp(&vu0TraceBefore[r], &ctx->vu0_vf[r], sizeof(__m128)) != 0
                                  : (f[0] != 0.0f || f[1] != 0.0f || f[2] != 0.0f || f[3] != 0.0f);
            if (show)
                std::fprintf(stderr, "[vu0]   vf%02u % .6g % .6g % .6g % .6g\n", r, f[0], f[1], f[2], f[3]);
        }
    };
    if (vu0Trace)
    {
        --vu0TraceLeft;
        vu0TraceRegs("call", false);
        std::memcpy(vu0TraceBefore, ctx->vu0_vf, sizeof(vu0TraceBefore));
    }

    m_vu0.reset();
    copyVu0ContextToState(ctx, m_vu0.state());
    // VCALLMS runs the program to its E bit before the EE continues. The cap
    // only guards against a program that never ends; a vertex loop over a
    // full VU0 data memory takes far more than a few thousand cycles.
    m_vu0.execute(vu0Code, PS2_VU0_CODE_SIZE,
                  vu0Data, PS2_VU0_DATA_SIZE,
                  m_gs, &m_memory,
                  startPC, 0u, ctx->vu0_itop, 1u << 20);
    copyVu0StateToContext(m_vu0.state(), ctx);
    if (vu0Trace)
    {
        std::fprintf(stderr, "[vu0]   ended pc=0x%x cycles=%llu\n", m_vu0.state().pc,
                     static_cast<unsigned long long>(m_vu0.state().cycles));
        vu0TraceRegs("done", true);
    }
}

void PS2Runtime::vu0StartMicroProgram(uint8_t *rdram, R5900Context *ctx, uint32_t address)
{
    // VCALLMS and VCALLMSR both route here.
    executeVU0Microprogram(rdram, ctx, address);
}

void PS2Runtime::handleSyscall(uint8_t *rdram, R5900Context *ctx)
{
    handleSyscall(rdram, ctx, 0);
}

void PS2Runtime::handleSyscall(uint8_t *rdram, R5900Context *ctx, uint32_t encodedSyscallId)
{
    if (ctx->in_delay_slot)
    {
        throw std::runtime_error("Attempted to execute a syscall inside a branch delay slot! "
                                 "This breaks the atomic basic block model and is structurally unsupported by the emulator.");
    }

    const uint32_t syscallId = (encodedSyscallId != 0u)
                                   ? encodedSyscallId
                                   : getRegU32(ctx, 3); // $v1 / $3 is the EE kernel syscall number

    if (ps2_syscalls::dispatchNumericSyscall(syscallId, rdram, ctx, this))
    {
        return;
    }

    // God help you
    ps2_syscalls::TODO(rdram, ctx, this, encodedSyscallId);
}

void PS2Runtime::handleBreak(uint8_t *rdram, R5900Context *ctx)
{
    raiseCop0Exception(ctx, EXCEPTION_BREAKPOINT);
}

void PS2Runtime::drainCompletedDmacHandlers(uint8_t *rdram)
{
    for (uint32_t cause : m_memory.consumeCompletedDmacCauses())
    {
        ps2_syscalls::dispatchDmacHandlersForCause(rdram, this, cause);
    }
}

void PS2Runtime::handleTrap(uint8_t *rdram, R5900Context *ctx)
{
    raiseCop0Exception(ctx, EXCEPTION_TRAP);
}

void PS2Runtime::handleTLBR(uint8_t *rdram, R5900Context *ctx)
{
    uint32_t vpn = 0;
    uint32_t pfn = 0;
    uint32_t mask = 0;
    bool valid = false;

    const uint32_t index = ctx->cop0_index & 0x3Fu;
    if (!m_memory.tlbRead(index, vpn, pfn, mask, valid))
    {
        raiseCop0Exception(ctx, EXCEPTION_RESERVED_INSTRUCTION);
        return;
    }

    // Preserve low ASID bits in EntryHi.
    ctx->cop0_entryhi = (ctx->cop0_entryhi & 0x00000FFFu) | (vpn & 0xFFFFF000u);
    ctx->cop0_entrylo0 = (ctx->cop0_entrylo0 & ~0x03FFFFC2u) |
                         ((pfn & 0x000FFFFFu) << 6) |
                         (valid ? 0x2u : 0u);
    ctx->cop0_pagemask = mask & 0x01FFE000u;
}

void PS2Runtime::handleTLBWI(uint8_t *rdram, R5900Context *ctx)
{
    const uint32_t index = ctx->cop0_index & 0x3Fu;
    const uint32_t vpn = ctx->cop0_entryhi & 0xFFFFF000u;
    const uint32_t pfn = (ctx->cop0_entrylo0 >> 6) & 0x000FFFFFu;
    const uint32_t mask = ctx->cop0_pagemask & 0x01FFE000u;
    const bool valid = (ctx->cop0_entrylo0 & 0x2u) != 0u;

    if (!m_memory.tlbWrite(index, vpn, pfn, mask, valid))
    {
        raiseCop0Exception(ctx, EXCEPTION_RESERVED_INSTRUCTION);
    }
}

void PS2Runtime::handleTLBWR(uint8_t *rdram, R5900Context *ctx)
{
    const uint32_t entryCount = static_cast<uint32_t>(m_memory.tlbEntryCount());
    if (entryCount == 0)
    {
        raiseCop0Exception(ctx, EXCEPTION_RESERVED_INSTRUCTION);
        return;
    }

    const uint32_t wired = std::min(ctx->cop0_wired, entryCount - 1);
    uint32_t random = ctx->cop0_random % entryCount;
    if (random < wired)
    {
        random = wired;
    }

    const uint32_t vpn = ctx->cop0_entryhi & 0xFFFFF000u;
    const uint32_t pfn = (ctx->cop0_entrylo0 >> 6) & 0x000FFFFFu;
    const uint32_t mask = ctx->cop0_pagemask & 0x01FFE000u;
    const bool valid = (ctx->cop0_entrylo0 & 0x2u) != 0u;

    if (!m_memory.tlbWrite(random, vpn, pfn, mask, valid))
    {
        raiseCop0Exception(ctx, EXCEPTION_RESERVED_INSTRUCTION);
        return;
    }

    // Keep COP0 bookkeeping in sync with the selected slot.
    ctx->cop0_index = (ctx->cop0_index & ~0x3Fu) | (random & 0x3Fu);
    ctx->cop0_random = (random <= wired) ? (entryCount - 1) : (random - 1);
}

void PS2Runtime::handleTLBP(uint8_t *rdram, R5900Context *ctx)
{
    const int32_t index = m_memory.tlbProbe(ctx->cop0_entryhi & 0xFFFFF000u);
    if (index >= 0)
    {
        ctx->cop0_index = (ctx->cop0_index & ~0x8000003Fu) |
                          (static_cast<uint32_t>(index) & 0x3Fu);
    }
    else
    {
        // MIPS sets probe failure bit (P) in Index[31].
        ctx->cop0_index |= 0x80000000u;
    }
}

void PS2Runtime::clearLLBit(R5900Context *ctx)
{
    // LL/SC reservation is tracked separately from COP0 Status.
    ctx->llbit = 0;
    ctx->lladdr = 0;
}

uint32_t PS2Runtime::alignGuestHeapValue(uint32_t value, uint32_t alignment)
{
    if (alignment == 0)
    {
        return value;
    }

    const uint32_t mask = alignment - 1u;
    if (value > (std::numeric_limits<uint32_t>::max() - mask))
    {
        return std::numeric_limits<uint32_t>::max();
    }
    return (value + mask) & ~mask;
}

bool PS2Runtime::isGuestHeapAlignmentValid(uint32_t alignment)
{
    return alignment != 0u && (alignment & (alignment - 1u)) == 0u;
}

uint32_t PS2Runtime::normalizeGuestHeapAlignment(uint32_t alignment)
{
    if (!isGuestHeapAlignmentValid(alignment))
    {
        return kGuestHeapDefaultAlignment;
    }
    return std::max(alignment, kGuestHeapDefaultAlignment);
}

uint32_t PS2Runtime::clampGuestHeapBase(uint32_t guestBase) const
{
    uint32_t normalized = guestBase;
    if (normalized >= PS2_RAM_SIZE)
    {
        normalized &= PS2_RAM_MASK;
    }
    const uint32_t hardLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
    return std::min(normalized, hardLimit);
}

uint32_t PS2Runtime::clampGuestHeapLimit(uint32_t guestLimit) const
{
    const uint32_t hardLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
    if (guestLimit == 0u || guestLimit > hardLimit)
    {
        return hardLimit;
    }
    return guestLimit;
}

void PS2Runtime::resetGuestHeapLocked(uint32_t guestBase, uint32_t guestLimit)
{
    uint32_t base = alignGuestHeapValue(clampGuestHeapBase(guestBase), kGuestHeapDefaultAlignment);
    uint32_t limit = clampGuestHeapLimit(guestLimit);
    if (base == 0u)
    {
        const uint32_t fallbackBase = (m_guestHeapSuggestedBase != 0u) ? m_guestHeapSuggestedBase : kGuestHeapDefaultBase;
        base = alignGuestHeapValue(clampGuestHeapBase(fallbackBase), kGuestHeapDefaultAlignment);
    }

    if (limit <= base)
    {
        base = alignGuestHeapValue(clampGuestHeapBase(m_guestHeapSuggestedBase), kGuestHeapDefaultAlignment);
        limit = clampGuestHeapLimit(0u);
    }

    if (limit <= base)
    {
        base = 0u;
        limit = 0u;
    }

    m_guestHeapBlocks.clear();
    if (limit > base)
    {
        m_guestHeapBlocks.push_back({base, limit - base, true});
    }

    m_guestHeapBase = base;
    m_guestHeapEnd = base;
    m_guestHeapLimit = limit;
    m_guestHeapConfigured = true;
}

void PS2Runtime::ensureGuestHeapInitializedLocked()
{
    if (m_guestHeapConfigured)
    {
        return;
    }

    const uint32_t suggested = (m_guestHeapSuggestedBase == 0u) ? kGuestHeapDefaultBase : m_guestHeapSuggestedBase;
    resetGuestHeapLocked(suggested, clampGuestHeapLimit(0u));
}

int32_t PS2Runtime::findGuestHeapBlockIndexLocked(uint32_t guestAddr) const
{
    const uint32_t normalizedAddr = guestAddr & PS2_RAM_MASK;
    for (size_t i = 0; i < m_guestHeapBlocks.size(); ++i)
    {
        const GuestHeapBlock &block = m_guestHeapBlocks[i];
        if (!block.free && block.addr == normalizedAddr)
        {
            return static_cast<int32_t>(i);
        }
    }
    return -1;
}

uint32_t PS2Runtime::allocateGuestBlockLocked(uint32_t size, uint32_t alignment)
{
    if (size == 0u)
    {
        return 0u;
    }

    const uint32_t normalizedAlignment = normalizeGuestHeapAlignment(alignment);
    if (size > (std::numeric_limits<uint32_t>::max() - (kGuestHeapDefaultAlignment - 1u)))
    {
        return 0u;
    }

    const uint32_t allocSize = alignGuestHeapValue(size, kGuestHeapDefaultAlignment);
    if (allocSize == 0u)
    {
        return 0u;
    }

    for (size_t i = 0; i < m_guestHeapBlocks.size(); ++i)
    {
        const GuestHeapBlock block = m_guestHeapBlocks[i];
        if (!block.free)
        {
            continue;
        }

        const uint64_t blockStart = block.addr;
        const uint64_t blockEnd = blockStart + static_cast<uint64_t>(block.size);
        const uint32_t alignedAddr = alignGuestHeapValue(block.addr, normalizedAlignment);
        if (alignedAddr < block.addr)
        {
            continue;
        }

        const uint64_t alignedStart = alignedAddr;
        if (alignedStart > blockEnd)
        {
            continue;
        }

        const uint64_t allocEnd = alignedStart + static_cast<uint64_t>(allocSize);
        if (allocEnd > blockEnd)
        {
            continue;
        }

        const uint32_t prefixSize = static_cast<uint32_t>(alignedStart - blockStart);
        const uint32_t suffixSize = static_cast<uint32_t>(blockEnd - allocEnd);

        std::vector<GuestHeapBlock> replacement;
        replacement.reserve(3);
        if (prefixSize > 0u)
        {
            replacement.push_back({block.addr, prefixSize, true});
        }
        replacement.push_back({alignedAddr, allocSize, false});
        if (suffixSize > 0u)
        {
            replacement.push_back({static_cast<uint32_t>(allocEnd), suffixSize, true});
        }

        m_guestHeapBlocks.erase(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(i));
        m_guestHeapBlocks.insert(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(i),
                                 replacement.begin(),
                                 replacement.end());

        m_guestHeapEnd = std::max(m_guestHeapEnd, static_cast<uint32_t>(allocEnd));
        return alignedAddr;
    }

    return 0u;
}

void PS2Runtime::coalesceGuestHeapLocked()
{
    if (m_guestHeapBlocks.empty())
    {
        return;
    }

    size_t i = 1;
    while (i < m_guestHeapBlocks.size())
    {
        GuestHeapBlock &prev = m_guestHeapBlocks[i - 1];
        GuestHeapBlock &curr = m_guestHeapBlocks[i];
        const uint64_t prevEnd = static_cast<uint64_t>(prev.addr) + static_cast<uint64_t>(prev.size);
        if (prev.free && curr.free && prevEnd == curr.addr)
        {
            prev.size += curr.size;
            m_guestHeapBlocks.erase(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(i));
            continue;
        }
        ++i;
    }
}

void PS2Runtime::freeGuestBlockLocked(uint32_t guestAddr)
{
    const int32_t index = findGuestHeapBlockIndexLocked(guestAddr);
    if (index < 0)
    {
        return;
    }

    m_guestHeapBlocks[static_cast<size_t>(index)].free = true;
    coalesceGuestHeapLocked();
}

void PS2Runtime::configureGuestHeap(uint32_t guestBase, uint32_t guestLimit)
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    uint32_t normalizedBase = alignGuestHeapValue(clampGuestHeapBase(guestBase), kGuestHeapDefaultAlignment);
    if (normalizedBase == 0u)
    {
        normalizedBase = (m_guestHeapSuggestedBase != 0u) ? m_guestHeapSuggestedBase : kGuestHeapDefaultBase;
    }
    m_guestHeapSuggestedBase = normalizedBase;
    resetGuestHeapLocked(normalizedBase, guestLimit);
}

uint32_t PS2Runtime::guestMalloc(uint32_t size, uint32_t alignment)
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    ensureGuestHeapInitializedLocked();
    return allocateGuestBlockLocked(size, alignment);
}

uint32_t PS2Runtime::guestCalloc(uint32_t count, uint32_t size, uint32_t alignment)
{
    if (count == 0u || size == 0u)
    {
        return 0u;
    }
    if (count > (std::numeric_limits<uint32_t>::max() / size))
    {
        return 0u;
    }

    const uint32_t totalSize = count * size;
    const uint32_t guestAddr = guestMalloc(totalSize, alignment);
    if (guestAddr != 0u)
    {
        uint8_t *rdram = m_memory.getRDRAM();
        if (rdram)
        {
            uint32_t physAddr = guestAddr & PS2_RAM_MASK;
            if (physAddr + totalSize <= PS2_RAM_SIZE)
                std::memset(rdram + physAddr, 0, totalSize);
        }
    }

    return guestAddr;
}

uint32_t PS2Runtime::guestRealloc(uint32_t guestAddr, uint32_t newSize, uint32_t alignment)
{
    if (guestAddr == 0u)
    {
        return guestMalloc(newSize, alignment);
    }
    if (newSize == 0u)
    {
        guestFree(guestAddr);
        return 0u;
    }

    if (newSize > (std::numeric_limits<uint32_t>::max() - (kGuestHeapDefaultAlignment - 1u)))
    {
        return 0u;
    }

    const uint32_t normalizedAlignment = normalizeGuestHeapAlignment(alignment);
    const uint32_t requestedSize = alignGuestHeapValue(newSize, kGuestHeapDefaultAlignment);

    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    ensureGuestHeapInitializedLocked();

    const int32_t index = findGuestHeapBlockIndexLocked(guestAddr);
    if (index < 0)
    {
        return 0u;
    }

    const size_t blockIndex = static_cast<size_t>(index);
    const uint32_t oldAddr = m_guestHeapBlocks[blockIndex].addr;
    const uint32_t oldSize = m_guestHeapBlocks[blockIndex].size;

    if (requestedSize <= oldSize)
    {
        if (requestedSize < oldSize)
        {
            const uint32_t tailAddr = oldAddr + requestedSize;
            const uint32_t tailSize = oldSize - requestedSize;
            m_guestHeapBlocks[blockIndex].size = requestedSize;
            m_guestHeapBlocks.insert(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(blockIndex + 1u),
                                     GuestHeapBlock{tailAddr, tailSize, true});
            coalesceGuestHeapLocked();
        }
        return oldAddr;
    }

    if (blockIndex + 1u < m_guestHeapBlocks.size())
    {
        GuestHeapBlock &next = m_guestHeapBlocks[blockIndex + 1u];
        const uint64_t blockEnd = static_cast<uint64_t>(m_guestHeapBlocks[blockIndex].addr) +
                                  static_cast<uint64_t>(m_guestHeapBlocks[blockIndex].size);
        if (next.free && blockEnd == next.addr)
        {
            const uint64_t combined = static_cast<uint64_t>(m_guestHeapBlocks[blockIndex].size) +
                                      static_cast<uint64_t>(next.size);
            if (combined >= requestedSize)
            {
                const uint32_t extraNeeded = requestedSize - m_guestHeapBlocks[blockIndex].size;
                m_guestHeapBlocks[blockIndex].size = requestedSize;
                if (next.size == extraNeeded)
                {
                    m_guestHeapBlocks.erase(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(blockIndex + 1u));
                }
                else
                {
                    next.addr += extraNeeded;
                    next.size -= extraNeeded;
                }
                m_guestHeapEnd = std::max(m_guestHeapEnd, oldAddr + requestedSize);
                return oldAddr;
            }
        }
    }

    const uint32_t newAddr = allocateGuestBlockLocked(newSize, normalizedAlignment);
    if (newAddr == 0u)
    {
        return 0u;
    }

    uint8_t *rdram = m_memory.getRDRAM();
    if (rdram)
    {
        const uint32_t copyBytes = std::min(oldSize, newSize);
        uint32_t dstPhys = newAddr & PS2_RAM_MASK;
        uint32_t srcPhys = oldAddr & PS2_RAM_MASK;
        if (dstPhys + copyBytes <= PS2_RAM_SIZE && srcPhys + copyBytes <= PS2_RAM_SIZE)
            std::memmove(rdram + dstPhys, rdram + srcPhys, copyBytes);
    }

    freeGuestBlockLocked(oldAddr);
    return newAddr;
}

void PS2Runtime::guestFree(uint32_t guestAddr)
{
    if (guestAddr == 0u)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    ensureGuestHeapInitializedLocked();
    freeGuestBlockLocked(guestAddr);
}

uint32_t PS2Runtime::guestHeapBase() const
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    return m_guestHeapConfigured ? m_guestHeapBase : m_guestHeapSuggestedBase;
}

uint32_t PS2Runtime::guestHeapEnd() const
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    return m_guestHeapConfigured ? m_guestHeapEnd : m_guestHeapSuggestedBase;
}

uint32_t PS2Runtime::guestHeapLimit() const
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    return m_guestHeapConfigured ? m_guestHeapLimit : m_guestHeapSuggestedBase;
}

uint32_t PS2Runtime::reserveAsyncCallbackStack(uint32_t size, uint32_t alignment)
{
    if (size == 0u)
    {
        return 0u;
    }

    const uint32_t normalizedAlignment = normalizeGuestHeapAlignment(alignment);
    const uint32_t allocSize = alignGuestHeapValue(size, kGuestHeapDefaultAlignment);
    if (allocSize == 0u)
    {
        return 0u;
    }

    std::lock_guard<std::mutex> lock(m_asyncCallbackStackMutex);
    uint32_t top = m_asyncCallbackStackTop;
    if (top > PS2_RAM_SIZE)
    {
        top = PS2_RAM_SIZE;
    }
    top &= ~(kGuestHeapDefaultAlignment - 1u);

    if (top <= allocSize)
    {
        return 0u;
    }

    uint32_t base = top - allocSize;
    base &= ~(normalizedAlignment - 1u);
    if (base < m_asyncCallbackStackFloor || base >= top)
    {
        return 0u;
    }

    m_asyncCallbackStackTop = base;
    return top - 0x10u;
}

uint8_t PS2Runtime::Load8(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    try
    {
        return m_memory.read8(vaddr);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD);
        return 0;
    }
}

uint16_t PS2Runtime::Load16(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    try
    {
        return m_memory.read16(vaddr);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD);
        return 0;
    }
}

uint32_t PS2Runtime::Load32(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    try
    {
        return m_memory.read32(vaddr);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD);
        return 0;
    }
}

uint64_t PS2Runtime::Load64(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    try
    {
        return m_memory.read64(vaddr);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD);
        return 0;
    }
}

__m128i PS2Runtime::Load128(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    try
    {
        return m_memory.read128(vaddr);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD);
        return _mm_setzero_si128();
    }
}

void PS2Runtime::Store8(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint8_t value)
{
    ps2TraceGuestWrite(rdram, vaddr, 1u, value, 0u, "WRITE8", ctx);
    try
    {
        m_memory.write8(vaddr, value);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_STORE);
    }
}

void PS2Runtime::Store16(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint16_t value)
{
    ps2TraceGuestWrite(rdram, vaddr, 2u, value, 0u, "WRITE16", ctx);
    try
    {
        m_memory.write16(vaddr, value);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_STORE);
    }
}

void PS2Runtime::Store32(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint32_t value)
{
    ps2TraceGuestWrite(rdram, vaddr, 4u, value, 0u, "WRITE32", ctx);
    try
    {
        m_memory.write32(vaddr, value);
        drainCompletedDmacHandlers(rdram);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_STORE);
    }
}

void PS2Runtime::Store64(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint64_t value)
{
    ps2TraceGuestWrite(rdram, vaddr, 8u, value, 0u, "WRITE64", ctx);
    try
    {
        m_memory.write64(vaddr, value);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_STORE);
    }
}

void PS2Runtime::Store128(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, __m128i value)
{
    alignas(16) uint64_t _parts[2];
    _mm_storeu_si128(reinterpret_cast<__m128i *>(_parts), value);
    ps2TraceGuestWrite(rdram, vaddr, 16u, _parts[0], _parts[1], "WRITE128", ctx);
    try
    {
        m_memory.write128(vaddr, value);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_STORE);
    }
}

void PS2Runtime::kickGifDmaChainFromMMIO(uint8_t *rdram,
                                         R5900Context *ctx,
                                         uint32_t dPcrValue,
                                         uint32_t dStatValue,
                                         uint32_t tadr,
                                         uint32_t chcr)
{
    constexpr uint32_t D_PCR = 0x1000E020u;
    constexpr uint32_t D_STAT = 0x1000E010u;
    constexpr uint32_t GIF_TADR = 0x1000A030u;
    constexpr uint32_t GIF_CHCR = 0x1000A000u;

    ps2TraceGuestWrite(rdram, D_PCR, 4u, dPcrValue, 0u, "WRITE32", ctx);
    m_memory.writeIORegister(D_PCR, dPcrValue);
    ps2TraceGuestWrite(rdram, D_STAT, 4u, dStatValue, 0u, "WRITE32", ctx);
    m_memory.writeIORegister(D_STAT, dStatValue);
    ps2TraceGuestWrite(rdram, GIF_TADR, 4u, tadr, 0u, "WRITE32", ctx);
    m_memory.writeIORegister(GIF_TADR, tadr);
    ps2TraceGuestWrite(rdram, GIF_CHCR, 4u, chcr, 0u, "WRITE32", ctx);
    // The native shortcuts call into the GS directly; with the GS on its own
    // thread everything goes through the packet queue instead.
    if (!GsWorker::running() && m_memory.tryProcessNativeGifImageUploadChain(m_gs, tadr, chcr))
    {
        drainCompletedDmacHandlers(rdram);
        return;
    }
    if (!GsWorker::running() && m_memory.tryProcessNativeGifPackedChain(m_gs, tadr, chcr))
    {
        drainCompletedDmacHandlers(rdram);
        return;
    }
    m_memory.writeIORegister(GIF_CHCR, chcr);
    m_memory.processPendingTransfers();
    drainCompletedDmacHandlers(rdram);
}

void PS2Runtime::requestStop()
{
    m_stopRequested.store(true, std::memory_order_relaxed);
    if (m_eeScheduler)
    {
        m_eeScheduler->requestStop();
    }
}

bool PS2Runtime::isStopRequested() const
{
    return m_stopRequested.load(std::memory_order_relaxed);
}

EeScheduler &PS2Runtime::eeScheduler()
{
    return *m_eeScheduler;
}

const EeScheduler &PS2Runtime::eeScheduler() const
{
    return *m_eeScheduler;
}

void PS2Runtime::postEeEvent(EeEvent event)
{
    m_eeScheduler->postEvent(event);
}

bool PS2Runtime::eeCheckpointDue(uint32_t cycles) noexcept
{
    // Generated code returns to the scheduler when this is true.
    const bool due = m_eeScheduler->checkpointDue(cycles);
    if (due)
    {
        t_guest.unwinding = true;
    }
    return due;
}

[[noreturn]] void PS2Runtime::eeWaitVSyncTicks(uint32_t ticks, uint32_t resumePc)
{
    const uint64_t currentTick = m_eeScheduler->currentVSyncTick();
    const uint64_t waitTicks = std::max<uint64_t>(1u, ticks);
    m_eeScheduler->waitVSync(currentTick + waitTicks - 1u,
                             0,
                             [resumePc](R5900Context &context)
                             {
                                 context.pc = resumePc;
                             });
}

void PS2Runtime::addEeExitHandler(int threadId, uint32_t function, uint32_t argument)
{
    std::lock_guard lock(m_eeKernelStateMutex);
    m_eeExitHandlers[threadId].push_back({function, argument});
}

std::vector<PS2Runtime::EeExitHandlerRegistration> PS2Runtime::takeEeExitHandlers(int threadId)
{
    std::lock_guard lock(m_eeKernelStateMutex);
    auto it = m_eeExitHandlers.find(threadId);
    if (it == m_eeExitHandlers.end())
    {
        return {};
    }
    auto handlers = std::move(it->second);
    m_eeExitHandlers.erase(it);
    return handlers;
}

void PS2Runtime::removeEeExitHandlers(int threadId)
{
    std::lock_guard lock(m_eeKernelStateMutex);
    m_eeExitHandlers.erase(threadId);
}

bool PS2Runtime::findEeSyscallOverride(uint32_t syscallNumber, uint32_t &handler) const
{
    std::lock_guard lock(m_eeKernelStateMutex);
    const auto it = m_eeSyscallOverrides.find(syscallNumber);
    if (it == m_eeSyscallOverrides.end())
    {
        return false;
    }
    handler = it->second;
    return true;
}

void PS2Runtime::setEeSyscallOverride(uint8_t *rdram, uint32_t syscallNumber, uint32_t handler)
{
    constexpr uint32_t kTableBase = 0x80011F80u & 0x1FFFFFFFu;
    constexpr uint32_t kMirrorLimit = 0x00080000u;
    const int64_t offset = static_cast<int64_t>(static_cast<int32_t>(syscallNumber)) * 4;
    const int64_t address = static_cast<int64_t>(kTableBase) + offset;

    std::lock_guard lock(m_eeKernelStateMutex);
    if (handler == 0u)
    {
        m_eeSyscallOverrides.erase(syscallNumber);
    }
    else
    {
        m_eeSyscallOverrides[syscallNumber] = handler;
    }
    if (!rdram || address < 0 || address + 4 > kMirrorLimit)
    {
        return;
    }
    const uint32_t guestAddress = static_cast<uint32_t>(address);
    std::memcpy(rdram + guestAddress, &handler, sizeof(handler));
    if (handler == 0u)
    {
        m_eeSyscallMirrorAddresses.erase(guestAddress);
    }
    else
    {
        m_eeSyscallMirrorAddresses.insert(guestAddress);
    }
}

void PS2Runtime::initializeEeKernelState(uint8_t *rdram)
{
    if (!rdram)
    {
        return;
    }
    constexpr uint32_t kTableGuestBase = 0x80011F80u;
    constexpr uint32_t kTableBase = kTableGuestBase & 0x1FFFFFFFu;
    constexpr uint32_t kMirrorLimit = 0x00080000u;
    constexpr uint32_t kProbeBase = 0x000002F0u;

    std::lock_guard lock(m_eeKernelStateMutex);
    for (const uint32_t address : m_eeSyscallMirrorAddresses)
    {
        const uint32_t zero = 0u;
        std::memcpy(rdram + address, &zero, sizeof(zero));
    }
    m_eeSyscallMirrorAddresses.clear();
    const uint32_t high = kTableGuestBase >> 16;
    const uint32_t low = kTableGuestBase & 0xFFFFu;
    std::memcpy(rdram + kProbeBase, &high, sizeof(high));
    std::memcpy(rdram + kProbeBase + 8u, &low, sizeof(low));
    m_eeSyscallMirrorAddresses.insert(kProbeBase);
    m_eeSyscallMirrorAddresses.insert(kProbeBase + 8u);

    for (const auto &[syscallNumber, handler] : m_eeSyscallOverrides)
    {
        const int64_t offset = static_cast<int64_t>(static_cast<int32_t>(syscallNumber)) * 4;
        const int64_t address = static_cast<int64_t>(kTableBase) + offset;
        if (address < 0 || address + 4 > kMirrorLimit)
        {
            continue;
        }
        const uint32_t guestAddress = static_cast<uint32_t>(address);
        std::memcpy(rdram + guestAddress, &handler, sizeof(handler));
        m_eeSyscallMirrorAddresses.insert(guestAddress);
    }
}

void PS2Runtime::HandleIntegerOverflow(R5900Context *ctx)
{
    raiseCop0Exception(ctx, EXCEPTION_INTEGER_OVERFLOW);
}

void PS2Runtime::run()
{
    m_stopRequested.store(false, std::memory_order_relaxed);
    ps2_stubs::resetSifState();
    resetIop();
    ps2_stubs::resetAudioStubState();
    ps2_stubs::resetMpegStubState();
    initializeEeKernelState(m_memory.getRDRAM());
    m_cpuContext.r[4] = _mm_setzero_si128();
    m_cpuContext.r[5] = _mm_setzero_si128();
    m_cpuContext.r[29] = _mm_set_epi64x(0, static_cast<int64_t>(PS2_RAM_SIZE - 0x10u));
    m_debugPc.store(m_cpuContext.pc, std::memory_order_relaxed);
    m_debugRa.store(static_cast<uint32_t>(_mm_extract_epi32(m_cpuContext.r[31], 0)), std::memory_order_relaxed);
    m_debugSp.store(static_cast<uint32_t>(_mm_extract_epi32(m_cpuContext.r[29], 0)), std::memory_order_relaxed);
    m_debugGp.store(static_cast<uint32_t>(_mm_extract_epi32(m_cpuContext.r[28], 0)), std::memory_order_relaxed);

    RUNTIME_LOG("Starting execution at address 0x" << std::hex << m_cpuContext.pc << std::dec);

    // A blank image to use as a framebuffer
    Image blank = GenImageColor(FB_WIDTH, FB_HEIGHT, BLANK);
    Texture2D frameTex = LoadTextureFromImage(blank);
    UnloadImage(blank);
    SetTextureFilter(frameTex, displaySmooth() ? TEXTURE_FILTER_BILINEAR : TEXTURE_FILTER_POINT);

    std::atomic<bool> gameThreadFinished{false};

    // Guest calls nest on this thread's stack: see LargeStackThread.h.
#if defined(PLATFORM_VITA)
    constexpr size_t kGameThreadStackBytes = 1024u * 1024u; // no address space to spare there
#else
    constexpr size_t kGameThreadStackBytes = 256u * 1024u * 1024u;
#endif
    LargeStackThread gameThread(kGameThreadStackBytes, [&]()
                           {
        ThreadNaming::SetCurrentThreadName("GameThread");
        try
        {
            m_eeScheduler->reset(m_memory.getRDRAM(), m_cpuContext);
            m_eeScheduler->run();
            uint32_t pc = m_debugPc.load(std::memory_order_relaxed);
            RUNTIME_LOG("Game thread returned. PC=0x" << std::hex << pc
                      << " RA=0x" << static_cast<uint32_t>(_mm_extract_epi32(m_cpuContext.r[31], 0)) << std::dec << std::endl);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error during program execution: " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "Error during program execution: unknown exception" << std::endl;
        }
        gameThreadFinished.store(true, std::memory_order_release); });

    uint64_t tick = 0;
    while (!isStopRequested() && !gameThreadFinished.load(std::memory_order_acquire))
    {
        PS2_IF_AGRESSIVE_LOGS({
            tick++;
            if ((tick % 120) == 0)
            {
                uint64_t curDma = m_memory.dmaStartCount();
                uint64_t curGif = m_memory.gifCopyCount();
                uint64_t curGs = m_memory.gsWriteCount();
                uint64_t curVif = m_memory.vifWriteCount();
                const GSRegisters &gs = m_memory.gs();
                const uint32_t dbgPc = m_debugPc.load(std::memory_order_relaxed);
                const uint32_t dbgRa = m_debugRa.load(std::memory_order_relaxed);
                const uint32_t dbgSp = m_debugSp.load(std::memory_order_relaxed);
                const uint32_t dbgGp = m_debugGp.load(std::memory_order_relaxed);
                const auto eeSnapshot = m_eeScheduler->snapshot();

                RUNTIME_LOG("[run:tick] tick=" << tick
                                               << " pc=0x" << std::hex << dbgPc
                                               << " ra=0x" << dbgRa
                                               << " sp=0x" << dbgSp
                                               << " gp=0x" << dbgGp
                                               << " dispfb1=0x" << gs.dispfb1
                                               << " display1=0x" << gs.display1
                                               << std::dec
                                               << " activeThreads=" << eeSnapshot.threads.size()
                                               << " dma=" << curDma
                                               << " gif=" << curGif
                                               << " gsw=" << curGs
                                               << " vif=" << curVif
                                               << std::endl);

            }
        });
        uint32_t presentWidth = FB_WIDTH;
        uint32_t presentHeight = DEFAULT_DISPLAY_HEIGHT;
        UploadFrame(frameTex, this, presentWidth, presentHeight);

        {
            // PS2X_GIF_TRACE=<secs>[:count]: see traceWildGifPrimitives.
            static const uint64_t gifTraceTick = envEmulatedTicks("PS2X_GIF_TRACE");
            static bool gifTraceArmed = false;
            if (gifTraceTick != 0u && !gifTraceArmed && m_eeScheduler && m_eeScheduler->currentVSyncTick() >= gifTraceTick)
            {
                gifTraceArmed = true;
                const char *colon = std::strchr(std::getenv("PS2X_GIF_TRACE"), ':');
                g_ps2xGifTraceRemaining = colon ? std::atoi(colon + 1) : 16;
            }
            // PS2X_VIF1_TRACE=<secs>[:count]: the VIFcodes VIF1 is sent from
            // that time on (see traceVif1Code), 256 of them unless told.
            static const uint64_t vif1TraceTick = envEmulatedTicks("PS2X_VIF1_TRACE");
            static bool vif1TraceArmed = false;
            if (vif1TraceTick != 0u && !vif1TraceArmed && m_eeScheduler && m_eeScheduler->currentVSyncTick() >= vif1TraceTick)
            {
                vif1TraceArmed = true;
                const char *colon = std::strchr(std::getenv("PS2X_VIF1_TRACE"), ':');
                g_ps2xVif1TraceRemaining = colon ? std::atoi(colon + 1) : 256;
            }
            static const uint64_t exitTick = envEmulatedTicks("PS2X_EXIT_AT");
            static const uint64_t perfTicks = envEmulatedTicks("PS2X_PERF_LOG");
            static const auto wallStart = std::chrono::steady_clock::now();
            static uint64_t nextPerfTick = perfTicks;
            static double lastWall = 0.0;
            static uint64_t lastTick = 0u;
            const uint64_t emulatedTick = m_eeScheduler ? m_eeScheduler->currentVSyncTick() : 0u;
            if (perfTicks != 0u && emulatedTick >= nextPerfTick)
            {
                const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wallStart).count();
                const double emulated = static_cast<double>(emulatedTick) / kTestVBlanksPerSecond;
                const double recent = (wall > lastWall)
                                          ? (static_cast<double>(emulatedTick - lastTick) / kTestVBlanksPerSecond) / (wall - lastWall)
                                          : 0.0;
                std::fprintf(stderr, "[perf] emu=%.1fs wall=%.1fs speed=%.2fx (recent %.2fx)\n",
                             emulated, wall, wall > 0.0 ? emulated / wall : 0.0, recent);
                lastWall = wall;
                lastTick = emulatedTick;
                while (nextPerfTick <= emulatedTick)
                    nextPerfTick += perfTicks;
            }
            // A GS that may lag: the run ends with the picture of that tick
            // shown (and saved, if frames are), not when the EE gets there.
            uint64_t shownTick = emulatedTick;
            if (GsWorker::Display shown; GsWorker::shownDisplay(shown))
                shownTick = shown.tick;
            if (exitTick != 0u && shownTick >= exitTick)
            {
                std::fprintf(stderr, "[run] PS2X_EXIT_AT reached at emu=%.2fs\n",
                             static_cast<double>(shownTick) / kTestVBlanksPerSecond);
                ps2x::audio::stopOutput(); // PS2X_AUDIO_DUMP is closed with its length in it
                std::fflush(nullptr);
                std::_Exit(0);
            }
        }

#if !defined(PLATFORM_VITA)
        if (IsKeyPressed(KEY_F11))
            ps2x::toggleDisplayMode();
#endif
        ps2xInputFrame();
        ps2x::settingsMenuFrame();
        BeginDrawing();
        ClearBackground(BLACK);
        const float srcWidth = static_cast<float>(std::max<uint32_t>(1u, presentWidth));
        const float srcHeight = static_cast<float>(std::max<uint32_t>(1u, presentHeight));
        const float screenWidth = static_cast<float>(GetScreenWidth());
        const float screenHeight = static_cast<float>(GetScreenHeight());
        // As large as fits, in the shape asked for: the game's pixels are
        // not square on a television, and a widescreen picture is drawn
        // squeezed into the same 640.
        const float asked = g_presentationAspect.load(std::memory_order_relaxed);
        const float shape = asked > 0.0f ? asked : hostDisplay().aspect > 0.0f ? hostDisplay().aspect : srcWidth / srcHeight;
        const float dstHeight = std::min(screenHeight, screenWidth / shape);
        const float dstWidth = dstHeight * shape;
        const Rectangle srcRect{0.0f, 0.0f, srcWidth, srcHeight};
        const Rectangle dstRect{
            (screenWidth - dstWidth) * 0.5f,
            (screenHeight - dstHeight) * 0.5f,
            dstWidth,
            dstHeight};
        // (The filter may have been changed since the textures were made.)
        {
            static int filterInForce = -1;
            const int filter = displaySmooth() ? TEXTURE_FILTER_BILINEAR : TEXTURE_FILTER_POINT;
            if (filter != filterInForce)
            {
                filterInForce = filter;
                SetTextureFilter(frameTex, filter);
                if (g_cardTexture.id != 0u)
                    SetTextureFilter(g_cardTexture, filter);
            }
        }
        DrawTexturePro(g_cardPicture ? g_cardTexture : frameTex, srcRect, dstRect, Vector2{0.0f, 0.0f}, 0.0f, WHITE);
        if (m_debugUiInitialized && m_debugUiDrawCallback)
        {
            m_debugUiDrawCallback(*this, m_debugUiUserData);
        }
        EndDrawing();

        if (WindowShouldClose())
        {
            RUNTIME_LOG("[run] window close requested, breaking out of loop");
            requestStop();
            break;
        }
    }

    requestStop();
    if (gameThread.joinable())
    {
        gameThread.join();
    }

    if (m_debugUiInitialized && m_debugUiShutdownCallback)
    {
        m_debugUiShutdownCallback(*this, m_debugUiUserData);
        m_debugUiInitialized = false;
    }
    UnloadTexture(frameTex);
    CloseWindow();

    RUNTIME_LOG("[run] exiting loop");
}
