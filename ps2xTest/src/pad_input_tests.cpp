#include "MiniTest.h"
#include "ps2_stubs.h"
#include "ps2_syscalls.h"
#include "Stubs/Pad.h"

#include <vector>
#include <cstdint>
#include <string>


namespace
{
    constexpr uint32_t kPadDataAddr = 0x1000;

    constexpr uint16_t kPadBtnSelect = 1u << 0;
    constexpr uint16_t kPadBtnL3 = 1u << 1;
    constexpr uint16_t kPadBtnR3 = 1u << 2;
    constexpr uint16_t kPadBtnStart = 1u << 3;
    constexpr uint16_t kPadBtnUp = 1u << 4;
    constexpr uint16_t kPadBtnRight = 1u << 5;
    constexpr uint16_t kPadBtnDown = 1u << 6;
    constexpr uint16_t kPadBtnLeft = 1u << 7;
    constexpr uint16_t kPadBtnL2 = 1u << 8;
    constexpr uint16_t kPadBtnR2 = 1u << 9;
    constexpr uint16_t kPadBtnL1 = 1u << 10;
    constexpr uint16_t kPadBtnR1 = 1u << 11;
    constexpr uint16_t kPadBtnTriangle = 1u << 12;
    constexpr uint16_t kPadBtnCircle = 1u << 13;
    constexpr uint16_t kPadBtnCross = 1u << 14;
    constexpr uint16_t kPadBtnSquare = 1u << 15;

    void setRegU32(R5900Context &ctx, int reg, uint32_t value)
    {
        ctx.r[reg] = _mm_set_epi64x(0, static_cast<int64_t>(value));
    }

    void openPadPort(R5900Context &ctx, std::vector<uint8_t> &rdram, uint32_t port = 0, uint32_t slot = 0)
    {
        setRegU32(ctx, 4, port);
        setRegU32(ctx, 5, slot);
        setRegU32(ctx, 6, kPadDataAddr + 0x200u);
        ps2_stubs::scePadPortOpen(rdram.data(), &ctx, nullptr);
    }

    void closePadPort(R5900Context &ctx, std::vector<uint8_t> &rdram, uint32_t port = 0, uint32_t slot = 0)
    {
        setRegU32(ctx, 4, port);
        setRegU32(ctx, 5, slot);
        ps2_stubs::scePadPortClose(rdram.data(), &ctx, nullptr);
    }

    void runPadRead(R5900Context &ctx, std::vector<uint8_t> &rdram)
    {
        setRegU32(ctx, 4, 0u);
        setRegU32(ctx, 5, 0u);
        setRegU32(ctx, 6, kPadDataAddr); // a2
        ps2_stubs::scePadRead(rdram.data(), &ctx, nullptr);
    }

    uint16_t readButtons(const std::vector<uint8_t> &rdram)
    {
        const uint8_t *data = rdram.data() + kPadDataAddr;
        return static_cast<uint16_t>(data[2] | (data[3] << 8));
    }
}

void register_pad_input_tests()
{
    MiniTest::Case("PadInput", [](TestCase &tc)
                   {
        tc.Run("scePadRead uses override state", [](TestCase &t)
               {
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0);
            R5900Context ctx;

            ps2_stubs::scePadInit(rdram.data(), &ctx, nullptr);
            openPadPort(ctx, rdram);

            const uint16_t buttons = static_cast<uint16_t>(0xFFFFu & ~kPadBtnCross & ~kPadBtnStart);
            ps2_stubs::setPadOverrideState(buttons, 0x00, 0xFF, 0x10, 0xEE);

            runPadRead(ctx, rdram);

            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadRead should return 1");
            t.Equals(readButtons(rdram), buttons, "button bitmask should match override state");
            const uint8_t *data = rdram.data() + kPadDataAddr;
            t.Equals(data[4], static_cast<uint8_t>(0x10), "rx should match override");
            t.Equals(data[5], static_cast<uint8_t>(0xEE), "ry should match override");
            t.Equals(data[6], static_cast<uint8_t>(0x00), "lx should match override");
            t.Equals(data[7], static_cast<uint8_t>(0xFF), "ly should match override");

            ps2_stubs::clearPadOverrideState();
            closePadPort(ctx, rdram);
        });

        tc.Run("scePadRead button bits are active-low", [](TestCase &t)
               {
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0);
            R5900Context ctx;

            ps2_stubs::scePadInit(rdram.data(), &ctx, nullptr);
            openPadPort(ctx, rdram);

            struct ButtonCase
            {
                uint16_t mask;
                const char *name;
            };

            const ButtonCase cases[] = {
                {kPadBtnSelect, "select"},
                {kPadBtnL3, "l3"},
                {kPadBtnR3, "r3"},
                {kPadBtnStart, "start"},
                {kPadBtnUp, "up"},
                {kPadBtnRight, "right"},
                {kPadBtnDown, "down"},
                {kPadBtnLeft, "left"},
                {kPadBtnL2, "l2"},
                {kPadBtnR2, "r2"},
                {kPadBtnL1, "l1"},
                {kPadBtnR1, "r1"},
                {kPadBtnTriangle, "triangle"},
                {kPadBtnCircle, "circle"},
                {kPadBtnCross, "cross"},
                {kPadBtnSquare, "square"}};

            for (const auto &entry : cases)
            {
                const uint16_t buttons = static_cast<uint16_t>(0xFFFFu & ~entry.mask);
                ps2_stubs::setPadOverrideState(buttons, 0x80, 0x80, 0x80, 0x80);
                runPadRead(ctx, rdram);

                t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadRead should succeed for opened ports");
                const uint16_t mask = readButtons(rdram);
                t.IsTrue((mask & entry.mask) == 0, std::string("button should be active-low: ").append(entry.name));
            }

            ps2_stubs::clearPadOverrideState();
            closePadPort(ctx, rdram);
        });

        tc.Run("scePadGetButtonMask returns all buttons", [](TestCase &t)
               {
            R5900Context ctx;
            ps2_stubs::scePadGetButtonMask(nullptr, &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(0xFFFF), "button mask should be 0xFFFF");
        });

        tc.Run("basic pad init/port/state functions return expected values", [](TestCase &t)
               {
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0);
            R5900Context ctx;

            ps2_stubs::scePadInit(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadInit should succeed");

            ps2_stubs::scePadInit2(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadInit2 should succeed");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadGetState(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(0), "closed port should report DISCONNECTED");

            openPadPort(ctx, rdram);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadPortOpen should succeed");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadGetState(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(6), "scePadGetState should return STABLE");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadGetReqState(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(0), "scePadGetReqState should return completed");

            ps2_stubs::scePadGetPortMax(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(2), "scePadGetPortMax should be 2");

            setRegU32(ctx, 4, 0u);
            ps2_stubs::scePadGetSlotMax(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadGetSlotMax should be 1");

            ps2_stubs::scePadGetModVersion(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(0x0200), "scePadGetModVersion should be 0x0200");

            closePadPort(ctx, rdram);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadPortClose should succeed");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadGetState(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(0), "closed port should return DISCONNECTED after close");
        });

        tc.Run("pad command state reports EXECCMD once before returning STABLE", [](TestCase &t)
               {
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0);
            R5900Context ctx;

            ps2_stubs::scePadInit(rdram.data(), &ctx, nullptr);
            openPadPort(ctx, rdram);

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            setRegU32(ctx, 6, 1u);
            setRegU32(ctx, 7, 3u);
            ps2_stubs::scePadSetMainMode(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadSetMainMode should succeed");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadGetState(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(5), "first state after mode command should be EXECCMD");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadGetState(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(6), "second state after mode command should return STABLE");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadEnterPressMode(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadEnterPressMode should succeed");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadGetState(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(5), "first state after press-mode command should be EXECCMD");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadGetState(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(6), "second state after press-mode command should return STABLE");

            closePadPort(ctx, rdram);
        });

        tc.Run("pad info and mode helpers return consistent values", [](TestCase &t)
               {
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0);
            R5900Context ctx;

            ps2_stubs::scePadInit(rdram.data(), &ctx, nullptr);
            openPadPort(ctx, rdram);

            setRegU32(ctx, 6, static_cast<uint32_t>(-1));
            ps2_stubs::scePadInfoAct(rdram.data(), &ctx, nullptr);
            t.IsTrue(static_cast<uint32_t>(getRegU32(&ctx, 2)) >= 1u, "scePadInfoAct should report at least one actuator descriptor");

            ps2_stubs::scePadInfoComb(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(0), "scePadInfoComb should return 0");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            setRegU32(ctx, 6, 1);
            setRegU32(ctx, 7, 0);
            ps2_stubs::scePadInfoMode(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(4), "scePadInfoMode CURID should return digital at open");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            setRegU32(ctx, 6, 4);
            setRegU32(ctx, 7, static_cast<uint32_t>(-1));
            ps2_stubs::scePadInfoMode(rdram.data(), &ctx, nullptr);
            t.IsTrue(static_cast<uint32_t>(getRegU32(&ctx, 2)) >= 1u, "scePadInfoMode table count should be non-zero");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadInfoPressMode(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadInfoPressMode should report pressure support");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            setRegU32(ctx, 6, 0u);
            setRegU32(ctx, 7, 3u);
            ps2_stubs::scePadSetMainMode(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadSetMainMode should accept digital mode");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            setRegU32(ctx, 6, 1u);
            setRegU32(ctx, 7, 0u);
            ps2_stubs::scePadInfoMode(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(4), "CURID should switch to digital mode");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            setRegU32(ctx, 6, 1u);
            setRegU32(ctx, 7, 3u);
            ps2_stubs::scePadSetMainMode(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadSetMainMode should accept analog mode");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            setRegU32(ctx, 6, 4);
            setRegU32(ctx, 7, 0u);
            ps2_stubs::scePadInfoMode(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(4), "the mode table of a DualShock 2 starts with digital");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            setRegU32(ctx, 6, 4);
            setRegU32(ctx, 7, 1u);
            ps2_stubs::scePadInfoMode(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(7), "its second entry is DualShock");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            setRegU32(ctx, 6, 3);
            setRegU32(ctx, 7, 0u);
            ps2_stubs::scePadInfoMode(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "in analog mode the current entry is the second");

            closePadPort(ctx, rdram);
        });

        tc.Run("pads open in digital mode and switch to analog on scePadSetMainMode", [](TestCase &t)
               {
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0);
            R5900Context ctx;

            ps2_stubs::scePadInit(rdram.data(), &ctx, nullptr);
            openPadPort(ctx, rdram);

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadGetState(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(6), "freshly opened port should report STABLE");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            setRegU32(ctx, 6, 1);
            setRegU32(ctx, 7, 0);
            ps2_stubs::scePadInfoMode(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(4), "scePadInfoMode CURID should return digital at open");

            runPadRead(ctx, rdram);
            const uint8_t *data = rdram.data() + kPadDataAddr;
            t.Equals(data[1], static_cast<uint8_t>(0x41), "mode byte should be 0x41 (digital) at open");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            setRegU32(ctx, 6, 1);
            setRegU32(ctx, 7, 3);
            ps2_stubs::scePadSetMainMode(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadSetMainMode should succeed switching to analog");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            setRegU32(ctx, 6, 1);
            setRegU32(ctx, 7, 0);
            ps2_stubs::scePadInfoMode(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(7), "scePadInfoMode CURID should return analog after SetMainMode");

            // scePadSetMainMode queues a one-shot EXECCMD transient state; pump scePadGetState
            // once so the port settles back to STABLE before reading, mirroring the existing
            // "pad command state reports EXECCMD once before returning STABLE" test.
            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadGetState(rdram.data(), &ctx, nullptr);

            runPadRead(ctx, rdram);
            t.Equals(data[1], static_cast<uint8_t>(0x73), "mode byte should be 0x73 (analog) after SetMainMode");

            closePadPort(ctx, rdram);
        });

        tc.Run("pad setters return success", [](TestCase &t)
               {
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0);
            R5900Context ctx;

            ps2_stubs::scePadInit(rdram.data(), &ctx, nullptr);
            openPadPort(ctx, rdram);

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadSetActAlign(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadSetActAlign should succeed");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadSetActDirect(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadSetActDirect should succeed");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            setRegU32(ctx, 6, 0xFFFFu);
            ps2_stubs::scePadSetButtonInfo(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadSetButtonInfo should succeed");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            setRegU32(ctx, 6, 1u);
            setRegU32(ctx, 7, 3u);
            ps2_stubs::scePadSetMainMode(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadSetMainMode should succeed");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadSetReqState(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadSetReqState should succeed");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadSetVrefParam(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadSetVrefParam should succeed");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadSetWarningLevel(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(0), "scePadSetWarningLevel should return 0");

            ps2_stubs::scePadEnd(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadEnd should succeed");

            openPadPort(ctx, rdram);
            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadEnterPressMode(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadEnterPressMode should succeed");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadExitPressMode(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadExitPressMode should succeed");

            closePadPort(ctx, rdram);
        });

        tc.Run("scePadRead fills pressure bytes and honors button info mask", [](TestCase &t)
               {
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0);
            R5900Context ctx;

            ps2_stubs::scePadInit(rdram.data(), &ctx, nullptr);
            openPadPort(ctx, rdram);

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            setRegU32(ctx, 6, 0xFFFFu);
            ps2_stubs::scePadSetButtonInfo(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadSetButtonInfo should accept all buttons");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadEnterPressMode(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadEnterPressMode should enable pressure data");

            const uint16_t pressedButtons = static_cast<uint16_t>(0xFFFFu &
                                                                   ~kPadBtnLeft &
                                                                   ~kPadBtnUp &
                                                                   ~kPadBtnTriangle &
                                                                   ~kPadBtnCross &
                                                                   ~kPadBtnL1 &
                                                                   ~kPadBtnR2);
            ps2_stubs::setPadOverrideState(pressedButtons, 0x80, 0x80, 0x80, 0x80);
            runPadRead(ctx, rdram);

            const uint8_t *data = rdram.data() + kPadDataAddr;
            t.Equals(data[8], static_cast<uint8_t>(0x00), "right pressure should be clear when not pressed");
            t.Equals(data[9], static_cast<uint8_t>(0xFF), "left pressure should be populated when pressed");
            t.Equals(data[10], static_cast<uint8_t>(0xFF), "up pressure should be populated when pressed");
            t.Equals(data[11], static_cast<uint8_t>(0x00), "down pressure should be clear when not pressed");
            t.Equals(data[12], static_cast<uint8_t>(0xFF), "triangle pressure should be populated when pressed");
            t.Equals(data[13], static_cast<uint8_t>(0x00), "circle pressure should be clear when not pressed");
            t.Equals(data[14], static_cast<uint8_t>(0xFF), "cross pressure should be populated when pressed");
            t.Equals(data[15], static_cast<uint8_t>(0x00), "square pressure should be clear when not pressed");
            t.Equals(data[16], static_cast<uint8_t>(0xFF), "L1 pressure should be populated when pressed");
            t.Equals(data[17], static_cast<uint8_t>(0x00), "L2 pressure should be clear when not pressed");
            t.Equals(data[18], static_cast<uint8_t>(0x00), "R1 pressure should be clear when not pressed");
            t.Equals(data[19], static_cast<uint8_t>(0xFF), "R2 pressure should be populated when pressed");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            setRegU32(ctx, 6, static_cast<uint32_t>(kPadBtnL1 | kPadBtnR2));
            ps2_stubs::scePadSetButtonInfo(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadSetButtonInfo should narrow the enabled pressure mask");

            runPadRead(ctx, rdram);

            t.Equals(data[9], static_cast<uint8_t>(0x00), "masked-out direction pressure should clear");
            t.Equals(data[10], static_cast<uint8_t>(0x00), "masked-out direction pressure should clear");
            t.Equals(data[12], static_cast<uint8_t>(0x00), "masked-out face-button pressure should clear");
            t.Equals(data[14], static_cast<uint8_t>(0x00), "masked-out face-button pressure should clear");
            t.Equals(data[16], static_cast<uint8_t>(0xFF), "enabled L1 pressure should remain populated");
            t.Equals(data[19], static_cast<uint8_t>(0xFF), "enabled R2 pressure should remain populated");

            ps2_stubs::clearPadOverrideState();
            closePadPort(ctx, rdram);
        });

        tc.Run("pad string helpers map state codes", [](TestCase &t)
               {
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0);
            R5900Context ctx;

            setRegU32(ctx, 4, 1);
            setRegU32(ctx, 5, kPadDataAddr);
            ps2_stubs::scePadStateIntToStr(rdram.data(), &ctx, nullptr);
            t.IsTrue(std::string(reinterpret_cast<const char *>(rdram.data() + kPadDataAddr)).find("FINDPAD") != std::string::npos,
                     "state 1 should map to FINDPAD");

            setRegU32(ctx, 4, 0);
            setRegU32(ctx, 5, kPadDataAddr + 64);
            ps2_stubs::scePadStateIntToStr(rdram.data(), &ctx, nullptr);
            t.IsTrue(std::string(reinterpret_cast<const char *>(rdram.data() + kPadDataAddr + 64)).find("DISCONNECTED") != std::string::npos,
                     "state 0 should map to DISCONNECTED");

            setRegU32(ctx, 4, 1);
            setRegU32(ctx, 5, kPadDataAddr + 128);
            ps2_stubs::scePadReqIntToStr(rdram.data(), &ctx, nullptr);
            t.IsTrue(std::string(reinterpret_cast<const char *>(rdram.data() + kPadDataAddr + 128)).find("BUSY") != std::string::npos,
                     "req state 1 should map to BUSY");
        });
        tc.Run("libpad2 socket reads a DualShock 2 in wire order", [](TestCase &t)
               {
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0);
            R5900Context ctx;
            constexpr uint32_t kParamAddr = 0x2000u;
            constexpr uint32_t kWorkAddr = 0x3000u; // 64-byte aligned
            constexpr uint32_t kStateAddr = 0x4000u;

            ps2_stubs::sceDbcInit(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "sceDbcInit should succeed");
            setRegU32(ctx, 4, 0u);
            ps2_stubs::scePad2Init(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePad2Init should succeed");

            setRegU32(ctx, 4, 0u);
            ps2_stubs::scePad2GetState(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(0), "a socket that was never created has no link");

            // scePad2SocketParam { option, port, slot, number, name[16] }
            const uint32_t param[4] = {2u, 0u, 0u, 0u};
            std::memcpy(rdram.data() + kParamAddr, param, sizeof(param));

            setRegU32(ctx, 4, kParamAddr);
            setRegU32(ctx, 5, kWorkAddr + 4u);
            ps2_stubs::scePad2CreateSocket(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<int32_t>(getRegU32(&ctx, 2)), static_cast<int32_t>(-1), "a work area that is not 64-byte aligned is refused");

            setRegU32(ctx, 4, kParamAddr);
            setRegU32(ctx, 5, kWorkAddr);
            ps2_stubs::scePad2CreateSocket(rdram.data(), &ctx, nullptr);
            const uint32_t socket = getRegU32(&ctx, 2);
            t.IsTrue(static_cast<int32_t>(socket) >= 0, "scePad2CreateSocket should return a socket");

            setRegU32(ctx, 4, socket);
            ps2_stubs::scePad2GetState(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "an open socket is stable");

            const uint16_t pressed = static_cast<uint16_t>(0xFFFFu & ~kPadBtnCross & ~kPadBtnStart & ~kPadBtnR1 & ~kPadBtnL2);
            ps2_stubs::setPadOverrideState(pressed, 0x11, 0x22, 0x33, 0x44);
            std::memset(rdram.data() + kStateAddr, 0xAA, 32);
            setRegU32(ctx, 4, socket);
            setRegU32(ctx, 5, kStateAddr);
            ps2_stubs::scePad2Read(rdram.data(), &ctx, nullptr);
            ps2_stubs::clearPadOverrideState();
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(18), "scePad2Read should return the 18 bytes of a DualShock 2");

            const uint8_t *data = rdram.data() + kStateAddr;
            t.Equals(static_cast<uint32_t>(data[0] | (data[1] << 8)), static_cast<uint32_t>(pressed), "switches come first, active low");
            t.Equals(data[2], static_cast<uint8_t>(0x33), "right stick x");
            t.Equals(data[3], static_cast<uint8_t>(0x44), "right stick y");
            t.Equals(data[4], static_cast<uint8_t>(0x11), "left stick x");
            t.Equals(data[5], static_cast<uint8_t>(0x22), "left stick y");
            t.Equals(data[12], static_cast<uint8_t>(0xFF), "cross pressure");
            t.Equals(data[13], static_cast<uint8_t>(0x00), "square pressure");
            t.Equals(data[14], static_cast<uint8_t>(0x00), "L1 pressure");
            t.Equals(data[15], static_cast<uint8_t>(0xFF), "R1 pressure");
            t.Equals(data[16], static_cast<uint8_t>(0xFF), "L2 pressure");
            t.Equals(data[17], static_cast<uint8_t>(0x00), "R2 pressure");
            t.Equals(data[18], static_cast<uint8_t>(0xAA), "nothing is written past the 18 bytes");

            setRegU32(ctx, 4, socket);
            ps2_stubs::scePad2DeleteSocket(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePad2DeleteSocket should succeed");
            setRegU32(ctx, 4, socket);
            setRegU32(ctx, 5, kStateAddr);
            ps2_stubs::scePad2Read(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<int32_t>(getRegU32(&ctx, 2)), static_cast<int32_t>(-1), "a deleted socket cannot be read");
        });

        tc.Run("scePadGetFrameCount increments", [](TestCase &t)
               {
            R5900Context ctx;
            ps2_stubs::scePadGetFrameCount(nullptr, &ctx, nullptr);
            const uint32_t first = getRegU32(&ctx, 2);
            ps2_stubs::scePadGetFrameCount(nullptr, &ctx, nullptr);
            const uint32_t second = getRegU32(&ctx, 2);
            t.Equals(second, first + 1, "frame count should increment");
        });

        tc.Run("scePadStateIntToStr and scePadReqIntToStr write strings", [](TestCase &t)
               {
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0);
            R5900Context ctx;

            setRegU32(ctx, 4, 6);
            setRegU32(ctx, 5, kPadDataAddr);
            ps2_stubs::scePadStateIntToStr(rdram.data(), &ctx, nullptr);
            const char *stateStr = reinterpret_cast<const char *>(rdram.data() + kPadDataAddr);
            t.IsTrue(std::string(stateStr).find("STABLE") != std::string::npos, "state string should include STABLE");

            setRegU32(ctx, 4, 0);
            setRegU32(ctx, 5, kPadDataAddr + 64);
            ps2_stubs::scePadReqIntToStr(rdram.data(), &ctx, nullptr);
            const char *reqStr = reinterpret_cast<const char *>(rdram.data() + kPadDataAddr + 64);
            t.IsTrue(std::string(reqStr).find("COMPLETE") != std::string::npos, "req string should include COMPLETE");
        });
    });
}
