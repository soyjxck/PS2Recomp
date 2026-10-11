#include "MiniTest.h"
#include "ps2x/iop/ps2_path.h"
#include "../../ps2xIOP/tests/iop_compat_test_support.h"
#include "../../ps2xIOP/src/emulator/core/iop_memory.h"

#include <cstdlib>
#include <vector>

namespace
{
    using ps2x::iop::detail::IopMemory;
    using ps2x::iop::detail::IopSpu2;

    // An IOP around the SPU2 that hands AutoDMA the blocks a test queued.
    struct SpuTestHost final : ps2x::iop::detail::IopSpu2Host
    {
        bool fetchAutoDma(int, uint16_t *block) override
        {
            if (blocksLeft == 0u)
                return false;
            --blocksLeft;
            ++blocksFetched;
            for (uint32_t index = 0; index < 256u; ++index)
            {
                block[index] = static_cast<uint16_t>(left);
                block[256u + index] = static_cast<uint16_t>(right);
            }
            return true;
        }
        void raiseSpuInterrupt() override { ++interrupts; }

        uint32_t blocksLeft = 0;
        uint32_t blocksFetched = 0;
        uint32_t interrupts = 0;
        int16_t left = 0;
        int16_t right = 0;
    };

    // Registers, as offsets from 0x1F900000.
    constexpr uint32_t kCore1 = 0x400u;
    constexpr uint32_t kVoiceVolL = 0x000u, kVoiceVolR = 0x002u, kVoicePitch = 0x004u, kVoiceAdsr1 = 0x006u,
                       kVoiceAdsr2 = 0x008u, kVoiceEnvX = 0x00Au;
    constexpr uint32_t kVmixL = 0x188u, kVmixR = 0x190u, kMmix = 0x198u, kAttr = 0x19Au, kIrqaH = 0x19Cu, kIrqaL = 0x19Eu,
                       kKeyOn = 0x1A0u, kKeyOff = 0x1A4u, kTsaH = 0x1A8u, kTsaL = 0x1AAu, kAdmas = 0x1B0u,
                       kVoiceSsaH = 0x1C0u, kVoiceSsaL = 0x1C2u, kVoiceNaxH = 0x1C8u, kVoiceNaxL = 0x1CAu,
                       kEndxL = 0x340u, kStatx = 0x344u;
    constexpr uint32_t kMvolL = 0x760u, kMvolR = 0x762u, kBvolL = 0x76Cu, kBvolR = 0x76Eu, kCore1Volumes = 0x28u,
                       kAvolL = 0x768u, kAvolR = 0x76Au, kIrqInfo = 0x7C2u;

    // Both cores at full volume, the first heard through the second as on
    // the console.
    void openSpuOutput(IopSpu2 &spu)
    {
        for (const uint32_t core : {0u, kCore1Volumes})
        {
            spu.write(kMvolL + core, 0x3FFFu);
            spu.write(kMvolR + core, 0x3FFFu);
        }
        spu.write(kAvolL + kCore1Volumes, 0x7FFFu);
        spu.write(kAvolR + kCore1Volumes, 0x7FFFu);
        spu.write(kMmix, 0x0FF0u);
        spu.write(kMmix + kCore1, 0x0FFCu);
    }
}

void register_ps2_iop_tests()
{
    MiniTest::Case("PS2IopSubsystem", [](TestCase &tc)
    {
        tc.Run("PS2 path parsing is shared and normalizes ISO/module names", [](TestCase &t)
        {
            const ps2x::iop::ParsedPs2Path cd = ps2x::iop::parsePs2Path("CDROM0:\\MODULES\\LIBSD.IRX;1");
            t.Equals(cd.device, ps2x::iop::Ps2PathDevice::Cdrom,
                     "device names should be case-insensitive");
            t.Equals(cd.path, std::string("MODULES/LIBSD.IRX"),
                     "separators and ISO version suffixes should normalize once");
            t.Equals(ps2x::iop::ps2PathLeafKey(cd), std::string("libsd"),
                     "module lookup should use a normalized IRX leaf key");

            const ps2x::iop::ParsedPs2Path rom = ps2x::iop::parsePs2Path("rom0:ROMVER");
            t.Equals(rom.device, ps2x::iop::Ps2PathDevice::Rom0,
                     "ROM0 should remain a distinct virtual device");
            t.IsFalse(static_cast<bool>(ps2x::iop::parsePs2Path("unknown0:file.irx")),
                      "unsupported devices must not fall through to cdrom0");
        });

        tc.Run("HLE services activate only after a recognized module load", [](TestCase &t)
        {
            iop_test::Host host;
            ps2x::iop::IopSubsystem subsystem(host);
            t.IsFalse(subsystem.canBindRpc(0x80000701u),
                      "LIBSD RPC must not exist before LIBSD is loaded");

            const ps2x::iop::ModuleLoadResult unknown = subsystem.loadModule("rom0:NOT_A_REAL_MODULE");
            t.IsTrue(unknown.handled, "the module manager should return a real load result");
            t.IsTrue(unknown.moduleId < 0, "unknown ROM modules must fail instead of receiving fake IDs");

            const ps2x::iop::ModuleLoadResult loaded = subsystem.loadModule("rom0:LIBSD");
            t.IsTrue(loaded.moduleId > 0, "a registered no-BIOS HLE module should load");
            t.IsTrue(subsystem.canBindRpc(0x80000701u),
                     "loading LIBSD should activate its HLE RPC endpoint");

            ps2x::iop::RpcRequest request{};
            request.sid = 0x80000701u;
            request.function = 3u;
            t.IsTrue(subsystem.handleRpc(request).handled,
                     "the activated LIBSD service should handle its RPC");
            t.Equals(host.audioCalls, 1u, "the RPC should reach the HLE audio contract");

            int32_t stopResult = -1;
            t.IsTrue(subsystem.stopModule(loaded.moduleId, &stopResult),
                     "an HLE module should have a real stoppable lifecycle");
            t.Equals(stopResult, 0, "stopping an HLE module should report success");
            t.IsFalse(subsystem.canBindRpc(0x80000701u),
                      "stopping LIBSD should deactivate its RPC endpoint");
        });

        tc.Run("unknown SID remains unhandled without a loaded IRX", [](TestCase &t)
        {
            iop_test::Host host;
            ps2x::iop::IopSubsystem subsystem(host);

            ps2x::iop::RpcRequest request{};
            request.sid = 0xDEADC0DEu;
            request.function = 0x99u;
            const ps2x::iop::RpcResult result = subsystem.handleRpc(request);
            t.IsFalse(result.handled, "an unknown SID should not be claimed by the IOP subsystem");
            t.Equals(result.resultAddress, 0u, "an unknown SID should not return a guest result address");
            t.IsFalse(result.signalNowaitCompletion, "an unknown SID should not signal nowait completion");
            t.Equals(result.callbackPolicy, ps2x::iop::CallbackPolicy::RuntimeDefault,
                     "an unknown SID should preserve runtime callback handling");

        });

    });

    MiniTest::Case("PS2IopSpu2", [](TestCase &tc)
    {
        tc.Run("a voice plays the block it is keyed on, loops it and is released", [](TestCase &t)
        {
            SpuTestHost host;
            IopSpu2 spu(host);
            openSpuOutput(spu);

            // One block that is its own loop: no shift, no filter, every
            // nibble 1, so every sample is 0x1000.
            constexpr uint32_t kBlock = 0x2800u;
            spu.ram()[kBlock] = 0x0700u;
            for (uint32_t word = 1; word < 8u; ++word)
                spu.ram()[kBlock + word] = 0x1111u;

            spu.write(kVoiceVolL, 0x3FFFu);
            spu.write(kVoiceVolR, 0x1FFFu);
            spu.write(kVoicePitch, 0x1000u);
            spu.write(kVoiceAdsr1, 0x00FFu);
            spu.write(kVoiceAdsr2, 0x1FC0u);
            spu.write(kVoiceSsaH, 0u);
            spu.write(kVoiceSsaL, static_cast<uint16_t>(kBlock));
            spu.write(kVmixL, 1u);
            spu.write(kVmixR, 1u);

            int16_t left = 0, right = 0;
            spu.tick(left, right);
            t.Equals(static_cast<int>(left), 0, "nothing sounds before a key on");

            spu.write(kKeyOn, 1u);
            for (int sample = 0; sample < 64; ++sample)
                spu.tick(left, right);
            t.IsTrue(left > 4060 && left <= 4096, "the left side should play the block at the voice's volume");
            t.IsTrue(right > 2020 && right <= 2048, "the right side should play it at half that");
            t.IsTrue(spu.read(kVoiceEnvX) > 0x7F00u, "ENVX should read the envelope after its attack");
            const uint32_t next = (static_cast<uint32_t>(spu.read(kVoiceNaxH)) << 16u) | spu.read(kVoiceNaxL);
            t.IsTrue(next >= kBlock && next <= kBlock + 8u, "NAX should stay in the block that loops");
            t.Equals(static_cast<uint32_t>(spu.read(kEndxL) & 1u), 1u, "ENDX should report the block's end");

            spu.write(kKeyOff, 1u);
            for (int sample = 0; sample < 8; ++sample)
                spu.tick(left, right);
            t.Equals(static_cast<int>(left), 0, "a released voice should fall silent");
            t.Equals(static_cast<uint32_t>(spu.read(kVoiceEnvX)), 0u, "and its envelope read zero");
        });

        tc.Run("a transfer lands at the transfer address and trips the interrupt address", [](TestCase &t)
        {
            SpuTestHost host;
            IopSpu2 spu(host);
            spu.write(kTsaH, 0u);
            spu.write(kTsaL, 0x3000u);
            spu.write(kIrqaH, 0u);
            spu.write(kIrqaL, 0x3002u);
            spu.write(kAttr, 0xC060u);

            const uint16_t words[4] = {0x1234u, 0x5678u, 0x9ABCu, 0xDEF0u};
            spu.dmaWrite(0, words, 4u);
            for (uint32_t index = 0; index < 4u; ++index)
                t.Equals(static_cast<uint32_t>(spu.ram()[0x3000u + index]), static_cast<uint32_t>(words[index]),
                         "the words should be in sound memory in order");
            t.Equals(static_cast<uint32_t>(spu.read(kTsaL)), 0x3004u, "the transfer address should have moved on");
            t.Equals(host.interrupts, 1u, "passing the interrupt address should raise the SPU2's interrupt once");
            t.Equals(static_cast<uint32_t>(spu.read(kIrqInfo) & 4u), 4u, "and say which core it was");
            spu.write(kAttr, 0xC020u);
            t.Equals(static_cast<uint32_t>(spu.read(kIrqInfo) & 4u), 0u,
                     "turning the core's interrupt off should acknowledge it");
        });

        tc.Run("AutoDMA plays the blocks the DMA channel has for it", [](TestCase &t)
        {
            SpuTestHost host;
            IopSpu2 spu(host);
            openSpuOutput(spu);
            spu.write(kBvolL, 0x7FFFu);
            spu.write(kBvolR, 0x7FFFu);
            host.left = 1000;
            host.right = -2000;
            host.blocksLeft = 4u;
            spu.write(kAdmas, 1u);

            int16_t left = 0, right = 0;
            for (int sample = 0; sample < 300; ++sample)
                spu.tick(left, right);
            // (Four volumes just short of one lie between the input and the output.)
            t.IsTrue(std::abs(left - 1000) <= 6, "the left half of a block should come out on the left");
            t.IsTrue(std::abs(right + 2000) <= 6, "and the right half on the right");
            t.IsTrue(host.blocksFetched >= 3u, "a block should be taken for each half of the input buffer played");
        });

        tc.Run("DMA channel 4 carries IOP memory to the first core and raises its interrupt", [](TestCase &t)
        {
            IopMemory memory;
            constexpr uint32_t kSource = 0x00100000u;
            std::vector<uint16_t> words(64u);
            for (size_t index = 0; index < words.size(); ++index)
                words[index] = static_cast<uint16_t>(0x4000u + index);
            t.IsTrue(memory.writeRam(kSource, words.data(), words.size() * sizeof(uint16_t)), "test data should fit IOP memory");

            memory.write16(0xBF900000u + kTsaL, 0x4000u);
            memory.write16(0xBF900000u + kTsaH, 0u);
            memory.write16(0xBF900000u + kAttr, 0xC020u);
            memory.write32(0xBF8010C0u, kSource);
            memory.write16(0xBF8010C4u, 0x0010u);
            memory.write16(0xBF8010C6u, 0x0002u);
            memory.write32(0xBF8010C8u, 0x01000201u);

            bool same = true;
            for (size_t index = 0; index < words.size(); ++index)
                same = same && memory.spu2().ram()[0x4000u + index] == words[index];
            t.IsTrue(same, "two blocks of sixteen words should be in sound memory at the transfer address");
            t.Equals(memory.read32(0xBF8010C8u) & 0x01000000u, 0u, "the channel should no longer be busy");
            t.Equals(static_cast<uint32_t>(memory.read16(0xBF900000u + kStatx) & 0x80u), 0x80u,
                     "STATX should report the transfer done, as LIBSD waits for");
            int raised = 0;
            memory.takeRaisedInterrupts([&](const IopMemory::RaisedInterrupt &interrupt) { raised = interrupt.irq; });
            t.Equals(raised, 0x24, "the channel's interrupt should be waiting for the emulator");
        });
    });
}
