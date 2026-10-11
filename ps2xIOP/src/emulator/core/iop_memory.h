#pragma once

#include "iop_spu2.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace ps2x::iop::detail
{
    class IopMemory final : private IopSpu2Host
    {
    public:
        static constexpr uint32_t RamSize = 2u * 1024u * 1024u;
        static constexpr uint32_t ScratchBase = 0x1F800000u;
        static constexpr uint32_t ScratchSize = 0x400u;
        static constexpr uint32_t HardwareBase = 0x1F801000u;
        static constexpr uint32_t HardwareEnd = 0x1F900000u;
        static constexpr uint32_t Spu2Base = 0x1F900000u;
        static constexpr uint32_t Spu2End = 0x1FA00000u;
        static constexpr uint32_t SifBase = 0x1D000000u;
        static constexpr uint32_t SifEnd = 0x1D001000u;
        // Modules load from 0x10000 up to HeapBase; the heap (AllocSysMemory,
        // sceSifAllocIopHeap) runs from here to HeapLimit. A real IOP shares one
        // pool, and typical module sets need well under 512 KB, while games
        // allocate ~1 MB+ of heap (Drakengard: 832 KB of EE-requested buffers
        // before its opening movie), so give the heap the larger share.
        static constexpr uint32_t HeapBase = 0x00090000u;
        static constexpr uint32_t HeapLimit = 0x001F0000u;

        struct Allocation
        {
            uint32_t address = 0;
            uint32_t size = 0;
        };

        // An interrupt a device has raised, for the emulator to deliver
        // that many cycles from now.
        struct RaisedInterrupt
        {
            int irq = 0;
            uint64_t delayCycles = 0;
        };

        IopMemory();

        void reset();

        [[nodiscard]] uint8_t read8(uint32_t address) const;
        [[nodiscard]] uint16_t read16(uint32_t address) const;
        [[nodiscard]] uint32_t read32(uint32_t address) const;
        void write8(uint32_t address, uint8_t value);
        void write16(uint32_t address, uint16_t value);
        void write32(uint32_t address, uint32_t value);

        [[nodiscard]] bool readRam(uint32_t address, void *destination, size_t size) const;
        [[nodiscard]] bool writeRam(uint32_t address, const void *source, size_t size);
        [[nodiscard]] bool zeroRam(uint32_t address, size_t size);
        [[nodiscard]] bool ownsRamRange(uint32_t address, size_t size) const;
        [[nodiscard]] bool isHardwareAddress(uint32_t address) const;
        [[nodiscard]] std::string readString(uint32_t address, size_t limit = 1024u) const;

        [[nodiscard]] uint32_t allocate(uint32_t size, uint32_t alignment = 16u, std::optional<uint32_t> fixed = std::nullopt);
        [[nodiscard]] bool freeAllocation(uint32_t address);
        [[nodiscard]] uint32_t maxFreeMemory() const;
        [[nodiscard]] std::optional<Allocation> allocationContaining(uint32_t address) const;

        [[nodiscard]] uint32_t interruptStatus() const noexcept { return m_interruptStatus; }
        [[nodiscard]] uint32_t interruptMask() const noexcept { return m_interruptMask; }
        [[nodiscard]] uint32_t interruptControl() const noexcept { return m_interruptControl; }
        void setInterruptStatus(uint32_t value) noexcept { m_interruptStatus = value; }
        void setInterruptMask(uint32_t value) noexcept { m_interruptMask = value; }
        void setInterruptControl(uint32_t value) noexcept { m_interruptControl = value & 1u; }

        [[nodiscard]] bool interruptsRaised() const noexcept { return !m_raised.empty(); }
        template <typename Deliver>
        void takeRaisedInterrupts(Deliver &&deliver)
        {
            for (const RaisedInterrupt &raised : m_raised)
                deliver(raised);
            m_raised.clear();
        }
        [[nodiscard]] IopSpu2 &spu2() noexcept { return m_spu2; }
        [[nodiscard]] std::span<const uint8_t> ram() const noexcept { return m_ram; }

        [[nodiscard]] static uint32_t physicalAddress(uint32_t address) noexcept;

    private:
        // One of the two DMA channels that feed the SPU2: 4 for the first
        // core, 7 for the second.
        struct SpuDma
        {
            uint32_t madr = 0;
            uint32_t bcr = 0;
            uint32_t chcr = 0;
            uint32_t remaining = 0; // bytes AutoDMA has still to take

            // The register `offset` bytes into the channel's three.
            [[nodiscard]] uint32_t &at(uint32_t offset) { return offset < 4u ? madr : offset < 8u ? bcr : chcr; }
            [[nodiscard]] uint32_t at(uint32_t offset) const { return offset < 4u ? madr : offset < 8u ? bcr : chcr; }
        };

        [[nodiscard]] uint32_t readHardware32(uint32_t address) const;
        // `written`: the bits of the register this write carries.
        void writeHardware32(uint32_t address, uint32_t value, uint32_t written = 0xFFFFFFFFu);
        void markOwned(uint32_t address, size_t size);
        void startSpuDma(int channel);
        bool fetchAutoDma(int core, uint16_t *block) override;
        void raiseSpuInterrupt() override;

        std::vector<uint8_t> m_ram;
        std::vector<uint8_t> m_owned;
        std::vector<uint8_t> m_scratch;
        std::unordered_map<uint32_t, uint32_t> m_hardware;
        std::vector<Allocation> m_allocations;
        uint32_t m_heapCursor = HeapBase;
        uint32_t m_interruptStatus = 0;
        uint32_t m_interruptMask = 0;
        uint32_t m_interruptControl = 1;
        mutable IopSpu2 m_spu2{*this};
        SpuDma m_spuDma[2];
        std::vector<uint16_t> m_dmaWords;
        std::vector<RaisedInterrupt> m_raised;
    };
}
