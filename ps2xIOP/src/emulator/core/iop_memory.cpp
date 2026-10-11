#include "iop_memory.h"

#include <algorithm>
#include <cstring>

namespace ps2x::iop::detail
{
    namespace
    {
        // MADR, BCR and CHCR of DMA channels 4 and 7, and their interrupts.
        constexpr uint32_t kSpuDmaBase[2] = {0x1F8010C0u, 0x1F801500u};
        constexpr int kSpuDmaIrq[2] = {0x24, 0x28};
        constexpr uint32_t kDmaToDevice = 1u;
        constexpr uint32_t kDmaStart = 1u << 24u;
        constexpr int kSpuIrq = 9;
        constexpr uint32_t kSpuRegisterBytes = 0x800u;
        constexpr uint32_t kAutoDmaBlockBytes = 1024u;

        bool isSpuRegister(uint32_t phys)
        {
            return phys - IopMemory::Spu2Base < kSpuRegisterBytes;
        }

        // Which SPU2 DMA channel's registers `address` is in, or -1.
        int spuDmaChannel(uint32_t address)
        {
            for (int channel = 0; channel < 2; ++channel)
                if (address - kSpuDmaBase[channel] < 12u)
                    return channel;
            return -1;
        }

        uint32_t alignUp(uint32_t value, uint32_t alignment)
        {
            return (value + alignment - 1u) & ~(alignment - 1u);
        }
    }

    IopMemory::IopMemory()
        : m_ram(RamSize), m_owned(RamSize), m_scratch(ScratchSize)
    {
        reset();
    }

    void IopMemory::reset()
    {
        std::fill(m_ram.begin(), m_ram.end(), uint8_t{0});
        std::fill(m_owned.begin(), m_owned.end(), uint8_t{0});
        std::fill(m_scratch.begin(), m_scratch.end(), uint8_t{0});
        m_hardware.clear();
        m_allocations.clear();
        m_heapCursor = HeapBase;
        m_interruptStatus = 0;
        m_interruptMask = 0;
        m_interruptControl = 1;
        m_spu2.reset();
        m_spuDma[0] = m_spuDma[1] = {};
        m_raised.clear();
    }

    uint32_t IopMemory::physicalAddress(uint32_t address) noexcept
    {
        return address & 0x1FFFFFFFu;
    }

    uint8_t IopMemory::read8(uint32_t address) const
    {
        const uint32_t phys = physicalAddress(address);
        if (phys < RamSize)
            return m_ram[phys];
        if (phys >= ScratchBase && phys < ScratchBase + ScratchSize)
            return m_scratch[phys - ScratchBase];
        if (isSpuRegister(phys))
            return static_cast<uint8_t>(m_spu2.read(phys - Spu2Base) >> ((phys & 1u) * 8u));
        const uint32_t value = readHardware32(phys & ~3u);
        return static_cast<uint8_t>(value >> ((phys & 3u) * 8u));
    }

    uint16_t IopMemory::read16(uint32_t address) const
    {
        const uint32_t phys = physicalAddress(address);
        if (phys + 1u < RamSize)
        {
            uint16_t value;
            std::memcpy(&value, m_ram.data() + phys, sizeof(value));
            return value;
        }
        if ((phys & 1u) == 0u && isSpuRegister(phys))
            return m_spu2.read(phys - Spu2Base);
        return static_cast<uint16_t>(read8(address) | (static_cast<uint16_t>(read8(address + 1u)) << 8u));
    }

    uint32_t IopMemory::read32(uint32_t address) const
    {
        const uint32_t phys = physicalAddress(address);
        if ((phys & 3u) == 0u && phys + 3u < RamSize)
        {
            uint32_t value;
            std::memcpy(&value, m_ram.data() + phys, sizeof(value));
            return value;
        }
        if ((phys & 3u) == 0u && phys >= ScratchBase && phys + 3u < ScratchBase + ScratchSize)
        {
            uint32_t value;
            std::memcpy(&value, m_scratch.data() + (phys - ScratchBase), sizeof(value));
            return value;
        }
        if ((phys & 3u) == 0u && isSpuRegister(phys))
            return m_spu2.read(phys - Spu2Base) |
                   (static_cast<uint32_t>(m_spu2.read(phys + 2u - Spu2Base)) << 16u);
        if ((phys & 3u) == 0u && isHardwareAddress(phys))
            return readHardware32(phys);

        return static_cast<uint32_t>(read8(address)) |
               (static_cast<uint32_t>(read8(address + 1u)) << 8u) |
               (static_cast<uint32_t>(read8(address + 2u)) << 16u) |
               (static_cast<uint32_t>(read8(address + 3u)) << 24u);
    }

    void IopMemory::write8(uint32_t address, uint8_t value)
    {
        const uint32_t phys = physicalAddress(address);
        if (phys < RamSize)
        {
            m_ram[phys] = value;
            markOwned(phys, sizeof(value));
            return;
        }
        if (phys >= ScratchBase && phys < ScratchBase + ScratchSize)
        {
            m_scratch[phys - ScratchBase] = value;
            return;
        }
        if (isSpuRegister(phys))
        {
            const uint32_t shift = (phys & 1u) * 8u;
            const uint16_t current = m_spu2.read(phys - Spu2Base);
            m_spu2.write(phys - Spu2Base,
                         static_cast<uint16_t>((current & ~(0xFFu << shift)) | (static_cast<uint32_t>(value) << shift)));
            return;
        }
        const uint32_t aligned = phys & ~3u;
        uint32_t current = readHardware32(aligned);
        const uint32_t shift = (phys & 3u) * 8u;
        current = (current & ~(0xFFu << shift)) | (static_cast<uint32_t>(value) << shift);
        writeHardware32(aligned, current, 0xFFu << shift);
    }

    void IopMemory::write16(uint32_t address, uint16_t value)
    {
        const uint32_t phys = physicalAddress(address);
        if (phys + 1u < RamSize)
        {
            std::memcpy(m_ram.data() + phys, &value, sizeof(value));
            markOwned(phys, sizeof(value));
            return;
        }
        if ((phys & 1u) == 0u && isSpuRegister(phys))
        {
            m_spu2.write(phys - Spu2Base, value);
            return;
        }
        write8(address, static_cast<uint8_t>(value));
        write8(address + 1u, static_cast<uint8_t>(value >> 8u));
    }

    void IopMemory::write32(uint32_t address, uint32_t value)
    {
        const uint32_t phys = physicalAddress(address);
        if ((phys & 3u) == 0u && phys + 3u < RamSize)
        {
            std::memcpy(m_ram.data() + phys, &value, sizeof(value));
            markOwned(phys, sizeof(value));
            return;
        }
        if ((phys & 3u) == 0u && phys >= ScratchBase && phys + 3u < ScratchBase + ScratchSize)
        {
            std::memcpy(m_scratch.data() + (phys - ScratchBase), &value, sizeof(value));
            return;
        }
        if ((phys & 3u) == 0u && isSpuRegister(phys))
        {
            m_spu2.write(phys - Spu2Base, static_cast<uint16_t>(value));
            m_spu2.write(phys + 2u - Spu2Base, static_cast<uint16_t>(value >> 16u));
            return;
        }
        if ((phys & 3u) == 0u)
        {
            writeHardware32(phys, value);
            return;
        }
        write8(address, static_cast<uint8_t>(value));
        write8(address + 1u, static_cast<uint8_t>(value >> 8u));
        write8(address + 2u, static_cast<uint8_t>(value >> 16u));
        write8(address + 3u, static_cast<uint8_t>(value >> 24u));
    }

    bool IopMemory::readRam(uint32_t address, void *destination, size_t size) const
    {
        const uint32_t phys = physicalAddress(address);
        if ((!destination && size != 0u) || phys > RamSize || size > RamSize - phys)
            return false;
        if (size != 0u)
            std::memcpy(destination, m_ram.data() + phys, size);
        return true;
    }

    bool IopMemory::writeRam(uint32_t address, const void *source, size_t size)
    {
        const uint32_t phys = physicalAddress(address);
        if ((!source && size != 0u) || phys > RamSize || size > RamSize - phys)
            return false;
        if (size != 0u)
        {
            std::memcpy(m_ram.data() + phys, source, size);
            markOwned(phys, size);
        }
        return true;
    }

    bool IopMemory::zeroRam(uint32_t address, size_t size)
    {
        const uint32_t phys = physicalAddress(address);
        if (phys > RamSize || size > RamSize - phys)
            return false;
        if (size != 0u)
        {
            std::memset(m_ram.data() + phys, 0, size);
            markOwned(phys, size);
        }
        return true;
    }

    bool IopMemory::ownsRamRange(uint32_t address, size_t size) const
    {
        const uint32_t phys = physicalAddress(address);
        if (phys > RamSize || size > RamSize - phys)
            return false;
        return std::all_of(m_owned.begin() + phys, m_owned.begin() + phys + size,
                           [](uint8_t value)
                           { return value != 0u; });
    }

    void IopMemory::markOwned(uint32_t address, size_t size)
    {
        if (address > RamSize || size > RamSize - address)
            return;
        std::fill(m_owned.begin() + address, m_owned.begin() + address + size, uint8_t{1});
    }

    bool IopMemory::isHardwareAddress(uint32_t address) const
    {
        const uint32_t phys = physicalAddress(address);
        return (phys >= HardwareBase && phys < HardwareEnd) ||
               (phys >= Spu2Base && phys < Spu2End) ||
               (phys >= SifBase && phys < SifEnd);
    }

    uint32_t IopMemory::readHardware32(uint32_t address) const
    {
        if (const int channel = spuDmaChannel(address); channel >= 0)
            return m_spuDma[channel].at(address - kSpuDmaBase[channel]);
        const auto value = m_hardware.find(address);
        if (value != m_hardware.end())
            return value->second;
        switch (address)
        {
        case 0x1F801070u:
            return m_interruptStatus;
        case 0x1F801074u:
            return m_interruptMask;
        case 0x1F801078u:
            return m_interruptControl;
        default:
            return 0u;
        }
    }

    void IopMemory::writeHardware32(uint32_t address, uint32_t value, uint32_t written)
    {
        switch (address)
        {
        case 0x1F801070u:
            m_interruptStatus &= value;
            return;
        case 0x1F801074u:
            m_interruptMask = value;
            return;
        case 0x1F801078u:
            m_interruptControl = value & 1u;
            return;
        default:
            break;
        }

        if (const int channel = spuDmaChannel(address); channel >= 0)
        {
            SpuDma &dma = m_spuDma[channel];
            uint32_t &reg = dma.at(address - kSpuDmaBase[channel]);
            reg = value;
            dma.madr &= 0x00FFFFFFu;
            if (&reg == &dma.chcr && (written & value & kDmaStart) != 0u)
                startSpuDma(channel);
            return;
        }
        m_hardware[address] = value;
    }

    void IopMemory::startSpuDma(int channel)
    {
        SpuDma &dma = m_spuDma[channel];
        const uint32_t wordsPerBlock = std::max<uint32_t>(dma.bcr & 0xFFFFu, 1u);
        const uint32_t blockCount = std::max<uint32_t>(dma.bcr >> 16u, 1u);
        const uint64_t bytes = static_cast<uint64_t>(wordsPerBlock) * blockCount * 4u;
        const bool toDevice = (dma.chcr & kDmaToDevice) != 0u;
        m_spu2.dmaStarted(channel);
        if (toDevice && m_spu2.autoDmaEnabled(channel))
        {
            // AutoDMA: the SPU2 takes this a block at a time as it plays
            // (fetchAutoDma), and the channel is done when it has it all.
            dma.remaining = static_cast<uint32_t>(std::min<uint64_t>(bytes, RamSize));
            m_spu2.trace(channel == 0 ? "autodma starts, core 0: at, bytes" : "autodma starts, core 1: at, bytes", dma.madr, dma.remaining);
            return;
        }

        const uint32_t address = dma.madr & (RamSize - 1u);
        const uint32_t length = static_cast<uint32_t>(std::min<uint64_t>(bytes, RamSize - address)) & ~1u;
        m_dmaWords.resize(length / 2u);
        if (toDevice)
        {
            std::memcpy(m_dmaWords.data(), m_ram.data() + address, length);
            m_spu2.dmaWrite(channel, m_dmaWords.data(), length / 2u);
        }
        else
        {
            m_spu2.dmaRead(channel, m_dmaWords.data(), length / 2u);
            std::memcpy(m_ram.data() + address, m_dmaWords.data(), length);
            markOwned(address, length);
        }
        dma.madr = (dma.madr + length) & 0x00FFFFFFu;
        dma.chcr &= ~kDmaStart;
        m_raised.push_back({kSpuDmaIrq[channel], std::max<uint64_t>(length / 2u, 64u)});
    }

    bool IopMemory::fetchAutoDma(int core, uint16_t *block)
    {
        SpuDma &dma = m_spuDma[core];
        if ((dma.chcr & kDmaStart) == 0u || dma.remaining < kAutoDmaBlockBytes)
            return false;
        const uint32_t address = dma.madr & (RamSize - 1u);
        if (address > RamSize - kAutoDmaBlockBytes)
            std::memset(block, 0, kAutoDmaBlockBytes);
        else
            std::memcpy(block, m_ram.data() + address, kAutoDmaBlockBytes);
        dma.madr = (dma.madr + kAutoDmaBlockBytes) & 0x00FFFFFFu;
        dma.remaining -= kAutoDmaBlockBytes;
        if (dma.remaining < kAutoDmaBlockBytes)
        {
            m_spu2.trace(core == 0 ? "autodma done, core 0: at" : "autodma done, core 1: at", dma.madr, 0u);
            dma.remaining = 0u;
            dma.chcr &= ~kDmaStart;
            m_raised.push_back({kSpuDmaIrq[core], 0u});
        }
        return true;
    }

    void IopMemory::raiseSpuInterrupt()
    {
        m_raised.push_back({kSpuIrq, 0u});
    }

    uint32_t IopMemory::allocate(uint32_t size, uint32_t alignment, std::optional<uint32_t> fixed)
    {
        size = alignUp(std::max(size, 1u), 16u);
        alignment = std::max<uint32_t>(alignment, 4u);
        if (fixed)
        {
            const uint32_t address = *fixed;
            if (address < HeapBase || address + size > HeapLimit)
                return 0u;
            for (const auto &block : m_allocations)
                if (address < block.address + block.size && block.address < address + size)
                    return 0u;
            m_allocations.push_back({address, size});
            markOwned(address, size);
            return address;
        }

        uint32_t candidate = alignUp(m_heapCursor, alignment);
        for (;;)
        {
            bool overlap = false;
            for (const auto &block : m_allocations)
            {
                if (candidate < block.address + block.size && block.address < candidate + size)
                {
                    candidate = alignUp(block.address + block.size, alignment);
                    overlap = true;
                    break;
                }
            }
            if (!overlap)
                break;
        }
        if (candidate > HeapLimit || size > HeapLimit - candidate)
            return 0u;
        m_allocations.push_back({candidate, size});
        markOwned(candidate, size);
        m_heapCursor = std::max(m_heapCursor, candidate + size);
        return candidate;
    }

    bool IopMemory::freeAllocation(uint32_t address)
    {
        const auto block = std::find_if(m_allocations.begin(), m_allocations.end(),
                                        [&](const Allocation &candidate)
                                        { return candidate.address == address; });
        if (block == m_allocations.end())
            return false;
        std::fill(m_owned.begin() + block->address,
                  m_owned.begin() + block->address + block->size,
                  uint8_t{0});
        m_allocations.erase(block);
        return true;
    }

    uint32_t IopMemory::maxFreeMemory() const
    {
        return m_heapCursor < HeapLimit ? HeapLimit - m_heapCursor : 0u;
    }

    std::optional<IopMemory::Allocation> IopMemory::allocationContaining(uint32_t address) const
    {
        const auto block = std::find_if(m_allocations.begin(), m_allocations.end(),
                                        [&](const Allocation &candidate)
                                        {
                                            return address >= candidate.address &&
                                                   address < candidate.address + candidate.size;
                                        });
        if (block == m_allocations.end())
            return std::nullopt;
        return *block;
    }

    std::string IopMemory::readString(uint32_t address, size_t limit) const
    {
        std::string result;
        result.reserve(std::min<size_t>(limit, 64u));
        for (size_t i = 0; i < limit; ++i)
        {
            const char ch = static_cast<char>(read8(address + static_cast<uint32_t>(i)));
            if (ch == '\0')
                break;
            result.push_back(ch);
        }
        return result;
    }
}
