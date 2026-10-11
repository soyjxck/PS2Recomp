// Based on Blackline Interactive implementation
#include "runtime/ps2_memory.h"
#include "runtime/vu1_worker.h"
#include <algorithm>
#include <cstdio>
#include <cstring>

enum VIFCmd : uint8_t
{
    VIF_NOP = 0x00,
    VIF_STCYCL = 0x01,
    VIF_OFFSET = 0x02,
    VIF_BASE = 0x03,
    VIF_ITOP = 0x04,
    VIF_STMOD = 0x05,
    VIF_MSKPATH3 = 0x06,
    VIF_MARK = 0x07,
    VIF_FLUSHE = 0x10,
    VIF_FLUSH = 0x11,
    VIF_FLUSHA = 0x13,
    VIF_MSCAL = 0x14,
    VIF_MSCALF = 0x15,
    VIF_MSCNT = 0x17,
    VIF_STMASK = 0x20,
    VIF_STROW = 0x30,
    VIF_STCOL = 0x31,
    VIF_MPG = 0x4A,
    VIF_DIRECT = 0x50,
    VIF_DIRECTHL = 0x51,
};

// PS2X_VIF1_TRACE=<secs>[:count] (armed by the runtime at that emulated
// time): one line for each VIFcode VIF1 is sent, in the order it gets them,
// and for a DIRECT what the GIF packets in it are: the registers an A+D
// packet sets with their values, the size of an image. What a display list
// does between two draws (which texture it loads, which state it sets) is
// read off this.
int g_ps2xVif1TraceRemaining = 0;

namespace
{
    constexpr uint8_t kGifFmtImage = 2u;

    void traceVif1Direct(const uint8_t *data, uint32_t sizeBytes)
    {
        uint32_t offset = 0u;
        while (offset + 16u <= sizeBytes)
        {
            uint64_t tagLo = 0u, tagHi = 0u;
            std::memcpy(&tagLo, data + offset, sizeof(tagLo));
            std::memcpy(&tagHi, data + offset + 8u, sizeof(tagHi));
            offset += 16u;
            const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFFu);
            const uint32_t flg = static_cast<uint32_t>((tagLo >> 58) & 0x3u);
            uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xFu);
            if (nreg == 0u)
                nreg = 16u;
            if (flg == 2u || flg == 3u)
            {
                std::fprintf(stderr, "[vif1]     image %u qwords (%u here)\n", nloop, (sizeBytes - offset) / 16u);
                offset += nloop * 16u;
                continue;
            }
            if (flg == 1u)
            {
                std::fprintf(stderr, "[vif1]     reglist nloop=%u nreg=%u\n", nloop, nreg);
                offset += ((nloop * nreg * 8u) + 15u) & ~15u;
                continue;
            }
            if (nreg == 1u && (tagHi & 0xFu) == 0xEu)
            {
                for (uint32_t i = 0; i < nloop && offset + 16u <= sizeBytes; ++i, offset += 16u)
                {
                    uint64_t value = 0u, reg = 0u;
                    std::memcpy(&value, data + offset, sizeof(value));
                    std::memcpy(&reg, data + offset + 8u, sizeof(reg));
                    std::fprintf(stderr, "[vif1]     reg 0x%02x = 0x%016llx\n", static_cast<uint32_t>(reg & 0xFFu),
                                 static_cast<unsigned long long>(value));
                }
                continue;
            }
            std::fprintf(stderr, "[vif1]     packed nloop=%u nreg=%u regs=0x%llx prim=0x%x (tag %016llx %016llx)\n", nloop, nreg,
                         static_cast<unsigned long long>(tagHi), static_cast<uint32_t>((tagLo >> 47) & 0x7FFu),
                         static_cast<unsigned long long>(tagLo), static_cast<unsigned long long>(tagHi));
            offset += nloop * nreg * 16u;
        }
    }

    void traceVif1Code(uint32_t cmd, const uint8_t *data, uint32_t pos, uint32_t sizeBytes, uint32_t pendingImageQw)
    {
        const uint8_t opcode = (cmd >> 24) & 0x7Fu;
        const uint32_t imm = cmd & 0xFFFFu;
        const uint32_t num = (cmd >> 16) & 0xFFu;
        if (opcode == VIF_NOP)
            return;
        --g_ps2xVif1TraceRemaining;
        const char *name = nullptr;
        switch (opcode)
        {
        case VIF_STCYCL: name = "STCYCL"; break;
        case VIF_OFFSET: name = "OFFSET"; break;
        case VIF_BASE: name = "BASE"; break;
        case VIF_ITOP: name = "ITOP"; break;
        case VIF_STMOD: name = "STMOD"; break;
        case VIF_MSKPATH3: name = "MSKPATH3"; break;
        case VIF_MARK: name = "MARK"; break;
        case VIF_FLUSHE: name = "FLUSHE"; break;
        case VIF_FLUSH: name = "FLUSH"; break;
        case VIF_FLUSHA: name = "FLUSHA"; break;
        case VIF_MSCAL: name = "MSCAL"; break;
        case VIF_MSCALF: name = "MSCALF"; break;
        case VIF_MSCNT: name = "MSCNT"; break;
        case VIF_STMASK: name = "STMASK"; break;
        case VIF_STROW: name = "STROW"; break;
        case VIF_STCOL: name = "STCOL"; break;
        case VIF_MPG: name = "MPG"; break;
        case VIF_DIRECT: name = "DIRECT"; break;
        case VIF_DIRECTHL: name = "DIRECTHL"; break;
        default: name = (opcode & 0x60u) == 0x60u ? "UNPACK" : "?"; break;
        }
        std::fprintf(stderr, "[vif1] +%06x %s%s imm=0x%x num=%u\n", pos - 4u, name, (cmd & 0x80000000u) ? " (irq)" : "", imm, num);
        if (opcode == VIF_DIRECT || opcode == VIF_DIRECTHL)
        {
            // The start of it may be image data an earlier tag announced.
            const uint32_t qwords = std::min<uint32_t>(imm == 0u ? 65536u : imm, (sizeBytes - pos) / 16u);
            const uint32_t image = std::min(pendingImageQw, qwords);
            traceVif1Direct(data + pos + image * 16u, (qwords - image) * 16u);
        }
    }

    uint32_t pendingGifImageQwc(const uint8_t *data, uint32_t sizeBytes)
    {
        if (!data || sizeBytes < 16u)
            return 0u;

        uint32_t offset = 0u;
        while (offset + 16u <= sizeBytes)
        {
            uint64_t tagLo = 0u;
            std::memcpy(&tagLo, data + offset, sizeof(tagLo));
            offset += 16u;

            const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFFu);
            const uint8_t flg = static_cast<uint8_t>((tagLo >> 58) & 0x3u);
            uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xFu);
            if (nreg == 0u)
                nreg = 16u;

            uint64_t payloadBytes = 0u;
            if (flg == 0u) // PACKED
            {
                payloadBytes = static_cast<uint64_t>(nloop) * nreg * 16ull;
            }
            else if (flg == 1u) // REGLIST, padded to a quadword
            {
                payloadBytes = static_cast<uint64_t>(nloop) * nreg * 8ull;
                payloadBytes = (payloadBytes + 15ull) & ~15ull;
            }
            else if (flg == kGifFmtImage)
            {
                payloadBytes = static_cast<uint64_t>(nloop) * 16ull;
                const uint64_t availableBytes = sizeBytes - offset;
                if (payloadBytes > availableBytes)
                {
                    return static_cast<uint32_t>((payloadBytes - availableBytes) / 16ull);
                }
            }
            else
            {
                return 0u;
            }

            if (payloadBytes > static_cast<uint64_t>(sizeBytes - offset))
                return 0u;
            offset += static_cast<uint32_t>(payloadBytes);
        }

        return 0u;
    }

    // UNPACK: decompress vectors from the VIF stream into VU data memory,
    // honouring STCYCL (skipping and filling writes), the STMASK/STROW/STCOL
    // write masks and the STMOD addition modes. `addressMask` wraps qword
    // addresses to the VU's data memory; only VIF1 has the TOPS-relative form.
    // Returns false when the stream ends before the UNPACK's data does.
    bool unpackVectors(VIFRegisters &regs, uint8_t *vuData, uint32_t vuDataSize, uint32_t addressMask,
                       bool hasDoubleBuffer, uint8_t opcode, uint16_t imm, uint8_t num,
                       const uint8_t *data, uint32_t &pos, uint32_t sizeBytes)
    {
        uint8_t vn = (opcode >> 2) & 0x3;
        uint8_t vl = opcode & 0x3;
        const bool maskEnable = (opcode & 0x10u) != 0u;
        int components = vn + 1;
        int bitsPerComponent = 32;
        switch (vl)
        {
        case 0:
            bitsPerComponent = 32;
            break;
        case 1:
            bitsPerComponent = 16;
            break;
        case 2:
            bitsPerComponent = 8;
            break;
        case 3:
            bitsPerComponent = (vn == 3) ? 4 : 16;
            break;
        default:
            break;
        }
        int bitsPerVector = (vl == 3 && vn == 3) ? 16 : (components * bitsPerComponent);
        uint32_t bytesPerVector = (bitsPerVector + 7) / 8;
        // UNPACK semantics: NUM is 8-bit and NUM==0 means 256 vectors (writes).
        const uint32_t writeVectorCount = (num == 0u) ? 256u : static_cast<uint32_t>(num);

        // STCYCL controls write cycles for UNPACK.
        uint32_t cl = regs.cycle & 0xFFu;
        uint32_t wl = (regs.cycle >> 8) & 0xFFu;
        if (cl == 0u)
            cl = 1u;
        if (wl == 0u)
            wl = 1u;

        uint32_t sourceVectorCount = writeVectorCount;
        if (cl < wl)
        {
            const uint32_t fullBlocks = writeVectorCount / wl;
            uint32_t remainder = writeVectorCount % wl;
            if (remainder > cl)
                remainder = cl;
            sourceVectorCount = fullBlocks * cl + remainder;
        }

        uint32_t totalBytes = sourceVectorCount * bytesPerVector;
        totalBytes = (totalBytes + 3) & ~3u;

        uint32_t vuAddr = (uint32_t)imm & addressMask;
        if (hasDoubleBuffer && (imm & 0x8000u) != 0u)
            vuAddr = (vuAddr + (regs.tops & addressMask)) & addressMask;

        const bool zeroExtend = (imm & 0x4000u) != 0u;

        if (vuData && totalBytes > 0 && pos + totalBytes <= sizeBytes)
        {
            const uint8_t *srcBase = data + pos;
            uint32_t srcIndex = 0u;
            for (uint32_t writeIndex = 0; writeIndex < writeVectorCount; ++writeIndex)
            {
                const uint32_t cyclePos = writeIndex % wl;
                const bool sourceAvailable = (cl >= wl) || (cyclePos < cl);

                uint32_t destVec = 0;
                if (cl >= wl)
                {
                    destVec = (vuAddr + (writeIndex / wl) * cl + cyclePos) & addressMask;
                }
                else
                {
                    destVec = (vuAddr + writeIndex) & addressMask;
                }

                uint32_t destOff = destVec * 16u;
                if (destOff + 16u > vuDataSize)
                {
                    if (sourceAvailable && srcIndex < sourceVectorCount)
                        ++srcIndex;
                    continue;
                }

                uint32_t lanes[4] = {0u, 0u, 0u, 0u};
                std::memcpy(lanes, vuData + destOff, sizeof(lanes));
                uint32_t decompressed[4] = {lanes[0], lanes[1], lanes[2], lanes[3]};
                bool decoded = false;

                const uint8_t *srcVec = nullptr;
                if (sourceAvailable && srcIndex < sourceVectorCount)
                {
                    srcVec = srcBase + srcIndex * bytesPerVector;
                    ++srcIndex;
                    decoded = true;
                }

                auto extend16 = [&](uint16_t raw) -> uint32_t
                {
                    if (zeroExtend)
                        return static_cast<uint32_t>(raw);
                    return static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(raw)));
                };

                auto extend8 = [&](uint8_t raw) -> uint32_t
                {
                    if (zeroExtend)
                        return static_cast<uint32_t>(raw);
                    return static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(raw)));
                };

                bool handledFormat = true;
                if (!decoded)
                {
                    handledFormat = false;
                }
                else if (vl == 0u)
                {
                    if (components == 1)
                    {
                        uint32_t scalar = 0;
                        std::memcpy(&scalar, srcVec, sizeof(scalar));
                        decompressed[0] = scalar;
                        decompressed[1] = scalar;
                        decompressed[2] = scalar;
                        decompressed[3] = scalar;
                    }
                    else
                    {
                        const uint32_t limit = (components > 4) ? 4u : static_cast<uint32_t>(components);
                        for (uint32_t c = 0; c < limit; ++c)
                        {
                            uint32_t scalar = 0;
                            std::memcpy(&scalar, srcVec + c * 4u, sizeof(scalar));
                            decompressed[c] = scalar;
                        }
                    }
                }
                else if (vl == 1u)
                {
                    if (components == 1)
                    {
                        uint16_t raw = 0;
                        std::memcpy(&raw, srcVec, sizeof(raw));
                        const uint32_t scalar = extend16(raw);
                        decompressed[0] = scalar;
                        decompressed[1] = scalar;
                        decompressed[2] = scalar;
                        decompressed[3] = scalar;
                    }
                    else
                    {
                        const uint32_t limit = (components > 4) ? 4u : static_cast<uint32_t>(components);
                        for (uint32_t c = 0; c < limit; ++c)
                        {
                            uint16_t raw = 0;
                            std::memcpy(&raw, srcVec + c * 2u, sizeof(raw));
                            decompressed[c] = extend16(raw);
                        }
                    }
                }
                else if (vl == 2u)
                {
                    if (components == 1)
                    {
                        const uint32_t scalar = extend8(srcVec[0]);
                        decompressed[0] = scalar;
                        decompressed[1] = scalar;
                        decompressed[2] = scalar;
                        decompressed[3] = scalar;
                    }
                    else
                    {
                        const uint32_t limit = (components > 4) ? 4u : static_cast<uint32_t>(components);
                        for (uint32_t c = 0; c < limit; ++c)
                        {
                            decompressed[c] = extend8(srcVec[c]);
                        }
                    }
                }
                else if (vl == 3u && vn == 3u)
                {
                    // V4-5: packed color-like format in a single 16-bit value.
                    uint16_t packed = 0;
                    std::memcpy(&packed, srcVec, sizeof(packed));
                    // Each 5-bit channel lands in the top of a byte, alpha as 0 or 0x80.
                    decompressed[0] = (packed & 0x1Fu) << 3;
                    decompressed[1] = ((packed >> 5) & 0x1Fu) << 3;
                    decompressed[2] = ((packed >> 10) & 0x1Fu) << 3;
                    decompressed[3] = ((packed >> 15) & 0x01u) << 7;
                }
                else
                {
                    handledFormat = false;
                }

                // Unknown compressed format fallback: preserve legacy raw-copy behavior.
                if (!handledFormat && decoded && !maskEnable && (regs.mode == 0u || regs.mode == 3u))
                {
                    uint32_t copyBytes = (bytesPerVector < 16u) ? bytesPerVector : 16u;
                    std::memcpy(vuData + destOff, srcVec, copyBytes);
                    continue;
                }

                const bool canAdd = (vl != 3u || vn != 3u);
                const uint32_t mode = regs.mode & 3u;
                const uint32_t colIdx = (cyclePos > 3u) ? 3u : cyclePos;
                const uint32_t maskCycle = (cyclePos > 3u) ? 3u : cyclePos;

                for (uint32_t field = 0u; field < 4u; ++field)
                {
                    uint32_t maskSpec = 0u;
                    if (maskEnable)
                    {
                        const uint32_t shift = ((maskCycle * 4u) + field) * 2u;
                        maskSpec = (regs.mask >> shift) & 0x3u;
                    }

                    // In fill-write cycles with suspended source reads, treat raw-data selections as row-fill.
                    if (!decoded && maskSpec == 0u)
                        maskSpec = 1u;

                    uint32_t writeVal = lanes[field];
                    if (maskSpec == 0u)
                    {
                        if (handledFormat)
                        {
                            writeVal = decompressed[field];
                            if (canAdd && (mode == 1u || mode == 2u))
                            {
                                writeVal = writeVal + regs.row[field];
                                if (mode == 2u)
                                    regs.row[field] = writeVal;
                            }
                        }
                    }
                    else if (maskSpec == 1u)
                    {
                        writeVal = regs.row[field];
                    }
                    else if (maskSpec == 2u)
                    {
                        writeVal = regs.col[colIdx];
                    }
                    else
                    {
                        continue; // write-protect
                    }

                    lanes[field] = writeVal;
                }

                std::memcpy(vuData + destOff, lanes, sizeof(lanes));
            }
        }
        pos += totalBytes;
        return pos <= sizeBytes;
    }
}

void PS2Memory::processVIF0Data(uint32_t srcPhys, uint32_t sizeBytes)
{
    if (sizeBytes == 0u || srcPhys >= PS2_RAM_SIZE)
        return;

    const uint64_t requestedEnd = static_cast<uint64_t>(srcPhys) + static_cast<uint64_t>(sizeBytes);
    if (requestedEnd > static_cast<uint64_t>(PS2_RAM_SIZE))
        sizeBytes = PS2_RAM_SIZE - srcPhys;

    processVIF0Data(m_rdram + srcPhys, sizeBytes);
}

void PS2Memory::processVIF0Data(const uint8_t *data, uint32_t sizeBytes)
{
    if (sizeBytes == 0u)
        return;

    uint32_t pos = 0;
    while (pos + 4 <= sizeBytes)
    {
        uint32_t cmd = 0u;
        std::memcpy(&cmd, data + pos, sizeof(cmd));
        pos += 4u;

        const uint8_t opcode = static_cast<uint8_t>((cmd >> 24) & 0x7Fu);
        const uint16_t imm = static_cast<uint16_t>(cmd & 0xFFFFu);
        const uint8_t num = static_cast<uint8_t>((cmd >> 16) & 0xFFu);
        const bool irq = (cmd & 0x80000000u) != 0u;

        vif0_regs.code = cmd;
        vif0_regs.num = num;
        if (irq)
            vif0_regs.stat |= (1u << 11);

        if (opcode == VIF_NOP)
        {
            continue;
        }
        else if (opcode == VIF_STCYCL)
        {
            vif0_regs.cycle = imm;
            continue;
        }
        else if (opcode == VIF_ITOP)
        {
            vif0_regs.itops = imm & 0x3FFu;
            continue;
        }
        else if (opcode == VIF_STMOD)
        {
            vif0_regs.mode = imm & 3u;
            continue;
        }
        else if (opcode == VIF_MARK)
        {
            vif0_regs.mark = imm;
            vif0_regs.stat |= (1u << 6);
            continue;
        }
        else if (opcode == VIF_FLUSHE || opcode == VIF_FLUSH || opcode == VIF_FLUSHA)
        {
            continue;
        }
        else if (opcode == VIF_STMASK)
        {
            if (pos + 4u > sizeBytes)
                break;
            std::memcpy(&vif0_regs.mask, data + pos, sizeof(vif0_regs.mask));
            pos += 4u;
            continue;
        }
        else if (opcode == VIF_STROW)
        {
            if (pos + 16u > sizeBytes)
                break;
            std::memcpy(vif0_regs.row, data + pos, 16u);
            pos += 16u;
            continue;
        }
        else if (opcode == VIF_STCOL)
        {
            if (pos + 16u > sizeBytes)
                break;
            std::memcpy(vif0_regs.col, data + pos, 16u);
            pos += 16u;
            continue;
        }
        else if (opcode == VIF_MPG)
        {
            const uint32_t destAddr = static_cast<uint32_t>(imm & 0x1FFu) * 8u;
            const uint32_t instructionCount = (num == 0u) ? 256u : static_cast<uint32_t>(num);
            const uint32_t mpgBytes = instructionCount * 8u;
            uint32_t copyBytes = 0u;
            if (m_vu0Code && destAddr < PS2_VU0_CODE_SIZE && mpgBytes > 0u)
            {
                copyBytes = mpgBytes;
                if (destAddr + copyBytes > PS2_VU0_CODE_SIZE)
                    copyBytes = PS2_VU0_CODE_SIZE - destAddr;
                if (pos + copyBytes <= sizeBytes)
                {
                    std::memcpy(m_vu0Code + destAddr, data + pos, copyBytes);
                    markVU0CodeModified();
                }
            }

            pos += mpgBytes;
            if (pos > sizeBytes)
                break;
            continue;
        }
        else if ((opcode & 0x60u) == 0x60u)
        {
            // Same decoder as VIF1: characters arrive as 16- and 8-bit vectors
            // with masks and offsets, not only as raw 32-bit ones.
            if (!unpackVectors(vif0_regs, m_vu0Data, PS2_VU0_DATA_SIZE, (PS2_VU0_DATA_SIZE / 16u) - 1u, false, opcode, imm, num, data, pos, sizeBytes))
                break;
            continue;
        }
        else
        {
            // Not a command VIF0 acts on here (MSCAL and friends): skip it
            // rather than drop the rest of the transfer.
            continue;
        }
    }
}

void PS2Memory::processVIF1Data(uint32_t srcPhys, uint32_t sizeBytes)
{
    if (sizeBytes == 0u || srcPhys >= PS2_RAM_SIZE)
        return;

    const uint64_t requestedEnd = static_cast<uint64_t>(srcPhys) + static_cast<uint64_t>(sizeBytes);
    if (requestedEnd > static_cast<uint64_t>(PS2_RAM_SIZE))
        sizeBytes = PS2_RAM_SIZE - srcPhys;

    processVIF1Data(m_rdram + srcPhys, sizeBytes);
}

void PS2Memory::processVIF1Data(const uint8_t *data, uint32_t sizeBytes)
{
    if (sizeBytes == 0u)
        return;

    if (Vu1Worker::defers())
    {
        Vu1Worker::postVif1(data, sizeBytes);
        return;
    }

    uint32_t pos = 0;

    // Sends qwords of image data an IMAGE tag announced earlier: the GS gets
    // them behind a tag of their own, which carries the transfer on.
    auto sendPendingImage = [&](const uint8_t *image, uint32_t chunkQw)
    {
        if (g_ps2xVif1TraceRemaining > 0)
            std::fprintf(stderr, "[vif1]     image data, %u of %u qwords\n", chunkQw, m_vif1PendingPath2ImageQwc);
        std::vector<uint8_t> imagePacket(16u + static_cast<size_t>(chunkQw) * 16u, 0u);
        const uint64_t imageTag =
            static_cast<uint64_t>(chunkQw & 0x7FFFu) |
            ((m_vif1PendingPath2ImageQwc == chunkQw) ? (1ull << 15) : 0ull) |
            (static_cast<uint64_t>(kGifFmtImage) << 58);
        std::memcpy(imagePacket.data(), &imageTag, sizeof(imageTag));
        std::memcpy(imagePacket.data() + 16u, image, static_cast<size_t>(chunkQw) * 16u);
        submitGifPacket(GifPathId::Path2, imagePacket.data(), static_cast<uint32_t>(imagePacket.size()), true, m_vif1PendingPath2DirectHl);

        m_vif1PendingPath2ImageQwc -= chunkQw;
        if (m_vif1PendingPath2ImageQwc == 0u)
        {
            m_vif1PendingPath2DirectHl = false;
            m_vif1PendingPath2ImageRaw = false;
        }
    };

    while (pos + 4 <= sizeBytes)
    {
        // The DIRECT that carried an image ran past the end of the last
        // transfer: this one starts with the rest of its data.
        if (m_vif1PendingPath2ImageQwc != 0u && m_vif1PendingPath2ImageRaw)
        {
            const uint32_t availableQw = (sizeBytes - pos) / 16u;
            if (availableQw == 0u)
            {
                break;
            }

            const uint32_t chunkQw = std::min<uint32_t>(m_vif1PendingPath2ImageQwc, availableQw);
            sendPendingImage(data + pos, chunkQw);
            pos += chunkQw * 16u;
            continue;
        }

        uint32_t cmd;
        memcpy(&cmd, data + pos, 4);
        pos += 4;

        uint8_t opcode = (cmd >> 24) & 0x7F;
        uint16_t imm = cmd & 0xFFFF;
        uint8_t num = (cmd >> 16) & 0xFF;
        const bool irq = (cmd & 0x80000000u) != 0u;

        // Track most-recent command for VIFn_CODE emulation.
        vif1_regs.code = cmd;
        vif1_regs.num = num;
        if (irq)
            vif1_regs.stat |= (1u << 11); // INT

        if (g_ps2xVif1TraceRemaining > 0)
            traceVif1Code(cmd, data, pos, sizeBytes, m_vif1PendingPath2ImageQwc);

        if (opcode == VIF_NOP)
        {
            continue;
        }
        else if (opcode == VIF_STCYCL)
        {
            vif1_regs.cycle = imm;
            continue;
        }
        else if (opcode == VIF_OFFSET)
        {
            // VIF double-buffer setup. OFFSET clears DBF and resets TOPS to BASE.
            // Do not rewrite BASE from the previous TOPS value.
            vif1_regs.ofst = imm & 0x3FFu;
            vif1_regs.tops = vif1_regs.base & 0x3FFu;
            vif1_regs.stat &= ~(1u << 7); // clear DBF
            continue;
        }
        else if (opcode == VIF_BASE)
        {
            // BASE only updates the base register. TOPS changes on OFFSET/MSCAL.
            vif1_regs.base = imm & 0x3FFu;
            continue;
        }
        else if (opcode == VIF_ITOP)
        {
            // ITOP VIFcode writes pending ITOPS; VU XITOP observes it after MSCAL/MSCNT.
            vif1_regs.itops = imm & 0x3FFu;
            continue;
        }
        else if (opcode == VIF_STMOD)
        {
            vif1_regs.mode = imm & 3u;
            continue;
        }
        else if (opcode == VIF_MSKPATH3)
        {
            // VIF command docs: MSKPATH3 uses IMMEDIATE bit 15.
            const bool wasMasked = m_path3Masked;
            m_path3Masked = (imm & 0x8000u) != 0u;
            if (wasMasked && !m_path3Masked)
                flushMaskedPath3Packets();
            continue;
        }
        else if (opcode == VIF_MARK)
        {
            vif1_regs.mark = imm;
            vif1_regs.stat |= (1u << 6); // MRK
            continue;
        }
        else if (opcode == VIF_FLUSHE || opcode == VIF_FLUSH || opcode == VIF_FLUSHA)
        {
            continue;
        }
        else if (opcode == VIF_MSCAL || opcode == VIF_MSCALF)
        {
            uint32_t startPC = (uint32_t)imm * 8u;

            const uint32_t runTop = vif1_regs.tops & 0x3FFu;
            const uint32_t runItop = vif1_regs.itops & 0x3FFu;
            vif1_regs.top = runTop;
            vif1_regs.itop = runItop;

            const bool dbf = (vif1_regs.stat & (1u << 7)) != 0u;
            if (dbf)
                vif1_regs.tops = vif1_regs.base & 0x3FFu;
            else
                vif1_regs.tops = (vif1_regs.base + vif1_regs.ofst) & 0x3FFu;
            vif1_regs.stat ^= (1u << 7); // toggle DBF

            if (m_vu1MscalCallback)
                m_vu1MscalCallback(startPC, runTop, runItop);
            continue;
        }
        else if (opcode == VIF_MSCNT)
        {
            const uint32_t runTop = vif1_regs.tops & 0x3FFu;
            const uint32_t runItop = vif1_regs.itops & 0x3FFu;
            vif1_regs.top = runTop;
            vif1_regs.itop = runItop;

            const bool dbf = (vif1_regs.stat & (1u << 7)) != 0u;
            if (dbf)
                vif1_regs.tops = vif1_regs.base & 0x3FFu;
            else
                vif1_regs.tops = (vif1_regs.base + vif1_regs.ofst) & 0x3FFu;
            vif1_regs.stat ^= (1u << 7); // toggle DBF

            if (m_vu1MscntCallback)
                m_vu1MscntCallback(runTop, runItop);
            continue;
        }
        else if (opcode == VIF_STMASK)
        {
            if (pos + 4 > sizeBytes)
                break;
            uint32_t maskValue = 0;
            std::memcpy(&maskValue, data + pos, sizeof(maskValue));
            vif1_regs.mask = maskValue;
            pos += 4;
            continue;
        }
        else if (opcode == VIF_STROW)
        {
            if (pos + 16 > sizeBytes)
                break;
            std::memcpy(vif1_regs.row, data + pos, 16);
            pos += 16;
            continue;
        }
        else if (opcode == VIF_STCOL)
        {
            if (pos + 16 > sizeBytes)
                break;
            std::memcpy(vif1_regs.col, data + pos, 16);
            pos += 16;
            continue;
        }
        else if (opcode == VIF_MPG)
        {
            uint32_t destAddr = (uint32_t)imm * 8u;
            const uint32_t instructionCount = (num == 0u) ? 256u : static_cast<uint32_t>(num);
            const uint32_t mpgBytes = instructionCount * 8u;
            if (m_vu1Code && destAddr < PS2_VU1_CODE_SIZE && mpgBytes > 0)
            {
                uint32_t copyBytes = mpgBytes;
                if (destAddr + copyBytes > PS2_VU1_CODE_SIZE)
                    copyBytes = PS2_VU1_CODE_SIZE - destAddr;
                if (pos + copyBytes <= sizeBytes)
                {
                    std::memcpy(m_vu1Code + destAddr, data + pos, copyBytes);
                    markVU1CodeModified();
                }
            }
            pos += mpgBytes;
            if (pos > sizeBytes)
                break;
            continue;
        }
        else if (opcode == VIF_DIRECT || opcode == VIF_DIRECTHL)
        {
            uint32_t qwCount = imm;
            if (qwCount == 0)
                qwCount = 65536;
            const uint32_t availableQw = (sizeBytes - pos) / 16u;
            const bool truncated = qwCount > availableQw;
            if (qwCount > availableQw)
                qwCount = availableQw;

            if (qwCount > 0)
            {
                const bool directHl = (opcode == VIF_DIRECTHL);
                const uint8_t *payload = data + pos;
                uint32_t payloadQw = qwCount;

                // An IMAGE tag in an earlier DIRECT announced more data than
                // came with it (a tag sent alone, its image by reference in
                // the next DMA tag, as sceVif1PkRefLoadImage builds it).
                // What this DIRECT sends is that data first: the VIFcodes in
                // between were codes, not image.
                if (m_vif1PendingPath2ImageQwc != 0u)
                {
                    const uint32_t chunkQw = std::min<uint32_t>(m_vif1PendingPath2ImageQwc, payloadQw);
                    sendPendingImage(payload, chunkQw);
                    payload += static_cast<size_t>(chunkQw) * 16u;
                    payloadQw -= chunkQw;
                }

                if (payloadQw > 0)
                {
                    submitGifPacket(GifPathId::Path2, payload, payloadQw * 16, true, directHl);

                    const uint32_t pendingImageQw = pendingGifImageQwc(payload, payloadQw * 16u);
                    if (pendingImageQw != 0u)
                    {
                        m_vif1PendingPath2ImageQwc = pendingImageQw;
                        m_vif1PendingPath2DirectHl = directHl;
                    }
                }
            }
            // Cut short by the end of the transfer, the image goes on at
            // the start of the next one without a VIFcode before it.
            m_vif1PendingPath2ImageRaw = truncated && m_vif1PendingPath2ImageQwc != 0u;

            pos += qwCount * 16;
            if (truncated)
            {
                pos = sizeBytes;
                break;
            }
            continue;
        }
        else if ((opcode & 0x60) == 0x60)
        {
            if (!unpackVectors(vif1_regs, m_vu1Data, PS2_VU1_DATA_SIZE, 0x3FFu, true, opcode, imm, num, data, pos, sizeBytes))
                break;
            continue;
        }
        else
        {
            continue;
        }
    }
}
