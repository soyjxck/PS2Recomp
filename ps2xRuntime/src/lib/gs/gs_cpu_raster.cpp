// Fast paths of the CPU rasteriser.
//
// GSCpuBackend's original DrawTriangle/DrawSprite walk a bounding box and send
// every pixel through SampleTexture and WritePixel, which look the surface
// formats up again per pixel through function pointers. They stay as the
// reference (PS2X_GS_REFERENCE=1 forces them). The paths here produce the same
// pixels from a pipeline resolved once per primitive:
//
//   - surfaces (frame, depth, texture) become a SurfaceLayout, so a pixel
//     address is two shifts and a table read;
//   - tests, blending, fog and the texture function are decoded up front;
//   - a triangle is walked span by span instead of over its bounding box;
//   - CLUT entries are converted to RGBA once per CLUT load, on first use;
//   - textures are sampled from a decoded RGBA copy, kept while the VRAM pages
//     it came from are unchanged (or are rewritten with the same contents,
//     which is what a game that streams its textures does every frame).
//
// One deliberate difference: a pixel centre exactly on an edge belongs to the
// triangle only for top and left edges, so two triangles that share an edge do
// not both draw it (the reference accepts it on every edge and blends shared
// edges twice).

#include "runtime/gs/gs_cpu_backend.h"
#include "runtime/gs/ps2_gs_common.h"
#include "runtime/gs/ps2_gs_memory.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

using namespace GSInternal;

namespace
{
    constexpr uint32_t kVramMask = static_cast<uint32_t>(GSMem::MEMORY_SIZE) - 1u;

    inline uint32_t rgba5551To8888(uint32_t c)
    {
        return ((c & 0x1Fu) << 3) | (((c >> 5) & 0x1Fu) << 11) | (((c >> 10) & 0x1Fu) << 19) | (((c >> 15) & 0x1u) << 31);
    }

    inline uint32_t rgba8888To5551(uint32_t c)
    {
        return ((c & 0xFFu) >> 3) | ((((c >> 8) & 0xFFu) >> 3) << 5) | ((((c >> 16) & 0xFFu) >> 3) << 10) | (((c >> 24) & 0xFFu) >> 7 << 15);
    }

    inline uint32_t applyTexa(const GSTexaReg &texa, uint8_t psm, uint32_t texel)
    {
        if (psm == GS_PSM_CT32)
            return texel;

        const bool rgbZero = (texel & 0x00FFFFFFu) == 0u;
        uint32_t a = texel >> 24;
        switch (psm)
        {
        case GS_PSM_CT24:
            a = (texa.aem && rgbZero) ? 0u : texa.ta0;
            break;
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
            if ((a & 0x80u) != 0u)
                a = texa.ta1;
            else
                a = (texa.aem && rgbZero) ? 0u : texa.ta0;
            break;
        default:
            break;
        }
        return (texel & 0x00FFFFFFu) | (a << 24);
    }

    inline uint32_t read32(const uint8_t *vram, uint32_t unit)
    {
        uint32_t value;
        std::memcpy(&value, vram + ((unit * 4u) & (kVramMask - 3u)), sizeof(value));
        return value;
    }

    inline void write32(uint8_t *vram, uint32_t unit, uint32_t value)
    {
        std::memcpy(vram + ((unit * 4u) & (kVramMask - 3u)), &value, sizeof(value));
    }

    inline uint32_t read16(const uint8_t *vram, uint32_t unit)
    {
        uint16_t value;
        std::memcpy(&value, vram + ((unit * 2u) & (kVramMask - 1u)), sizeof(value));
        return value;
    }

    inline void write16(uint8_t *vram, uint32_t unit, uint32_t value)
    {
        const uint16_t narrow = static_cast<uint16_t>(value);
        std::memcpy(vram + ((unit * 2u) & (kVramMask - 1u)), &narrow, sizeof(narrow));
    }

    enum class FrameKind : uint8_t
    {
        C32,
        C24,
        C16,
    };

    enum class DepthKind : uint8_t
    {
        Z32,
        Z24,
        Z16,
    };

    enum class TextureKind : uint8_t
    {
        Direct32, // CT32, Z32
        Direct24, // CT24, Z24
        Direct16, // CT16, CT16S, Z16, Z16S
        Index8,   // T8
        Index8H,  // T8H
        Index4,   // T4
        Index4HL, // T4HL
        Index4HH, // T4HH
    };

    inline int wrapCoordinate(int coordinate, int size, uint8_t mode, int regionMin, int regionMax)
    {
        switch (mode)
        {
        case 0: // REPEAT
            return static_cast<int>(static_cast<uint32_t>(coordinate) & static_cast<uint32_t>(size - 1));
        case 1: // CLAMP
            return coordinate < 0 ? 0 : (coordinate > size - 1 ? size - 1 : coordinate);
        case 2: // REGION_CLAMP
            return std::min(std::max(coordinate, regionMin), regionMax);
        default: // REGION_REPEAT
            return static_cast<int>((static_cast<uint32_t>(coordinate) & static_cast<uint32_t>(regionMin)) | static_cast<uint32_t>(regionMax));
        }
    }

    // a + (b - a) * f / 256 on all four channels, rounded; f is 0..256.
    inline uint32_t lerpColor(uint32_t a, uint32_t b, uint32_t f)
    {
        const uint32_t g = 256u - f;
        const uint32_t rb = (((a & 0x00FF00FFu) * g + (b & 0x00FF00FFu) * f + 0x00800080u) >> 8) & 0x00FF00FFu;
        const uint32_t ag = (((a >> 8) & 0x00FF00FFu) * g + ((b >> 8) & 0x00FF00FFu) * f + 0x00800080u) & 0xFF00FF00u;
        return rb | ag;
    }

    inline int floorToInt(float value)
    {
        const int truncated = static_cast<int>(value);
        return static_cast<float>(truncated) > value ? truncated - 1 : truncated;
    }

    inline uint64_t hashBytes(const uint8_t *data, size_t size, uint64_t hash)
    {
        size_t i = 0;
        for (; i + 8u <= size; i += 8u)
        {
            uint64_t word;
            std::memcpy(&word, data + i, sizeof(word));
            hash = (hash ^ word) * 0x9E3779B97F4A7C15ull;
            hash ^= hash >> 29;
        }
        for (; i < size; ++i)
            hash = (hash ^ data[i]) * 0x100000001B3ull;
        return hash;
    }

    inline uint64_t mixKey(uint64_t hash, uint64_t value)
    {
        hash = (hash ^ value) * 0x9E3779B97F4A7C15ull;
        return hash ^ (hash >> 32);
    }

    // A set of 8 KiB VRAM pages.
    constexpr uint32_t kPageCount = 512u;
    using PageSet = std::array<uint64_t, kPageCount / 64u>;

    // Adds the pages pixels (x0, y0)..(x1, y1) of a surface live in.
    inline void addPages(PageSet &set, const GSMem::SurfaceLayout &layout, int x0, int y0, int x1, int y1)
    {
        const uint32_t firstColumn = static_cast<uint32_t>(std::max(x0, 0)) >> layout.pageWidthShift;
        const uint32_t lastColumn = static_cast<uint32_t>(std::max(x1, 0)) >> layout.pageWidthShift;
        const uint32_t firstRow = static_cast<uint32_t>(std::max(y0, 0)) >> layout.pageHeightShift;
        const uint32_t lastRow = static_cast<uint32_t>(std::max(y1, 0)) >> layout.pageHeightShift;
        for (uint32_t row = firstRow; row <= lastRow; ++row)
        {
            for (uint32_t column = firstColumn; column <= lastColumn; ++column)
            {
                const uint32_t page = (layout.basePage + row * layout.pagesPerRow + column) % kPageCount;
                set[page >> 6] |= 1ull << (page & 63u);
                if (layout.spillsIntoNextPage)
                {
                    const uint32_t next = (page + 1u) % kPageCount;
                    set[next >> 6] |= 1ull << (next & 63u);
                }
            }
        }
    }

    inline bool intersects(const PageSet &a, const PageSet &b)
    {
        for (size_t i = 0; i < a.size(); ++i)
            if ((a[i] & b[i]) != 0u)
                return true;
        return false;
    }
}

namespace
{
    // Pixels are shaded a block of neighbours on one row at a time: every
    // stage is a short loop over the lanes with nothing in it that depends on
    // another lane, which the compiler turns into vector code.
    constexpr int kLanes = 8;

    struct PixelBlock
    {
        alignas(32) float dx[kLanes];
        alignas(32) uint32_t z[kLanes];
        alignas(32) int32_t r[kLanes], g[kLanes], b[kLanes], a[kLanes];
        alignas(32) int32_t fog[kLanes];
        alignas(32) float texU[kLanes], texV[kLanes];
    };
}

// Everything a pixel needs, resolved from the draw state once per primitive.
struct GSCpuBackend::FastPipeline
{
    uint8_t *vram = nullptr;
    GSCpuBackend *backend = nullptr;

    // Targets.
    GSMem::SurfaceLayout frame;
    GSMem::SurfaceLayout depth;
    FrameKind frameKind = FrameKind::C32;
    DepthKind depthKind = DepthKind::Z32;
    uint8_t framePsm = 0;
    uint32_t fbmsk = 0;
    bool zmask = false;
    int scissorX0 = 0, scissorX1 = 0, scissorY0 = 0, scissorY1 = 0;

    // Tests.
    bool alphaTest = false;
    uint8_t atst = 1;
    uint8_t aref = 0;
    uint8_t afail = 0;
    bool date = false;
    bool datm = false;
    uint8_t ztst = 1;

    // Blending.
    bool abe = false;
    bool pabe = false;
    uint8_t asel = 0, bsel = 0, csel = 0, dsel = 0, fix = 0;
    int maskAS = 0, maskAD = 0, maskBS = 0, maskBD = 0, maskDS = 0, maskDD = 0;
    bool fba = false;

    // Fog.
    bool fge = false;
    uint8_t fogR = 0, fogG = 0, fogB = 0;

    // Texture.
    bool tme = false;
    bool fst = false;
    bool linear = false;
    uint8_t tfx = 0;
    bool tcc = false;
    uint8_t texPsm = 0;
    TextureKind texKind = TextureKind::Direct32;
    GSMem::SurfaceLayout texture;
    GSTexaReg texa{};
    int texW = 1, texH = 1;
    uint8_t wrapU = 0, wrapV = 0;
    int minU = 0, maxU = 0, minV = 0, maxV = 0;
    uint8_t cpsm = 0, csm = 0, csa = 0;
    uint32_t tbp0 = 0, tbw = 0;
    // Wrapping as mask then clamp (see texel()).
    bool regionRepeat = false;
    int maskU = -1, maskV = -1;
    int lowU = 0, highU = 0, lowV = 0, highV = 0;
    // The decoded copy, when the texture can use one.
    const uint32_t *decoded = nullptr;
    // That copy is never decoded again (see DecodedTexture::immutable).
    bool decodedIsFixed = false;
    // The pixels the primitive can touch, for telling a texture that is also
    // the render target.
    int boundsX0 = 0, boundsY0 = 0, boundsX1 = 0, boundsY1 = 0;
    bool textureIsTarget = false;
    // The texels the primitive can sample (the whole texture when it wraps).
    int sampledU0 = 0, sampledV0 = 0, sampledU1 = 0, sampledV1 = 0;

    // Sets the sampled rectangle from the lowest and highest texel
    // coordinates the primitive's pixels can have, as sample() receives them;
    // `slack` widens them where they are only known approximately.
    void setSampledCoordinates(float u0, float v0, float u1, float v1, float slack)
    {
        sampledU0 = 0;
        sampledV0 = 0;
        sampledU1 = texW - 1;
        sampledV1 = texH - 1;
        if (!tme)
            return;
        if (u0 > u1)
            std::swap(u0, u1);
        if (v0 > v1)
            std::swap(v0, v1);
        auto range = [this, slack](float low, float high, int size, uint8_t wrap, int &first, int &last)
        {
            if (!(low > -32768.0f && high < 32768.0f))
                return;
            // sample() reads the texel a coordinate truncates to or, when
            // filtering, the one half a texel before it and the next.
            int lo, hi;
            if (linear)
            {
                lo = floorToInt(low - slack - 0.5f);
                hi = floorToInt(high + slack - 0.5f) + 1;
            }
            else
            {
                lo = floorToInt(low - slack);
                hi = static_cast<int>(high + slack);
            }
            if (lo < 0 || hi > size - 1)
            {
                // REPEAT brings those in from the far side; CLAMP holds them at the edge.
                if (wrap != 1u)
                    return;
                lo = std::min(std::max(lo, 0), size - 1);
                hi = std::min(std::max(hi, 0), size - 1);
            }
            first = lo;
            last = hi;
        };
        range(u0, u1, texW, wrapU, sampledU0, sampledU1);
        range(v0, v1, texH, wrapV, sampledV0, sampledV1);
    }

    bool setup(GSCpuBackend &owner, const GSDrawState &state);

    // Call once the primitive's bounds are known.
    void useDecodedTexture(int x0, int y0, int x1, int y1)
    {
        boundsX0 = x0;
        boundsY0 = y0;
        boundsX1 = x1;
        boundsY1 = y1;
        // Region wrap modes address texels outside the texture's own
        // rectangle; those sample VRAM directly.
        if (tme && wrapU < 2u && wrapV < 2u)
        {
            if (const DecodedTexture *copy = backend->DecodedTextureFor(*this))
            {
                decoded = copy->pixels.data();
                decodedIsFixed = copy->immutable;
                // Such a copy ends a row below its image; nothing the
                // primitive samples is further down.
                if (copy->immutable)
                    highV = std::min(highV, copy->height - 1);
            }
        }
    }

    [[gnu::always_inline]] inline uint32_t clutColor(uint32_t index) const
    {
        GSCpuBackend &b = *backend;
        if (b.m_fastClutStamp[index] != b.m_fastClutKey)
        {
            b.m_fastClut[index] = b.LookupCLUT(*drawState, static_cast<uint8_t>(index), cpsm, csm, csa, texPsm);
            b.m_fastClutStamp[index] = b.m_fastClutKey;
        }
        return b.m_fastClut[index];
    }

    // Straight from VRAM; (u, v) must be inside the texture.
    [[gnu::always_inline]] inline uint32_t fetch(int u, int v) const
    {
        const uint32_t unit = texture.unit(static_cast<uint32_t>(u), static_cast<uint32_t>(v));
        switch (texKind)
        {
        case TextureKind::Direct32:
            return read32(vram, unit);
        case TextureKind::Direct24:
            return applyTexa(texa, texPsm, read32(vram, unit) & 0x00FFFFFFu);
        case TextureKind::Direct16:
            return applyTexa(texa, texPsm, rgba5551To8888(read16(vram, unit)));
        case TextureKind::Index8:
            return clutColor(vram[unit & kVramMask]);
        case TextureKind::Index8H:
            return clutColor(vram[(unit * 4u + 3u) & kVramMask]);
        case TextureKind::Index4:
            return clutColor((vram[(unit >> 1) & kVramMask] >> ((unit & 1u) * 4u)) & 0x0Fu);
        case TextureKind::Index4HL:
            return clutColor(vram[(unit * 4u + 3u) & kVramMask] & 0x0Fu);
        case TextureKind::Index4HH:
            return clutColor(vram[(unit * 4u + 3u) & kVramMask] >> 4);
        }
        return 0xFFFF00FFu;
    }

    [[gnu::always_inline]] inline uint32_t texel(int u, int v) const
    {
        // REPEAT is a mask, both clamps are bounds; REGION_REPEAT is rare.
        if (regionRepeat) [[unlikely]]
        {
            u = wrapCoordinate(u, texW, wrapU, minU, maxU);
            v = wrapCoordinate(v, texH, wrapV, minV, maxV);
        }
        else
        {
            u &= maskU;
            v &= maskV;
            u = u < lowU ? lowU : (u > highU ? highU : u);
            v = v < lowV ? lowV : (v > highV ? highV : v);
        }
        if (decoded)
            return decoded[v * texW + u];
        return fetch(u, v);
    }

    // texU/texV are texel coordinates (already divided by Q and scaled).
    [[gnu::always_inline]] inline uint32_t sample(float texU, float texV) const
    {
        if (!linear)
            return texel(static_cast<int>(texU), static_cast<int>(texV));

        const float su = texU - 0.5f;
        const float sv = texV - 0.5f;
        const int u0 = floorToInt(su);
        const int v0 = floorToInt(sv);
        const uint32_t c00 = texel(u0, v0);
        const uint32_t c10 = texel(u0 + 1, v0);
        const uint32_t c01 = texel(u0, v0 + 1);
        const uint32_t c11 = texel(u0 + 1, v0 + 1);
        if (c00 == c10 && c00 == c01 && c00 == c11)
            return c00;
        const uint32_t wx = static_cast<uint32_t>((su - static_cast<float>(u0)) * 256.0f + 0.5f);
        const uint32_t wy = static_cast<uint32_t>((sv - static_cast<float>(v0)) * 256.0f + 0.5f);
        return lerpColor(lerpColor(c00, c10, wx), lerpColor(c01, c11, wx), wy);
    }

    // The TEX0 texture function: vertex colour (r, g, b, a) with a texel.
    [[gnu::always_inline]] inline void combine(uint32_t texelColor, int &r, int &g, int &b, int &a) const
    {
        const int tr = static_cast<int>(texelColor & 0xFFu);
        const int tg = static_cast<int>((texelColor >> 8) & 0xFFu);
        const int tb = static_cast<int>((texelColor >> 16) & 0xFFu);
        const int ta = static_cast<int>(texelColor >> 24);
        const int va = a;
        switch (tfx)
        {
        case 0: // MODULATE
            r = std::min((tr * r) >> 7, 255);
            g = std::min((tg * g) >> 7, 255);
            b = std::min((tb * b) >> 7, 255);
            a = tcc ? std::min((ta * va) >> 7, 255) : va;
            break;
        case 1: // DECAL
            r = tr;
            g = tg;
            b = tb;
            a = tcc ? ta : va;
            break;
        case 2: // HIGHLIGHT
            r = std::min(((tr * r) >> 7) + va, 255);
            g = std::min(((tg * g) >> 7) + va, 255);
            b = std::min(((tb * b) >> 7) + va, 255);
            a = tcc ? std::min(ta + va, 255) : va;
            break;
        default: // HIGHLIGHT2
            r = std::min(((tr * r) >> 7) + va, 255);
            g = std::min(((tg * g) >> 7) + va, 255);
            b = std::min(((tb * b) >> 7) + va, 255);
            a = tcc ? ta : va;
            break;
        }
    }

    // Fog, the tests, blending and the stores: GSCpuBackend::WritePixel for a
    // pixel already known to be inside the scissor rectangle.
    // One scanline of a surface: the page its first column is in and the
    // row of the page table that applies.
    struct Row
    {
        const uint16_t *table;
        uint32_t page;
    };

    [[gnu::always_inline]] inline static Row rowOf(const GSMem::SurfaceLayout &layout, int y)
    {
        const uint32_t uy = static_cast<uint32_t>(y);
        return {layout.table + ((uy & ((1u << layout.pageHeightShift) - 1u)) << layout.pageWidthShift),
                layout.basePage + (uy >> layout.pageHeightShift) * layout.pagesPerRow};
    }

    [[gnu::always_inline]] inline static uint32_t unitIn(const GSMem::SurfaceLayout &layout, const Row &row, int x)
    {
        const uint32_t ux = static_cast<uint32_t>(x);
        return ((row.page + (ux >> layout.pageWidthShift)) << layout.unitsPerPageShift) +
               row.table[ux & ((1u << layout.pageWidthShift) - 1u)];
    }

    [[gnu::always_inline]] inline void write(int x, const Row &frameRow, const Row &depthRow, uint32_t z, int r, int g, int b, int a, int fog) const
    {
        if (fge)
        {
            const int inverse = 255 - fog;
            r = ((fog * r) >> 8) + ((inverse * fogR) >> 8);
            g = ((fog * g) >> 8) + ((inverse * fogG) >> 8);
            b = ((fog * b) >> 8) + ((inverse * fogB) >> 8);
        }

        bool writeRgb = true, writeAlpha = true, writeDepth = true;
        if (alphaTest)
        {
            bool pass = true;
            switch (atst)
            {
            case 0: pass = false; break;
            case 1: pass = true; break;
            case 2: pass = a < aref; break;
            case 3: pass = a <= aref; break;
            case 4: pass = a == aref; break;
            case 5: pass = a >= aref; break;
            case 6: pass = a > aref; break;
            default: pass = a != aref; break;
            }
            if (!pass)
            {
                switch (afail)
                {
                case 1: // FB_ONLY
                    writeDepth = false;
                    break;
                case 2: // ZB_ONLY
                    writeRgb = writeAlpha = false;
                    break;
                case 3: // RGB_ONLY (only distinct for RGBA32)
                    writeDepth = false;
                    writeAlpha = framePsm != GS_PSM_CT32;
                    break;
                default: // KEEP
                    return;
                }
            }
        }

        const bool writesFrame = writeRgb || writeAlpha;
        const bool preserveAlpha = writeRgb && !writeAlpha && framePsm == GS_PSM_CT32;
        const uint32_t frameUnit = unitIn(frame, frameRow, x);

        uint32_t rawFrame = 0u;
        uint32_t frameRgba = 0u;
        if (date || (writesFrame && (fbmsk != 0u || abe || preserveAlpha)))
        {
            switch (frameKind)
            {
            case FrameKind::C32:
                rawFrame = read32(vram, frameUnit);
                frameRgba = rawFrame;
                break;
            case FrameKind::C24:
                rawFrame = read32(vram, frameUnit) & 0x00FFFFFFu;
                frameRgba = rawFrame | 0x80000000u; // the GS supplies 0x80 as destination alpha
                break;
            case FrameKind::C16:
                rawFrame = read16(vram, frameUnit);
                frameRgba = rgba5551To8888(rawFrame);
                break;
            }
        }

        if (date)
        {
            switch (frameKind)
            {
            case FrameKind::C32:
                if ((((rawFrame >> 31) & 1u) != 0u) != datm)
                    return;
                break;
            case FrameKind::C16:
                if ((((rawFrame >> 15) & 1u) != 0u) != datm)
                    return;
                break;
            case FrameKind::C24:
                break;
            }
        }

        uint32_t depthUnit = 0u;
        if (ztst >= 2u || (writeDepth && !zmask))
            depthUnit = unitIn(depth, depthRow, x);

        if (ztst == 0u)
            return;
        if (ztst >= 2u)
        {
            uint32_t stored = 0u;
            switch (depthKind)
            {
            case DepthKind::Z32:
                stored = read32(vram, depthUnit);
                break;
            case DepthKind::Z24:
                stored = read32(vram, depthUnit) & 0x00FFFFFFu;
                break;
            case DepthKind::Z16:
                stored = read16(vram, depthUnit);
                break;
            }
            if (ztst == 2u ? z < stored : z <= stored)
                return;
        }

        if (writesFrame)
        {
            if (abe && !(pabe && (a & 0x80) == 0))
            {
                const int dr = static_cast<int>(frameRgba & 0xFFu);
                const int dg = static_cast<int>((frameRgba >> 8) & 0xFFu);
                const int db = static_cast<int>((frameRgba >> 16) & 0xFFu);
                const int da = static_cast<int>(frameRgba >> 24);
                const int alpha = csel == 0 ? a : (csel == 1 ? da : fix);
                // ((A - B) * alpha >> 7) + D with A, B, D each source, destination
                // or zero: the selectors are masks, set up once.
                auto blend = [&](int source, int dest)
                {
                    const int difference = ((source & maskAS) | (dest & maskAD)) - ((source & maskBS) | (dest & maskBD));
                    return clampU8(((difference * alpha) >> 7) + ((source & maskDS) | (dest & maskDD)));
                };
                r = blend(r, dr);
                g = blend(g, dg);
                b = blend(b, db);
            }

            if (writeAlpha && fba && framePsm != GS_PSM_CT24)
                a |= 0x80;

            uint32_t pixel = static_cast<uint32_t>(r & 0xFF) | (static_cast<uint32_t>(g & 0xFF) << 8) |
                             (static_cast<uint32_t>(b & 0xFF) << 16) | (static_cast<uint32_t>(a & 0xFF) << 24);
            if (fbmsk != 0u)
                pixel = (pixel & ~fbmsk) | (frameRgba & fbmsk);
            if (preserveAlpha)
                pixel = (pixel & 0x00FFFFFFu) | (frameRgba & 0xFF000000u);

            switch (frameKind)
            {
            case FrameKind::C32:
                write32(vram, frameUnit, pixel);
                break;
            case FrameKind::C24:
                write32(vram, frameUnit, (read32(vram, frameUnit) & 0xFF000000u) | (pixel & 0x00FFFFFFu));
                break;
            case FrameKind::C16:
                write16(vram, frameUnit, rgba8888To5551(pixel));
                break;
            }
        }

        if (writeDepth && !zmask)
        {
            switch (depthKind)
            {
            case DepthKind::Z32:
                write32(vram, depthUnit, z);
                break;
            case DepthKind::Z24:
                write32(vram, depthUnit, (read32(vram, depthUnit) & 0xFF000000u) | (z & 0x00FFFFFFu));
                break;
            case DepthKind::Z16:
                write16(vram, depthUnit, z);
                break;
            }
        }
    }

    // write() for the first `count` lanes of a block starting at column x0,
    // sampling and applying the texture first.
    void shade(const PixelBlock &in, int x0, int count, const Row &frameRow, const Row &depthRow) const;

    const GSDrawState *drawState = nullptr;
};

[[gnu::noinline]] bool GSCpuBackend::FastPipeline::setup(GSCpuBackend &owner, const GSDrawState &state)
{
    const GSContext &ctx = state.context;
    backend = &owner;
    drawState = &state;
    vram = owner.m_vram;

    framePsm = ctx.frame.psm;
    switch (ctx.frame.psm)
    {
    case GS_PSM_CT32:
        frameKind = FrameKind::C32;
        break;
    case GS_PSM_CT24:
        frameKind = FrameKind::C24;
        break;
    case GS_PSM_CT16:
    case GS_PSM_CT16S:
        frameKind = FrameKind::C16;
        break;
    default:
        return false;
    }
    switch (ctx.zbuf.psm)
    {
    case GS_PSM_Z32:
        depthKind = DepthKind::Z32;
        break;
    case GS_PSM_Z24:
        depthKind = DepthKind::Z24;
        break;
    case GS_PSM_Z16:
    case GS_PSM_Z16S:
        depthKind = DepthKind::Z16;
        break;
    default:
        return false;
    }

    const uint32_t fbw = std::max<uint32_t>(ctx.frame.fbw, 1u);
    frame = GSMem::LayoutOf(ctx.frame.psm, framePageBaseToBlock(ctx.frame.fbp), fbw);
    depth = GSMem::LayoutOf(ctx.zbuf.psm, framePageBaseToBlock(ctx.zbuf.zbp), fbw);
    if (!frame.table || !depth.table)
        return false;

    fbmsk = ctx.frame.fbmsk;
    zmask = ctx.zbuf.zmask;
    scissorX0 = ctx.scissor.x0;
    scissorX1 = ctx.scissor.x1;
    scissorY0 = ctx.scissor.y0;
    scissorY1 = ctx.scissor.y1;

    alphaTest = (ctx.test & 0x1u) != 0u;
    atst = static_cast<uint8_t>((ctx.test >> 1) & 0x7u);
    aref = static_cast<uint8_t>((ctx.test >> 4) & 0xFFu);
    afail = static_cast<uint8_t>((ctx.test >> 12) & 0x3u);
    date = ((ctx.test >> 14) & 0x1u) != 0u && frameKind != FrameKind::C24;
    datm = ((ctx.test >> 15) & 0x1u) != 0u;
    ztst = static_cast<uint8_t>((ctx.test >> 17) & 0x3u);

    abe = state.prim.abe;
    pabe = state.pabe;
    asel = static_cast<uint8_t>(ctx.alpha & 3u);
    bsel = static_cast<uint8_t>((ctx.alpha >> 2) & 3u);
    csel = static_cast<uint8_t>((ctx.alpha >> 4) & 3u);
    dsel = static_cast<uint8_t>((ctx.alpha >> 6) & 3u);
    fix = static_cast<uint8_t>((ctx.alpha >> 32) & 0xFFu);
    maskAS = asel == 0u ? -1 : 0;
    maskAD = asel == 1u ? -1 : 0;
    maskBS = bsel == 0u ? -1 : 0;
    maskBD = bsel == 1u ? -1 : 0;
    maskDS = dsel == 0u ? -1 : 0;
    maskDD = dsel == 1u ? -1 : 0;
    fba = (ctx.fba & 0x1ull) != 0ull;

    fge = state.prim.fge;
    fogR = state.fogR;
    fogG = state.fogG;
    fogB = state.fogB;

    tme = state.prim.tme;
    if (!tme)
        return true;

    const GSTex0Reg &tex = ctx.tex0;
    switch (tex.psm)
    {
    case GS_PSM_CT32:
    case GS_PSM_Z32:
        texKind = TextureKind::Direct32;
        break;
    case GS_PSM_CT24:
    case GS_PSM_Z24:
        texKind = TextureKind::Direct24;
        break;
    case GS_PSM_CT16:
    case GS_PSM_CT16S:
    case GS_PSM_Z16:
    case GS_PSM_Z16S:
        texKind = TextureKind::Direct16;
        break;
    case GS_PSM_T8:
        texKind = TextureKind::Index8;
        break;
    case GS_PSM_T8H:
        texKind = TextureKind::Index8H;
        break;
    case GS_PSM_T4:
        texKind = TextureKind::Index4;
        break;
    case GS_PSM_T4HL:
        texKind = TextureKind::Index4HL;
        break;
    case GS_PSM_T4HH:
        texKind = TextureKind::Index4HH;
        break;
    default:
        return false;
    }
    texture = GSMem::LayoutOf(tex.psm, tex.tbp0, tex.tbw);
    if (!texture.table)
        return false;

    fst = state.prim.fst;
    linear = state.linearFilter;
    tfx = tex.tfx & 3u;
    tcc = tex.tcc != 0u;
    texPsm = tex.psm;
    texa = state.texa;
    texW = state.textureWidth;
    texH = state.textureHeight;
    wrapU = static_cast<uint8_t>(ctx.clamp & 0x3u);
    wrapV = static_cast<uint8_t>((ctx.clamp >> 2) & 0x3u);
    minU = static_cast<int>((ctx.clamp >> 4) & 0x3FFu);
    maxU = static_cast<int>((ctx.clamp >> 14) & 0x3FFu);
    minV = static_cast<int>((ctx.clamp >> 24) & 0x3FFu);
    maxV = static_cast<int>((ctx.clamp >> 34) & 0x3FFu);
    cpsm = tex.cpsm;
    csm = tex.csm;
    csa = tex.csa;
    tbp0 = tex.tbp0;
    tbw = tex.tbw;

    regionRepeat = wrapU == 3u || wrapV == 3u;
    auto wrapAs = [](uint8_t mode, int size, int regionMin, int regionMax, int &mask, int &low, int &high)
    {
        mask = mode == 0u ? size - 1 : -1;
        low = mode == 2u ? regionMin : 0;
        high = mode == 2u ? regionMax : size - 1;
    };
    wrapAs(wrapU, texW, minU, maxU, maskU, lowU, highU);
    wrapAs(wrapV, texH, minV, maxV, maskV, lowV, highV);

    if (texKind >= TextureKind::Index8)
    {
        // Converted CLUT entries stay valid until the CLUT is reloaded or the
        // conversion (format, offset, TEXA) changes.
        const uint64_t conversion = static_cast<uint64_t>(cpsm) | (static_cast<uint64_t>(csm) << 8) |
                                    (static_cast<uint64_t>(csa) << 16) | (static_cast<uint64_t>(texPsm) << 24) |
                                    (static_cast<uint64_t>(texa.ta0) << 32) | (static_cast<uint64_t>(texa.ta1) << 40) |
                                    (static_cast<uint64_t>(texa.aem ? 1u : 0u) << 48);
        if (owner.m_fastClutConversion != conversion || owner.m_fastClutLoad != owner.m_clutLoads)
        {
            owner.m_fastClutConversion = conversion;
            owner.m_fastClutLoad = owner.m_clutLoads;
            if (++owner.m_fastClutKey == 0u)
            {
                owner.m_fastClutStamp.fill(0u);
                owner.m_fastClutKey = 1u;
            }
        }
    }

    return true;
}

// The stages are the ones sample(), combine() and write() go through for one
// pixel, in the same order and with the same arithmetic, so a lane comes out
// exactly as that pixel would.
[[gnu::noinline]] void GSCpuBackend::FastPipeline::shade(const PixelBlock &in, int x0, int count, const Row &frameRow,
                                                         const Row &depthRow) const
{
    constexpr int N = kLanes;
    if (ztst == 0u)
        return;

    int32_t r[N], g[N], b[N], a[N];
    for (int i = 0; i < N; ++i)
    {
        r[i] = in.r[i];
        g[i] = in.g[i];
        b[i] = in.b[i];
        a[i] = in.a[i];
    }

    if (tme)
    {
        uint32_t texels[N];
        const uint32_t *const pixels = decoded;
        const int width = texW;
        auto wrap = [&](int32_t *u, int32_t *v)
        {
            if (regionRepeat) [[unlikely]]
            {
                for (int i = 0; i < N; ++i)
                {
                    u[i] = wrapCoordinate(u[i], texW, wrapU, minU, maxU);
                    v[i] = wrapCoordinate(v[i], texH, wrapV, minV, maxV);
                }
                return;
            }
            const int mu = maskU, mv = maskV, lu = lowU, hu = highU, lv = lowV, hv = highV;
            for (int i = 0; i < N; ++i)
            {
                const int cu = u[i] & mu;
                const int cv = v[i] & mv;
                u[i] = cu < lu ? lu : (cu > hu ? hu : cu);
                v[i] = cv < lv ? lv : (cv > hv ? hv : cv);
            }
        };

        if (!linear)
        {
            int32_t u[N], v[N];
            for (int i = 0; i < N; ++i)
            {
                u[i] = static_cast<int>(in.texU[i]);
                v[i] = static_cast<int>(in.texV[i]);
            }
            wrap(u, v);
            if (pixels)
            {
                for (int i = 0; i < N; ++i)
                    texels[i] = pixels[v[i] * width + u[i]];
            }
            else
            {
                for (int i = 0; i < N; ++i)
                    texels[i] = fetch(u[i], v[i]);
            }
        }
        else
        {
            int32_t u0[N], v0[N], u1[N], v1[N];
            uint32_t wx[N], wy[N];
            for (int i = 0; i < N; ++i)
            {
                const float su = in.texU[i] - 0.5f;
                const float sv = in.texV[i] - 0.5f;
                const int iu = floorToInt(su);
                const int iv = floorToInt(sv);
                wx[i] = static_cast<uint32_t>((su - static_cast<float>(iu)) * 256.0f + 0.5f);
                wy[i] = static_cast<uint32_t>((sv - static_cast<float>(iv)) * 256.0f + 0.5f);
                u0[i] = iu;
                v0[i] = iv;
                u1[i] = iu + 1;
                v1[i] = iv + 1;
            }
            wrap(u0, v0);
            wrap(u1, v1);
            uint32_t c00[N], c10[N], c01[N], c11[N];
            if (pixels)
            {
                for (int i = 0; i < N; ++i)
                {
                    const uint32_t *const top = pixels + v0[i] * width;
                    const uint32_t *const bottom = pixels + v1[i] * width;
                    c00[i] = top[u0[i]];
                    c10[i] = top[u1[i]];
                    c01[i] = bottom[u0[i]];
                    c11[i] = bottom[u1[i]];
                }
            }
            else
            {
                for (int i = 0; i < N; ++i)
                {
                    c00[i] = fetch(u0[i], v0[i]);
                    c10[i] = fetch(u1[i], v0[i]);
                    c01[i] = fetch(u0[i], v1[i]);
                    c11[i] = fetch(u1[i], v1[i]);
                }
            }
            for (int i = 0; i < N; ++i)
                texels[i] = lerpColor(lerpColor(c00[i], c10[i], wx[i]), lerpColor(c01[i], c11[i], wx[i]), wy[i]);
        }

        // The texture function.
        int32_t tr[N], tg[N], tb[N], ta[N];
        for (int i = 0; i < N; ++i)
        {
            tr[i] = static_cast<int32_t>(texels[i] & 0xFFu);
            tg[i] = static_cast<int32_t>((texels[i] >> 8) & 0xFFu);
            tb[i] = static_cast<int32_t>((texels[i] >> 16) & 0xFFu);
            ta[i] = static_cast<int32_t>(texels[i] >> 24);
        }
        switch (tfx)
        {
        case 0: // MODULATE
            for (int i = 0; i < N; ++i)
            {
                r[i] = std::min((tr[i] * r[i]) >> 7, 255);
                g[i] = std::min((tg[i] * g[i]) >> 7, 255);
                b[i] = std::min((tb[i] * b[i]) >> 7, 255);
            }
            if (tcc)
                for (int i = 0; i < N; ++i)
                    a[i] = std::min((ta[i] * a[i]) >> 7, 255);
            break;
        case 1: // DECAL
            for (int i = 0; i < N; ++i)
            {
                r[i] = tr[i];
                g[i] = tg[i];
                b[i] = tb[i];
            }
            if (tcc)
                for (int i = 0; i < N; ++i)
                    a[i] = ta[i];
            break;
        default: // HIGHLIGHT, HIGHLIGHT2
            for (int i = 0; i < N; ++i)
            {
                r[i] = std::min(((tr[i] * r[i]) >> 7) + a[i], 255);
                g[i] = std::min(((tg[i] * g[i]) >> 7) + a[i], 255);
                b[i] = std::min(((tb[i] * b[i]) >> 7) + a[i], 255);
            }
            if (tcc)
            {
                if (tfx == 2u)
                    for (int i = 0; i < N; ++i)
                        a[i] = std::min(ta[i] + a[i], 255);
                else
                    for (int i = 0; i < N; ++i)
                        a[i] = ta[i];
            }
            break;
        }
    }

    if (fge)
    {
        const int fr = fogR, fg = fogG, fb = fogB;
        for (int i = 0; i < N; ++i)
        {
            const int fog = in.fog[i];
            const int inverse = 255 - fog;
            r[i] = ((fog * r[i]) >> 8) + ((inverse * fr) >> 8);
            g[i] = ((fog * g[i]) >> 8) + ((inverse * fg) >> 8);
            b[i] = ((fog * b[i]) >> 8) + ((inverse * fb) >> 8);
        }
    }

    // Which lanes store to the frame and to the depth buffer, and which of
    // those keep the alpha that is there.
    int32_t frameMask[N], depthMask[N];
    uint32_t keep[N];
    for (int i = 0; i < N; ++i)
    {
        frameMask[i] = i < count ? -1 : 0;
        depthMask[i] = frameMask[i];
        keep[i] = fbmsk;
    }

    bool keepsAlpha = false;
    if (alphaTest && atst != 1u)
    {
        int32_t pass[N];
        const int ref = aref;
        switch (atst)
        {
        case 0:
            for (int i = 0; i < N; ++i)
                pass[i] = 0;
            break;
        case 2:
            for (int i = 0; i < N; ++i)
                pass[i] = a[i] < ref ? -1 : 0;
            break;
        case 3:
            for (int i = 0; i < N; ++i)
                pass[i] = a[i] <= ref ? -1 : 0;
            break;
        case 4:
            for (int i = 0; i < N; ++i)
                pass[i] = a[i] == ref ? -1 : 0;
            break;
        case 5:
            for (int i = 0; i < N; ++i)
                pass[i] = a[i] >= ref ? -1 : 0;
            break;
        case 6:
            for (int i = 0; i < N; ++i)
                pass[i] = a[i] > ref ? -1 : 0;
            break;
        default:
            for (int i = 0; i < N; ++i)
                pass[i] = a[i] != ref ? -1 : 0;
            break;
        }
        switch (afail)
        {
        case 1: // FB_ONLY
            for (int i = 0; i < N; ++i)
                depthMask[i] &= pass[i];
            break;
        case 2: // ZB_ONLY
            for (int i = 0; i < N; ++i)
                frameMask[i] &= pass[i];
            break;
        case 3: // RGB_ONLY (only distinct for RGBA32)
            keepsAlpha = framePsm == GS_PSM_CT32;
            for (int i = 0; i < N; ++i)
            {
                depthMask[i] &= pass[i];
                if (keepsAlpha)
                    keep[i] |= ~static_cast<uint32_t>(pass[i]) & 0xFF000000u;
            }
            break;
        default: // KEEP
            for (int i = 0; i < N; ++i)
            {
                frameMask[i] &= pass[i];
                depthMask[i] &= pass[i];
            }
            break;
        }
    }

    uint32_t frameUnit[N];
    for (int i = 0; i < N; ++i)
        frameUnit[i] = unitIn(frame, frameRow, x0 + i);

    uint32_t dest[N];
    if (date || fbmsk != 0u || abe || keepsAlpha)
    {
        switch (frameKind)
        {
        case FrameKind::C32:
            for (int i = 0; i < N; ++i)
                dest[i] = read32(vram, frameUnit[i]);
            break;
        case FrameKind::C24:
            for (int i = 0; i < N; ++i)
                dest[i] = (read32(vram, frameUnit[i]) & 0x00FFFFFFu) | 0x80000000u; // the GS supplies 0x80 as destination alpha
            break;
        case FrameKind::C16:
            for (int i = 0; i < N; ++i)
                dest[i] = rgba5551To8888(read16(vram, frameUnit[i]));
            break;
        }
    }
    else
    {
        for (int i = 0; i < N; ++i)
            dest[i] = 0u;
    }

    if (date)
    {
        const uint32_t wanted = datm ? 1u : 0u;
        for (int i = 0; i < N; ++i)
        {
            const int32_t pass = (dest[i] >> 31) == wanted ? -1 : 0;
            frameMask[i] &= pass;
            depthMask[i] &= pass;
        }
    }

    const bool storesDepth = !zmask;
    uint32_t depthUnit[N];
    if (ztst >= 2u || storesDepth)
    {
        for (int i = 0; i < N; ++i)
            depthUnit[i] = unitIn(depth, depthRow, x0 + i);
    }
    if (ztst >= 2u)
    {
        uint32_t stored[N];
        switch (depthKind)
        {
        case DepthKind::Z32:
            for (int i = 0; i < N; ++i)
                stored[i] = read32(vram, depthUnit[i]);
            break;
        case DepthKind::Z24:
            for (int i = 0; i < N; ++i)
                stored[i] = read32(vram, depthUnit[i]) & 0x00FFFFFFu;
            break;
        case DepthKind::Z16:
            for (int i = 0; i < N; ++i)
                stored[i] = read16(vram, depthUnit[i]);
            break;
        }
        if (ztst == 2u)
        {
            for (int i = 0; i < N; ++i)
            {
                const int32_t pass = in.z[i] >= stored[i] ? -1 : 0;
                frameMask[i] &= pass;
                depthMask[i] &= pass;
            }
        }
        else
        {
            for (int i = 0; i < N; ++i)
            {
                const int32_t pass = in.z[i] > stored[i] ? -1 : 0;
                frameMask[i] &= pass;
                depthMask[i] &= pass;
            }
        }
    }

    int32_t any = 0;
    for (int i = 0; i < N; ++i)
        any |= frameMask[i] | (storesDepth ? depthMask[i] : 0);
    if (any == 0)
        return;

    if (abe)
    {
        const int alphaSource = csel == 0u ? -1 : 0;
        const int alphaDest = csel == 1u ? -1 : 0;
        const int alphaFixed = csel >= 2u ? fix : 0;
        const int as = maskAS, ad = maskAD, bs = maskBS, bd = maskBD, ds = maskDS, dd = maskDD;
        const int perPixel = pabe ? -1 : 0;
        for (int i = 0; i < N; ++i)
        {
            const int dr = static_cast<int>(dest[i] & 0xFFu);
            const int dg = static_cast<int>((dest[i] >> 8) & 0xFFu);
            const int db = static_cast<int>((dest[i] >> 16) & 0xFFu);
            const int da = static_cast<int>(dest[i] >> 24);
            const int alpha = (a[i] & alphaSource) | (da & alphaDest) | alphaFixed;
            // ((A - B) * alpha >> 7) + D with A, B, D each source, destination or zero.
            const int br = (((((r[i] & as) | (dr & ad)) - ((r[i] & bs) | (dr & bd))) * alpha) >> 7) + ((r[i] & ds) | (dr & dd));
            const int bg = (((((g[i] & as) | (dg & ad)) - ((g[i] & bs) | (dg & bd))) * alpha) >> 7) + ((g[i] & ds) | (dg & dd));
            const int bb = (((((b[i] & as) | (db & ad)) - ((b[i] & bs) | (db & bd))) * alpha) >> 7) + ((b[i] & ds) | (db & dd));
            // PABE blends only pixels whose alpha has its top bit set.
            const int blended = ((a[i] & 0x80) != 0 ? -1 : 0) | ~perPixel;
            r[i] = (std::min(std::max(br, 0), 255) & blended) | (r[i] & ~blended);
            g[i] = (std::min(std::max(bg, 0), 255) & blended) | (g[i] & ~blended);
            b[i] = (std::min(std::max(bb, 0), 255) & blended) | (b[i] & ~blended);
        }
    }

    uint32_t pixel[N];
    const int alphaOr = (fba && framePsm != GS_PSM_CT24) ? 0x80 : 0;
    for (int i = 0; i < N; ++i)
    {
        const uint32_t packed = static_cast<uint32_t>(r[i] & 0xFF) | (static_cast<uint32_t>(g[i] & 0xFF) << 8) |
                                (static_cast<uint32_t>(b[i] & 0xFF) << 16) | (static_cast<uint32_t>((a[i] | alphaOr) & 0xFF) << 24);
        pixel[i] = (packed & ~keep[i]) | (dest[i] & keep[i]);
    }

    switch (frameKind)
    {
    case FrameKind::C32:
        for (int i = 0; i < count; ++i)
            if (frameMask[i])
                write32(vram, frameUnit[i], pixel[i]);
        break;
    case FrameKind::C24:
        for (int i = 0; i < count; ++i)
            if (frameMask[i])
                write32(vram, frameUnit[i], (read32(vram, frameUnit[i]) & 0xFF000000u) | (pixel[i] & 0x00FFFFFFu));
        break;
    case FrameKind::C16:
        for (int i = 0; i < count; ++i)
            if (frameMask[i])
                write16(vram, frameUnit[i], rgba8888To5551(pixel[i]));
        break;
    }

    if (storesDepth)
    {
        switch (depthKind)
        {
        case DepthKind::Z32:
            for (int i = 0; i < count; ++i)
                if (depthMask[i])
                    write32(vram, depthUnit[i], in.z[i]);
            break;
        case DepthKind::Z24:
            for (int i = 0; i < count; ++i)
                if (depthMask[i])
                    write32(vram, depthUnit[i], (read32(vram, depthUnit[i]) & 0xFF000000u) | (in.z[i] & 0x00FFFFFFu));
            break;
        case DepthKind::Z16:
            for (int i = 0; i < count; ++i)
                if (depthMask[i])
                    write16(vram, depthUnit[i], in.z[i]);
            break;
        }
    }
}

[[gnu::noinline]] uint64_t GSCpuBackend::HashPages(const std::vector<uint32_t> &pages) const
{
    uint64_t hash = 0xCBF29CE484222325ull;
    for (const uint32_t page : pages)
        hash = hashBytes(m_vram + static_cast<size_t>(page) * GSMem::GS_PAGE_SIZE, GSMem::GS_PAGE_SIZE, hash);
    return hash;
}

// A hash of the texels just right of and below an image of imageW x imageH at
// the texture's base, out to width x height. Kept until those pages change.
uint64_t GSCpuBackend::BorderHash(FastPipeline &p, uint64_t conversion, int imageW, int imageH, int width, int height)
{
    uint64_t key = mixKey(conversion, (static_cast<uint64_t>(p.tbp0) << 32) | (static_cast<uint64_t>(p.tbw) << 16) | p.texPsm);
    key = mixKey(key, (static_cast<uint64_t>(p.texW) << 48) | (static_cast<uint64_t>(p.texH) << 32) |
                          (static_cast<uint64_t>(imageW) << 16) | static_cast<uint64_t>(imageH));
    if (m_borderHashes.size() > 4096u)
        m_borderHashes.clear();
    BorderRecord &border = m_borderHashes[key];
    const bool right = width > imageW, below = height > imageH;
    if (border.serial != 0u &&
        !(right && PagesWrittenSince(p.texture, imageW, 0, imageW, height - 1, border.serial)) &&
        !(below && PagesWrittenSince(p.texture, 0, imageH, width - 1, imageH, border.serial)))
    {
        return border.hash;
    }
    if (right)
        FlushPrimitivesTouching(p.texture, imageW, 0, imageW, height - 1, false, "texture");
    if (below)
        FlushPrimitivesTouching(p.texture, 0, imageH, width - 1, imageH, false, "texture");
    uint64_t hash = 0x452821E638D01377ull;
    if (right)
        for (int v = 0; v < height; ++v)
            hash = mixKey(hash, p.fetch(imageW, v));
    if (below)
        for (int u = 0; u < width; ++u)
            hash = mixKey(hash, p.fetch(u, imageH));
    border.hash = hash;
    border.serial = m_writeSerial;
    return hash;
}

[[gnu::noinline]] const GSCpuBackend::DecodedTexture *GSCpuBackend::DecodedTextureFor(FastPipeline &p)
{
    // The VRAM pages the texture occupies.
    const uint32_t lastColumn = static_cast<uint32_t>(p.texW - 1) >> p.texture.pageWidthShift;
    const uint32_t lastRow = static_cast<uint32_t>(p.texH - 1) >> p.texture.pageHeightShift;
    const uint32_t spill = p.texture.spillsIntoNextPage ? 1u : 0u;
    const uint32_t firstPage = p.texture.basePage;

    // A texture that shares pages with the pixels being drawn changes under
    // every primitive; a copy would be redecoded each time.
    {
        PageSet sampled{};
        PageSet target{};
        addPages(sampled, p.texture, p.sampledU0, p.sampledV0, p.sampledU1, p.sampledV1);
        addPages(target, p.frame, p.boundsX0, p.boundsY0, p.boundsX1, p.boundsY1);
        if (!p.zmask)
            addPages(target, p.depth, p.boundsX0, p.boundsY0, p.boundsX1, p.boundsY1);
        if (intersects(sampled, target))
        {
            p.textureIsTarget = true;
            return nullptr;
        }
    }

    // What turns the stored texels into colours: the CLUT entries the texture
    // can reach (as raw words) and TEXA.
    uint64_t conversion = 0x243F6A8885A308D3ull;
    if (p.texKind >= TextureKind::Index8)
    {
        const bool fourBit = p.texKind >= TextureKind::Index4;
        const bool sixteenBit = p.cpsm == GS_PSM_CT16 || p.cpsm == GS_PSM_CT16S;
        uint64_t clutHash = (static_cast<uint64_t>(p.cpsm) << 16) | (static_cast<uint64_t>(p.csm) << 8) | p.csa;
        if (fourBit)
        {
            const uint32_t base = (static_cast<uint32_t>(p.csa) & (sixteenBit ? 0x1Fu : 0x0Fu)) << 4;
            clutHash = hashBytes(reinterpret_cast<const uint8_t *>(m_clut.data() + (base & 0x1F0u)), 16u * sizeof(uint16_t), clutHash);
            if (!sixteenBit)
                clutHash = hashBytes(reinterpret_cast<const uint8_t *>(m_clut.data() + 256u + (base & 0xF0u)), 16u * sizeof(uint16_t), clutHash);
        }
        else
        {
            if (m_clutHashLoad != m_clutLoads)
            {
                m_clutHash = hashBytes(reinterpret_cast<const uint8_t *>(m_clut.data()), m_clut.size() * sizeof(uint16_t), 0x13198A2E03707344ull);
                m_clutHashLoad = m_clutLoads;
            }
            clutHash = mixKey(clutHash, m_clutHash);
        }
        conversion = mixKey(conversion, clutHash);
    }
    if (p.texKind != TextureKind::Direct32)
    {
        conversion = mixKey(conversion, (static_cast<uint64_t>(p.texa.ta0) << 16) | (static_cast<uint64_t>(p.texa.ta1) << 8) | (p.texa.aem ? 1u : 0u));
    }

    static const bool wantStats = []
    {
        const char *value = std::getenv("PS2X_GS_STATS");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    static uint64_t statLookups = 0, statFixed = 0, statFixedDecodes = 0, statNoRecord = 0, statOutside = 0, statStale = 0, statRehash = 0, statRedecode = 0;
    if (wantStats && ++statLookups % 2000000u == 0u)
        std::fprintf(stderr, "[gs-textures] %llu lookups: %llu by upload (%llu decoded), by location: %llu no upload, %llu outside it, %llu overwritten; %llu rehashed, %llu decoded again; %zu copies, %llu MB\n",
                     (unsigned long long)statLookups, (unsigned long long)statFixed, (unsigned long long)statFixedDecodes,
                     (unsigned long long)statNoRecord, (unsigned long long)statOutside, (unsigned long long)statStale,
                     (unsigned long long)statRehash, (unsigned long long)statRedecode, m_decodedTextures.size(),
                     (unsigned long long)(m_decodedBytes >> 20));

    // A texture that is exactly what one transfer wrote, and still there, is
    // known by the hash of that image: the same image sent again, anywhere,
    // is the same copy, and nothing has to be read back from VRAM to tell.
    // It holds only the transferred rectangle, so it serves a primitive that
    // samples nothing outside it.
    {
        const auto found = m_uploadAt.find((static_cast<uint64_t>(p.tbp0) << 32) | (static_cast<uint64_t>(p.tbw) << 8) | p.texPsm);
        if (found == m_uploadAt.end())
            ++statNoRecord;
        if (found != m_uploadAt.end())
        {
            UploadRecord &record = found->second;
            const int imageW = std::min(static_cast<int>((record.rect >> 16) & 0xFFFFu), p.texW);
            const int imageH = std::min(static_cast<int>(record.rect & 0xFFFFu), p.texH);
            // Filtering at the image's last row or column reads one texel
            // beyond it. The copy has that border too, as VRAM has it.
            const int width = std::min(imageW + 1, p.texW);
            const int height = std::min(imageH + 1, p.texH);
            const bool inside = p.sampledU0 >= 0 && p.sampledV0 >= 0 && p.sampledU1 < width && p.sampledV1 < height;
            // Its blocks are looked at again only after another transfer.
            bool intact = inside && !DrawnSince(p.texture, 0, 0, imageW - 1, imageH - 1, record.serial);
            if (intact && record.checkedUpload != m_lastUploadSerial)
            {
                intact = !UploadedSince(p.texture, 0, 0, imageW - 1, imageH - 1, record.serial);
                if (intact)
                    record.checkedUpload = m_lastUploadSerial;
            }
            if (!inside)
                ++statOutside;
            else if (!intact)
                ++statStale;
            if (intact)
            {
                ++statFixed;
                uint64_t key = mixKey(conversion, record.hash);
                key = mixKey(key, (static_cast<uint64_t>(p.texPsm) << 48) | (static_cast<uint64_t>(p.texW) << 32) |
                                      (static_cast<uint64_t>(imageW) << 16) | static_cast<uint64_t>(imageH));
                // The border is not part of the image, so a primitive that
                // reaches it needs a copy made with the border as it is now.
                if (p.sampledU1 >= imageW || p.sampledV1 >= imageH)
                    key = mixKey(key, BorderHash(p, conversion, imageW, imageH, width, height));
                const auto [it, inserted] = m_decodedTextures.try_emplace(key);
                DecodedTexture &copy = it->second;
                copy.lastUsed = ++m_decodedUseClock;
                if (inserted)
                {
                    ++statFixedDecodes;
                    // The image is in VRAM (nothing queued writes it, or it
                    // would not be intact); its border may still be drawn to.
                    FlushPrimitivesTouching(p.texture, 0, 0, width - 1, height - 1, false, "texture");
                    copy.immutable = true;
                    copy.width = p.texW;
                    copy.height = height;
                    copy.pixels.assign(static_cast<size_t>(p.texW) * static_cast<size_t>(height), 0u);
                    for (int v = 0; v < height; ++v)
                    {
                        uint32_t *out = copy.pixels.data() + static_cast<size_t>(v) * static_cast<size_t>(p.texW);
                        for (int u = 0; u < width; ++u)
                            out[u] = p.fetch(u, v);
                    }
                    m_decodedBytes += copy.pixels.size() * sizeof(uint32_t);
                    ForgetOldDecodedTextures();
                }
                return &copy;
            }
        }
    }

    // Anything else is known only by where it is. Colours stored as such are
    // as quick to read from VRAM as from a copy, and a texture nobody sent is
    // one that was drawn (a render target read back), which a copy would only
    // chase. Indexed texels are worth converting once, while they stay put.
    if (p.texKind < TextureKind::Index8)
        return nullptr;

    uint64_t key = mixKey(conversion, (static_cast<uint64_t>(p.tbp0) << 32) | (static_cast<uint64_t>(p.tbw) << 16) | p.texPsm);
    key = mixKey(key, (static_cast<uint64_t>(p.texW) << 32) | static_cast<uint64_t>(p.texH));

    // In texture lookups: how soon a second content change marks a texture as
    // volatile, and for how long it is then read from VRAM.
    constexpr uint64_t kVolatileWindow = 30000u;
    constexpr uint64_t kVolatileHold = 300000u;

    const auto [it, inserted] = m_decodedTextures.try_emplace(key);
    DecodedTexture &copy = it->second;
    copy.lastUsed = ++m_decodedUseClock;
    if (copy.directUntil > m_decodedUseClock)
    {
        copy.checkedSerial = m_writeSerial;
        return nullptr;
    }

    // From here VRAM is read: after whatever is still to be drawn into it.
    FlushPrimitivesTouching(p.texture, 0, 0, p.texW - 1, p.texH - 1, false, "texture");

    bool decode = inserted;
    if (inserted)
    {
        copy.width = p.texW;
        copy.height = p.texH;
        for (uint32_t row = 0; row <= lastRow; ++row)
            for (uint32_t column = 0; column <= lastColumn + spill; ++column)
                copy.pages.push_back((firstPage + row * p.texture.pagesPerRow + column) % kVramPages);
        std::sort(copy.pages.begin(), copy.pages.end());
        copy.pages.erase(std::unique(copy.pages.begin(), copy.pages.end()), copy.pages.end());
    }
    else if (copy.checkedSerial != m_writeSerial)
    {
        bool written = false;
        for (const uint32_t page : copy.pages)
        {
            if (m_pageSerial[page] > copy.checkedSerial)
            {
                written = true;
                break;
            }
        }
        // Rewritten with the same bytes is still the same texture.
        if (written)
            ++statRehash;
        if (written && HashPages(copy.pages) != copy.contentHash)
        {
            decode = true;
            ++statRedecode;
        }
    }

    if (decode && !inserted)
    {
        // Queued primitives may be sampling the copy this replaces.
        FlushPrimitives();
        // Contents that keep changing: decoding all of it for the few texels
        // one use reads costs more than reading them from VRAM.
        const bool changedRecently = copy.lastDecoded + kVolatileWindow > m_decodedUseClock;
        copy.lastDecoded = m_decodedUseClock;
        if (changedRecently)
        {
            copy.directUntil = m_decodedUseClock + kVolatileHold;
            m_decodedBytes -= copy.pixels.size() * sizeof(uint32_t);
            copy.pixels.clear();
            copy.pixels.shrink_to_fit();
            copy.checkedSerial = m_writeSerial;
            return nullptr;
        }
    }
    if (copy.pixels.empty())
        decode = true;
    if (decode)
    {
        m_decodedBytes -= copy.pixels.size() * sizeof(uint32_t);
        copy.pixels.resize(static_cast<size_t>(p.texW) * static_cast<size_t>(p.texH));
        m_decodedBytes += copy.pixels.size() * sizeof(uint32_t);
        uint32_t *out = copy.pixels.data();
        for (int v = 0; v < p.texH; ++v)
            for (int u = 0; u < p.texW; ++u)
                *out++ = p.fetch(u, v);
        copy.contentHash = HashPages(copy.pages);
        if (inserted)
            ForgetOldDecodedTextures();
    }
    copy.checkedSerial = m_writeSerial;
    return &copy;
}

// Drops copies that have not been used for a while once they add up to a lot
// of memory. Queued primitives hold pointers into them, so those are drawn
// first.
void GSCpuBackend::ForgetOldDecodedTextures()
{
    constexpr uint64_t kBudgetBytes = 256ull << 20;
    if (m_decodedBytes <= kBudgetBytes)
        return;
    FlushPrimitives();
    // In texture lookups, one per primitive: a few seconds of drawing.
    for (uint64_t unusedFor = 2000000u; m_decodedBytes > kBudgetBytes / 2u && unusedFor >= 2000u; unusedFor /= 10u)
    {
        for (auto entry = m_decodedTextures.begin(); entry != m_decodedTextures.end();)
        {
            if (entry->second.lastUsed + unusedFor < m_decodedUseClock)
            {
                m_decodedBytes -= entry->second.pixels.size() * sizeof(uint32_t);
                entry = m_decodedTextures.erase(entry);
            }
            else
                ++entry;
        }
    }
}

bool GSCpuBackend::UseReferenceRasteriser()
{
    static const bool reference = []
    {
        const char *value = std::getenv("PS2X_GS_REFERENCE");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return reference;
}

// PS2X_GS_STATS=2: thread time and pixels per kind of primitive (which
// combination of pipeline features), to see which loops are worth specialising.
namespace
{
    struct RasterClass
    {
        std::atomic<uint32_t> key{0};
        std::atomic<uint64_t> nanoseconds{0}, pixels{0}, calls{0}, rows{0}, blocks{0};
    };
    constexpr size_t kRasterClassSlots = 1024u;
    RasterClass g_rasterClasses[kRasterClassSlots];

    bool rasterClassStatsWanted()
    {
        static const bool wanted = []
        {
            const char *value = std::getenv("PS2X_GS_STATS");
            return value != nullptr && std::atoi(value) >= 2;
        }();
        return wanted;
    }

    void addRasterClassSample(uint32_t key, uint64_t nanoseconds, uint64_t pixels, uint64_t rows, uint64_t blocks)
    {
        key |= 0x80000000u; // never zero, which marks a free slot
        for (size_t probe = 0, slot = (key * 0x9E3779B1u) >> 22; probe < kRasterClassSlots; ++probe, slot = (slot + 1u) % kRasterClassSlots)
        {
            RasterClass &entry = g_rasterClasses[slot];
            uint32_t seen = entry.key.load(std::memory_order_relaxed);
            if (seen == 0u && entry.key.compare_exchange_strong(seen, key, std::memory_order_relaxed))
                seen = key;
            if (seen != key)
                continue;
            entry.nanoseconds.fetch_add(nanoseconds, std::memory_order_relaxed);
            entry.pixels.fetch_add(pixels, std::memory_order_relaxed);
            entry.calls.fetch_add(1u, std::memory_order_relaxed);
            entry.rows.fetch_add(rows, std::memory_order_relaxed);
            entry.blocks.fetch_add(blocks, std::memory_order_relaxed);
            return;
        }
    }

    void printRasterClasses()
    {
        struct Line
        {
            uint32_t key;
            uint64_t nanoseconds, pixels, calls, rows, blocks;
        };
        std::vector<Line> lines;
        uint64_t total = 0;
        for (const RasterClass &entry : g_rasterClasses)
        {
            const uint32_t key = entry.key.load(std::memory_order_relaxed);
            if (key == 0u)
                continue;
            lines.push_back({key, entry.nanoseconds.load(), entry.pixels.load(), entry.calls.load(), entry.rows.load(), entry.blocks.load()});
            total += lines.back().nanoseconds;
        }
        std::sort(lines.begin(), lines.end(), [](const Line &a, const Line &b)
                  { return a.nanoseconds > b.nanoseconds; });
        std::fprintf(stderr, "[gs-classes] %.2fs of thread time in %zu classes\n", static_cast<double>(total) * 1e-9, lines.size());
        static const char *const kTexKinds[8] = {"32", "24", "16", "8", "8H", "4", "4HL", "4HH"};
        for (size_t i = 0; i < lines.size() && i < 24u; ++i)
        {
            const Line &line = lines[i];
            const uint32_t k = line.key;
            char text[160];
            std::snprintf(text, sizeof(text), "%s%s%s%s%s%s%s ztst=%u%s%s%s tfx=%u%s tex=%s frame=%u depth=%u%s%s",
                          (k & 1u) ? "sprite" : "tri", (k & 2u) ? " tme" : "", (k & 4u) ? " linear" : "", (k & 8u) ? " decoded" : "",
                          (k & 16u) ? " fge" : "", (k & 32u) ? " abe" : "", (k & 64u) ? " atest" : "", (k >> 7) & 3u,
                          (k & 0x200u) ? " zmask" : "", (k & 0x400u) ? " fst" : "", (k & 0x800u) ? " gouraud" : "", (k >> 12) & 3u,
                          (k & 0x4000u) ? " tcc" : "", kTexKinds[(k >> 15) & 7u], (k >> 18) & 3u, (k >> 20) & 3u,
                          (k & 0x400000u) ? " date" : "", (k & 0x800000u) ? " fbmsk" : "");
            std::fprintf(stderr, "[gs-classes] %5.1f%% %6.1f ns/px %8.2fM px %8llu calls %5.1f rows/call %5.1f px/row %4.1f px/block  %s\n",
                         100.0 * static_cast<double>(line.nanoseconds) / static_cast<double>(std::max<uint64_t>(total, 1u)),
                         static_cast<double>(line.nanoseconds) / static_cast<double>(std::max<uint64_t>(line.pixels, 1u)),
                         static_cast<double>(line.pixels) * 1e-6, (unsigned long long)line.calls,
                         static_cast<double>(line.rows) / static_cast<double>(std::max<uint64_t>(line.calls, 1u)),
                         static_cast<double>(line.pixels) / static_cast<double>(std::max<uint64_t>(line.rows, 1u)),
                         static_cast<double>(line.pixels) / static_cast<double>(std::max<uint64_t>(line.blocks, 1u)), text);
        }
    }
}

// A primitive ready to rasterise: its pipeline, its bounds on the target and
// the per-primitive constants of the walk.
struct GSCpuBackend::PreparedPrimitive
{
    FastPipeline pipeline;
    GSVertex v[3];
    bool sprite = false;
    bool gouraud = false;
    int minX = 0, minY = 0, maxX = -1, maxY = -1;

    // Triangle: barycentric planes measured from vertex 2.
    float fx2 = 0.0f, fy2 = 0.0f;
    float wa[3]{}, wb[3]{}, wc[3]{};
    float invWa[3]{}; // 1 / wa, zero for an edge that does not cross rows
    bool ownsEdge[3]{};
    // Each interpolated attribute as a plane: value at vertex 2, and its
    // change per pixel in x and in y.
    struct Plane
    {
        float base = 0.0f, dx = 0.0f, dy = 0.0f;
    };
    Plane red, green, blue, alpha, fog, texS, texT, texQ;
    double zBase = 0.0, zDx = 0.0, zDy = 0.0;

    // Sprite: its unclipped origin and size, and texel coordinates at the
    // corners.
    int originX = 0, originY = 0;
    float spanX = 1.0f, spanY = 1.0f;
    float tu0 = 0.0f, tv0 = 0.0f, tu1 = 0.0f, tv1 = 0.0f;

    bool empty() const { return maxX < minX || maxY < minY; }
};

// Primitives waiting to be drawn, and the pool that draws them.
//
// The GS thread collects primitives and hands them over a batch at a time; the
// pool draws the batches one after another, in the order they were handed
// over, while the GS thread carries on decoding. Within a batch rows are given
// out in bands, and within a band primitives are drawn in submission order, so
// the pixels are the ones drawing them one by one would give.
//
// What keeps that safe is knowing which VRAM pages the primitives not yet
// drawn write and read: anything else that reads or writes VRAM (a transfer, a
// CLUT load, decoding a texture, the display) first waits for the pool if its
// pages are among them.
struct GSCpuBackend::RasterQueue
{
    static constexpr int kBandHeight = 16;
    // Bounding-box pixels worth waking an idle pool for.
    static constexpr uint64_t kEagerPixels = 24000u;
    static constexpr size_t kMaxPending = 4096u;
    // Primitives the pool may have outstanding before the GS thread stops to
    // help, so a slow pool cannot fall behind without limit.
    static constexpr size_t kMaxInFlight = 24000u;

    // --- GS thread only -----------------------------------------------------
    std::vector<PreparedPrimitive> pending;
    uint64_t pendingPixels = 0;
    PageSet written{};  // pages pending primitives write
    PageSet read{};     // pages of textures they sample that could change under them
    PageSet readLive{}; // ...of those, pages sampled from VRAM while drawing
    // Batches are numbered as they are handed over. For every page, the last
    // batch that writes it and the last that reads it: once that one has been
    // retired, nothing outstanding touches the page.
    uint64_t submitted = 0;
    std::array<uint64_t, kPageCount> pageWriter{};
    std::array<uint64_t, kPageCount> pageReader{};
    // Target of the pending primitives.
    uint32_t fbp = 0, fbw = 0, zbp = 0;
    uint8_t fpsm = 0, zpsm = 0;

    // --- shared with the pool -----------------------------------------------
    // A batch is primitives to draw or, in their place, an image to store
    // (a host-to-local transfer): both write VRAM and keep their order.
    struct Batch
    {
        std::vector<PreparedPrimitive> primitives;
        bool image = false;
        uint8_t *vram = nullptr;
        GSMem::SurfaceLayout layout;
        uint8_t dpsm = 0;
        uint32_t streamBits = 0, dsax = 0, dsay = 0, rrw = 0, rrh = 0;
        std::vector<uint8_t> data;
        size_t weight = 0; // what it counts for against kMaxInFlight
        uint64_t number = 0;
        int firstRow = 0;
        int bandCount = 0;
        std::atomic<int> nextBand{0};
        int drawing = 0; // threads inside drawBands (under the mutex)
    };
    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable wake; // the front batch has bands to take
    std::condition_variable done; // a batch was finished
    std::deque<Batch *> flight;   // handed over, in order; only the front one is drawn
    std::vector<Batch *> spare;
    size_t flightPrimitives = 0;
    std::atomic<uint64_t> retired{0}; // number of the last batch drawn completely
    bool quit = false;

    static void drawBands(Batch &batch)
    {
        for (;;)
        {
            const int band = batch.nextBand.fetch_add(1, std::memory_order_relaxed);
            if (band >= batch.bandCount)
                return;
            const int bandFirst = batch.firstRow + band * kBandHeight;
            const int bandLast = bandFirst + kBandHeight - 1;
            if (batch.image)
            {
                // firstRow is a multiple of the band height: see StoreImagePixels.
                const int y0 = std::max(bandFirst, static_cast<int>(batch.dsay));
                const int y1 = std::min(bandLast, static_cast<int>(batch.dsay + batch.rrh) - 1);
                if (y0 <= y1)
                {
                    StoreImagePixels(batch.vram, batch.layout, batch.dpsm, batch.streamBits, batch.dsax, batch.dsay, batch.rrw,
                                     (static_cast<uint32_t>(y0) - batch.dsay) * batch.rrw,
                                     (static_cast<uint32_t>(y1) - batch.dsay + 1u) * batch.rrw, batch.data.data(), batch.data.size(), 0u);
                }
                continue;
            }
            for (const PreparedPrimitive &primitive : batch.primitives)
            {
                const int y0 = std::max(primitive.minY, bandFirst);
                const int y1 = std::min(primitive.maxY, bandLast);
                if (y0 <= y1)
                    Rasterise(primitive, y0, y1);
            }
        }
    }

    // Draws bands of the front batch until none are left to take, and retires
    // the batch if this was the last thread in it. Takes and returns the lock
    // held. False when there was no band to take.
    bool drawFront(std::unique_lock<std::mutex> &lock)
    {
        if (flight.empty())
            return false;
        Batch *batch = flight.front();
        if (batch->nextBand.load(std::memory_order_relaxed) >= batch->bandCount)
            return false;
        ++batch->drawing;
        lock.unlock();
        drawBands(*batch);
        lock.lock();
        if (--batch->drawing == 0)
        {
            flight.pop_front();
            flightPrimitives -= batch->weight;
            batch->primitives.clear();
            batch->data.clear();
            spare.push_back(batch);
            retired.store(batch->number, std::memory_order_release);
            done.notify_all();
            if (!flight.empty())
                wake.notify_all();
        }
        return true;
    }

    void workerLoop()
    {
        std::unique_lock<std::mutex> lock(mutex);
        while (!quit)
        {
            if (!drawFront(lock))
                wake.wait(lock);
        }
    }

    void startWorkers()
    {
        if (!workers.empty())
            return;
        // Leave cores for the EE, VU1, the GS thread and the window;
        // efficiency cores would only hold a band up.
        unsigned count = std::thread::hardware_concurrency();
        count = count > 3u ? std::min(count - 3u, 6u) : 0u;
        if (const char *value = std::getenv("PS2X_GS_THREADS"))
            count = static_cast<unsigned>(std::max(0, std::atoi(value) - 1));
        for (unsigned i = 0; i < count; ++i)
            workers.emplace_back([this]
                                 { workerLoop(); });
    }

    bool busy() const
    {
        return retired.load(std::memory_order_acquire) != submitted;
    }

    // The last batch handed over that touches any of `pages`.
    static uint64_t lastBatch(const PageSet &pages, const std::array<uint64_t, kPageCount> &lastUse)
    {
        uint64_t last = 0;
        for (size_t word = 0; word < pages.size(); ++word)
        {
            for (uint64_t bits = pages[word]; bits != 0u; bits &= bits - 1u)
                last = std::max(last, lastUse[word * 64u + static_cast<size_t>(__builtin_ctzll(bits))]);
        }
        return last;
    }

    // PS2X_GS_STATS=1: how the queue is being used, every few thousand batches.
    struct Stats
    {
        uint64_t batches = 0, primitives = 0, waits = 0, images = 0, imageBytes = 0;
        double waitSeconds = 0.0;
        uint64_t reason[8]{};
        uint64_t waitsFor[8]{};
        double waitSecondsFor[8]{};
    } stats;

    static int reasonSlot(const char *why)
    {
        static const char *const kReasons[8] = {"target", "texture", "direct", "full", "upload", "idle", "present", "other"};
        if (why)
            for (int i = 0; i < 7; ++i)
                if (std::strcmp(why, kReasons[i]) == 0)
                    return i;
        return 7;
    }

    static bool wantStats()
    {
        static const bool wanted = []
        {
            const char *value = std::getenv("PS2X_GS_STATS");
            return value != nullptr && value[0] != '\0' && value[0] != '0';
        }();
        return wanted;
    }

    // Hands the pending primitives to the pool. Does not wait for them.
    void submit(const char *why = nullptr)
    {
        if (pending.empty())
            return;

        if (wantStats())
        {
            static const char *const kReasons[8] = {"target", "texture", "direct", "full", "upload", "idle", "present", "other"};
            const char *reason = why ? why : "other";
            int slot = 7;
            for (int i = 0; i < 7; ++i)
                if (std::strcmp(reason, kReasons[i]) == 0)
                    slot = i;
            ++stats.reason[slot];
            ++stats.batches;
            stats.primitives += pending.size();
            if (stats.batches % 4000u == 0u)
            {
                std::fprintf(stderr, "[gs-queue] batches=%llu prims/batch=%.1f; handed over for",
                             (unsigned long long)stats.batches,
                             static_cast<double>(stats.primitives) / static_cast<double>(stats.batches));
                for (int i = 0; i < 8; ++i)
                    std::fprintf(stderr, " %s=%llu", kReasons[i], (unsigned long long)stats.reason[i]);
                std::fprintf(stderr, "; %llu images (%llu MB) stored by the pool; the GS thread waited for the pool %llu times, %.2fs:",
                             (unsigned long long)stats.images, (unsigned long long)(stats.imageBytes >> 20),
                             (unsigned long long)stats.waits, stats.waitSeconds);
                for (int i = 0; i < 8; ++i)
                    if (stats.waitsFor[i] != 0u)
                        std::fprintf(stderr, " %s %llu/%.2fs", kReasons[i], (unsigned long long)stats.waitsFor[i], stats.waitSecondsFor[i]);
                std::fprintf(stderr, "\n");
                if (rasterClassStatsWanted() && stats.batches % 16000u == 0u)
                    printRasterClasses();
            }
        }

        int top = pending.front().minY;
        int bottom = pending.front().maxY;
        for (const PreparedPrimitive &primitive : pending)
        {
            top = std::min(top, primitive.minY);
            bottom = std::max(bottom, primitive.maxY);
        }

        startWorkers();
        Batch *batch = nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!spare.empty())
            {
                batch = spare.back();
                spare.pop_back();
            }
        }
        if (!batch)
            batch = new Batch();
        batch->primitives.swap(pending);
        batch->image = false;
        batch->weight = batch->primitives.size();
        batch->number = ++submitted;
        batch->firstRow = top;
        batch->bandCount = (bottom - top) / kBandHeight + 1;
        batch->nextBand.store(0, std::memory_order_relaxed);
        batch->drawing = 0;

        for (size_t word = 0; word < written.size(); ++word)
        {
            for (uint64_t bits = written[word]; bits != 0u; bits &= bits - 1u)
                pageWriter[word * 64u + static_cast<size_t>(__builtin_ctzll(bits))] = submitted;
            for (uint64_t bits = read[word]; bits != 0u; bits &= bits - 1u)
                pageReader[word * 64u + static_cast<size_t>(__builtin_ctzll(bits))] = submitted;
        }
        pendingPixels = 0;
        written.fill(0u);
        read.fill(0u);
        readLive.fill(0u);

        size_t outstanding = 0;
        {
            std::lock_guard<std::mutex> lock(mutex);
            flight.push_back(batch);
            flightPrimitives += batch->weight;
            outstanding = flightPrimitives;
        }
        wake.notify_all();

        if (workers.empty())
            waitIdle();
        else if (outstanding > kMaxInFlight)
            waitUntil(kMaxInFlight / 2u);
    }

    // Hands the pool an image to store in the pages `pages`, after everything
    // handed over so far. Does not wait for it.
    void submitImage(uint8_t *vram, const GSMem::SurfaceLayout &layout, uint8_t dpsm, uint32_t streamBits, uint32_t dsax,
                     uint32_t dsay, uint32_t rrw, uint32_t rrh, const uint8_t *data, size_t bytes, const PageSet &pages)
    {
        startWorkers();
        Batch *batch = nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!spare.empty())
            {
                batch = spare.back();
                spare.pop_back();
            }
        }
        if (!batch)
            batch = new Batch();
        batch->image = true;
        batch->vram = vram;
        batch->layout = layout;
        batch->dpsm = dpsm;
        batch->streamBits = streamBits;
        batch->dsax = dsax;
        batch->dsay = dsay;
        batch->rrw = rrw;
        batch->rrh = rrh;
        batch->data.assign(data, data + bytes);
        batch->weight = bytes / 256u + 1u;
        batch->number = ++submitted;
        batch->firstRow = static_cast<int>(dsay - dsay % static_cast<uint32_t>(kBandHeight));
        batch->bandCount = (static_cast<int>(dsay + rrh) - 1 - batch->firstRow) / kBandHeight + 1;
        batch->nextBand.store(0, std::memory_order_relaxed);
        batch->drawing = 0;
        for (size_t word = 0; word < pages.size(); ++word)
        {
            for (uint64_t bits = pages[word]; bits != 0u; bits &= bits - 1u)
                pageWriter[word * 64u + static_cast<size_t>(__builtin_ctzll(bits))] = submitted;
        }
        if (wantStats())
        {
            ++stats.images;
            stats.imageBytes += bytes;
        }

        size_t outstanding = 0;
        {
            std::lock_guard<std::mutex> lock(mutex);
            flight.push_back(batch);
            flightPrimitives += batch->weight;
            outstanding = flightPrimitives;
        }
        // One thread is enough for a few bands.
        if (batch->bandCount <= 8)
            wake.notify_one();
        else
            wake.notify_all();

        if (workers.empty())
            waitIdle();
        else if (outstanding > kMaxInFlight)
            waitUntil(kMaxInFlight / 2u);
    }

    // Helps the pool until it has at most `limit` primitives left.
    void waitUntil(size_t limit)
    {
        const bool timed = wantStats();
        const auto start = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        {
            std::unique_lock<std::mutex> lock(mutex);
            while (flightPrimitives > limit)
            {
                if (!drawFront(lock))
                    done.wait(lock);
            }
        }
        if (timed)
        {
            ++stats.waits;
            stats.waitSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        }
    }

    // Helps the pool until batch `number` has been drawn. Uses nothing that
    // belongs to the GS thread, so another thread may call it (with the number
    // it was given under the backend lock).
    void waitFor(uint64_t number)
    {
        std::unique_lock<std::mutex> lock(mutex);
        while (retired.load(std::memory_order_acquire) < number)
        {
            if (!drawFront(lock))
                done.wait(lock);
        }
    }

    // As waitFor, on the GS thread, counted in the statistics.
    void waitForCounted(uint64_t number, const char *why = nullptr)
    {
        if (retired.load(std::memory_order_acquire) >= number)
            return;
        const bool timed = wantStats();
        const auto start = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        waitFor(number);
        if (timed)
        {
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            ++stats.waits;
            stats.waitSeconds += seconds;
            const int slot = reasonSlot(why);
            ++stats.waitsFor[slot];
            stats.waitSecondsFor[slot] += seconds;
        }
    }

    // Returns once everything handed over has been drawn.
    void waitIdle(const char *why = nullptr)
    {
        waitForCounted(submitted, why);
    }

    // Hands over and waits: everything queued is in VRAM on return.
    void run(const char *why = nullptr)
    {
        submit(why);
        waitIdle(why);
    }

    ~RasterQueue()
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            quit = true;
        }
        wake.notify_all();
        for (std::thread &worker : workers)
            worker.join();
        for (Batch *batch : flight)
            delete batch;
        for (Batch *batch : spare)
            delete batch;
    }
};

void GSCpuBackend::RasterQueueDeleter::operator()(RasterQueue *queue) const
{
    delete queue;
}

bool GSCpuBackend::HasQueuedPrimitives() const
{
    return m_rasterQueue && (!m_rasterQueue->pending.empty() || m_rasterQueue->busy());
}

void GSCpuBackend::FlushPrimitives()
{
    if (m_rasterQueue)
        m_rasterQueue->run();
}

// Hands the pending primitives to the pool without waiting for them.
void GSCpuBackend::SubmitPrimitives()
{
    if (m_rasterQueue)
        m_rasterQueue->submit();
}

// Before reading (or, with `writes`, writing) those pixels of a surface in
// VRAM: waits for queued primitives that write them and, for a writer, for
// those that still have to read them.
void GSCpuBackend::FlushPrimitivesTouching(const GSMem::SurfaceLayout &layout, int x0, int y0, int x1, int y1, bool writes,
                                           const char *why)
{
    if (const uint64_t batch = LastBatchTouching(layout, x0, y0, x1, y1, writes, why))
        m_rasterQueue->waitForCounted(batch, why);
}

// Hands over what is pending if it touches those pixels, and returns the
// number of the last outstanding batch that does (0 if none): once that batch
// has been drawn the pixels are final.
uint64_t GSCpuBackend::LastBatchTouching(const GSMem::SurfaceLayout &layout, int x0, int y0, int x1, int y1, bool writes,
                                         const char *why)
{
    if (!m_rasterQueue || !layout.table)
        return 0u;
    RasterQueue &queue = *m_rasterQueue;
    if (queue.pending.empty() && !queue.busy())
        return 0u;
    PageSet pages{};
    addPages(pages, layout, x0, y0, x1, y1);
    if (!queue.pending.empty() && (intersects(pages, queue.written) || (writes && intersects(pages, queue.read))))
        queue.submit(why);
    uint64_t last = RasterQueue::lastBatch(pages, queue.pageWriter);
    if (writes)
        last = std::max(last, RasterQueue::lastBatch(pages, queue.pageReader));
    return last > queue.retired.load(std::memory_order_acquire) ? last : 0u;
}

void GSCpuBackend::WaitForBatch(uint64_t batch)
{
    if (m_rasterQueue && batch != 0u)
        m_rasterQueue->waitFor(batch);
}

// Hands a whole host-to-local transfer to the pool, to be stored after what
// has been handed over so far.
void GSCpuBackend::QueueImage(const GSMem::SurfaceLayout &layout, uint8_t dpsm, uint32_t streamBits, uint32_t dsax, uint32_t dsay,
                              uint32_t rrw, uint32_t rrh, const uint8_t *data, size_t bytes)
{
    if (!m_rasterQueue)
        m_rasterQueue.reset(new RasterQueue());
    RasterQueue &queue = *m_rasterQueue;
    PageSet pages{};
    addPages(pages, layout, static_cast<int>(dsax), static_cast<int>(dsay), static_cast<int>(dsax + rrw) - 1,
             static_cast<int>(dsay + rrh) - 1);
    // Pending primitives that write or sample these pages come first.
    if (!queue.pending.empty() && (intersects(pages, queue.written) || intersects(pages, queue.read)))
        queue.submit("upload");
    queue.submitImage(m_vram, layout, dpsm, streamBits, dsax, dsay, rrw, rrh, data, bytes, pages);
}

// Before queueing a primitive that samples those texels while it is drawn:
// what still has to write them goes to the pool first, so it is drawn first.
void GSCpuBackend::SubmitPrimitivesWriting(const GSMem::SurfaceLayout &layout, int x0, int y0, int x1, int y1, const char *why)
{
    if (!m_rasterQueue || m_rasterQueue->pending.empty() || !layout.table)
        return;
    PageSet pages{};
    addPages(pages, layout, x0, y0, x1, y1);
    if (intersects(pages, m_rasterQueue->written))
        m_rasterQueue->submit(why);
}

// Sets `primitive` up for the triangle in `batch`. Returns false when the fast
// path cannot draw it; a primitive that covers nothing comes back empty.
[[gnu::noinline]] bool GSCpuBackend::PrepareTriangle(const GSPrimitiveBatch &batch, PreparedPrimitive &primitive)
{
    const GSDrawState &state = batch.state;
    FastPipeline &p = primitive.pipeline;
    if (!p.setup(*this, state))
        return false;

    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];
    const GSVertex &v2 = batch.vertices[2];
    const GSContext &ctx = state.context;

    const float ofx = static_cast<float>(ctx.xyoffset.ofx >> 4);
    const float ofy = static_cast<float>(ctx.xyoffset.ofy >> 4);
    const float fx0 = v0.x - ofx, fy0 = v0.y - ofy;
    const float fx1 = v1.x - ofx, fy1 = v1.y - ofy;
    const float fx2 = v2.x - ofx, fy2 = v2.y - ofy;

    const float denom = (fy1 - fy2) * (fx0 - fx2) + (fx2 - fx1) * (fy0 - fy2);
    if (std::fabs(denom) < 0.001f)
        return true;
    const float invDenom = 1.0f / denom;

    int minX = static_cast<int>(std::floor(std::min({fx0, fx1, fx2})));
    int maxX = static_cast<int>(std::ceil(std::max({fx0, fx1, fx2})));
    int minY = static_cast<int>(std::floor(std::min({fy0, fy1, fy2})));
    int maxY = static_cast<int>(std::ceil(std::max({fy0, fy1, fy2})));
    primitive.minX = clampInt(minX, p.scissorX0, p.scissorX1);
    primitive.maxX = clampInt(maxX, p.scissorX0, p.scissorX1);
    primitive.minY = clampInt(minY, p.scissorY0, p.scissorY1);
    primitive.maxY = clampInt(maxY, p.scissorY0, p.scissorY1);

    primitive.sprite = false;
    primitive.gouraud = state.prim.iip;
    primitive.v[0] = v0;
    primitive.v[1] = v1;
    primitive.v[2] = v2;
    primitive.fx2 = fx2;
    primitive.fy2 = fy2;

    // Barycentric weights as planes over the screen, measured from vertex 2
    // (differences first, as the reference does, to keep float precision):
    // w_i = wa[i] * (px - fx2) + wb[i] * (py - fy2) + wc[i].
    float *wa = primitive.wa, *wb = primitive.wb, *wc = primitive.wc;
    wa[0] = (fy1 - fy2) * invDenom;
    wb[0] = (fx2 - fx1) * invDenom;
    wc[0] = 0.0f;
    wa[1] = (fy2 - fy0) * invDenom;
    wb[1] = (fx0 - fx2) * invDenom;
    wc[1] = 0.0f;
    wa[2] = -wa[0] - wa[1];
    wb[2] = -wb[0] - wb[1];
    wc[2] = 1.0f;

    // A centre exactly on an edge counts for left edges (inside is towards +x)
    // and top edges (horizontal, inside is towards +y) only.
    for (int i = 0; i < 3; ++i)
    {
        primitive.ownsEdge[i] = wa[i] > 0.0f || (wa[i] == 0.0f && wb[i] > 0.0f);
        primitive.invWa[i] = wa[i] != 0.0f ? 1.0f / wa[i] : 0.0f;
    }

    // value = a0 * w0 + a1 * w1 + a2 * w2 with w2 = 1 - w0 - w1 is a plane.
    auto plane = [&](float a0, float a1, float a2)
    {
        PreparedPrimitive::Plane result;
        result.base = a2;
        result.dx = (a0 - a2) * wa[0] + (a1 - a2) * wa[1];
        result.dy = (a0 - a2) * wb[0] + (a1 - a2) * wb[1];
        return result;
    };
    primitive.red = plane(v0.r, v1.r, v2.r);
    primitive.green = plane(v0.g, v1.g, v2.g);
    primitive.blue = plane(v0.b, v1.b, v2.b);
    primitive.alpha = plane(v0.a, v1.a, v2.a);
    primitive.fog = plane(v0.fog, v1.fog, v2.fog);
    if (state.prim.fst)
    {
        primitive.texS = plane(v0.u, v1.u, v2.u);
        primitive.texT = plane(v0.v, v1.v, v2.v);
    }
    else
    {
        primitive.texS = plane(v0.s, v1.s, v2.s);
        primitive.texT = plane(v0.t, v1.t, v2.t);
        primitive.texQ = plane(v0.q, v1.q, v2.q);
    }
    primitive.zBase = v2.z;
    primitive.zDx = (v0.z - v2.z) * static_cast<double>(wa[0]) + (v1.z - v2.z) * static_cast<double>(wa[1]);
    primitive.zDy = (v0.z - v2.z) * static_cast<double>(wb[0]) + (v1.z - v2.z) * static_cast<double>(wb[1]);

    if (p.tme)
    {
        float lowU = 1.0e30f, lowV = 1.0e30f, highU = -1.0e30f, highV = -1.0e30f;
        for (const GSVertex &vertex : primitive.v)
        {
            float u, v;
            if (p.fst)
            {
                u = static_cast<float>(vertex.u) / 16.0f;
                v = static_cast<float>(vertex.v) / 16.0f;
            }
            else
            {
                const float q = std::fabs(vertex.q) > 1.0e-8f ? vertex.q : 1.0f;
                u = (vertex.s / q) * static_cast<float>(p.texW);
                v = (vertex.t / q) * static_cast<float>(p.texH);
            }
            lowU = std::min(lowU, u);
            highU = std::max(highU, u);
            lowV = std::min(lowV, v);
            highV = std::max(highV, v);
        }
        // Every pixel's coordinates lie between the vertices', give or take rounding.
        const float largest = std::max(std::max(std::fabs(lowU), std::fabs(highU)), std::max(std::fabs(lowV), std::fabs(highV)));
        p.setSampledCoordinates(lowU, lowV, highU, highV, 0.05f + 1.0e-5f * largest);
    }
    return true;
}

[[gnu::noinline]] bool GSCpuBackend::PrepareSprite(const GSPrimitiveBatch &batch, PreparedPrimitive &primitive)
{
    const GSDrawState &state = batch.state;
    FastPipeline &p = primitive.pipeline;
    if (!p.setup(*this, state))
        return false;

    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];
    const GSContext &ctx = state.context;

    const int ofx = ctx.xyoffset.ofx >> 4;
    const int ofy = ctx.xyoffset.ofy >> 4;
    int x0 = static_cast<int>(v0.x) - ofx;
    int y0 = static_cast<int>(v0.y) - ofy;
    int x1 = static_cast<int>(v1.x) - ofx;
    int y1 = static_cast<int>(v1.y) - ofy;

    // Which corner each vertex is at decides the direction the texture runs.
    const bool flipX = x0 > x1;
    const bool flipY = y0 > y1;
    if (flipX)
        std::swap(x0, x1);
    if (flipY)
        std::swap(y0, y1);

    const int spanX = std::max(1, x1 - x0);
    const int spanY = std::max(1, y1 - y0);
    const int lastX = x0 + spanX - 1;
    const int lastY = y0 + spanY - 1;
    if (lastX < p.scissorX0 || x0 > p.scissorX1 || lastY < p.scissorY0 || y0 > p.scissorY1)
        return true;

    primitive.sprite = true;
    primitive.v[0] = v0;
    primitive.v[1] = v1;
    primitive.minX = clampInt(x0, p.scissorX0, p.scissorX1);
    primitive.minY = clampInt(y0, p.scissorY0, p.scissorY1);
    primitive.maxX = clampInt(lastX, p.scissorX0, p.scissorX1);
    primitive.maxY = clampInt(lastY, p.scissorY0, p.scissorY1);
    primitive.originX = x0;
    primitive.originY = y0;
    primitive.spanX = static_cast<float>(spanX);
    primitive.spanY = static_cast<float>(spanY);

    if (!p.tme)
        return true;

    // Texel coordinates at the two vertices.
    float tu0, tv0, tu1, tv1;
    if (p.fst)
    {
        tu0 = static_cast<float>(v0.u >> 4);
        tv0 = static_cast<float>(v0.v >> 4);
        tu1 = static_cast<float>(v1.u >> 4);
        tv1 = static_cast<float>(v1.v >> 4);
    }
    else
    {
        const float q0 = std::fabs(v0.q) > 1.0e-8f ? v0.q : 1.0f;
        const float q1 = std::fabs(v1.q) > 1.0e-8f ? v1.q : 1.0f;
        tu0 = (v0.s / q0) * static_cast<float>(p.texW);
        tv0 = (v0.t / q0) * static_cast<float>(p.texH);
        tu1 = (v1.s / q1) * static_cast<float>(p.texW);
        tv1 = (v1.t / q1) * static_cast<float>(p.texH);
    }
    // The texture runs from the first vertex to the second, wherever they are.
    // (The reference path maps the first vertex's coordinates to the top left
    // corner regardless, which un-mirrors a sprite drawn right to left.)
    if (flipX)
        std::swap(tu0, tu1);
    if (flipY)
        std::swap(tv0, tv1);
    primitive.tu0 = tu0;
    primitive.tv0 = tv0;
    primitive.tu1 = tu1;
    primitive.tv1 = tv1;
    // The coordinates of the first and last pixel drawn, computed as
    // RasteriseRows does; the others lie between them.
    {
        const float stepU = (primitive.tu1 - primitive.tu0) / primitive.spanX;
        const float stepV = (primitive.tv1 - primitive.tv0) / primitive.spanY;
        const bool fst = p.fst;
        auto quantise = [fst](float coordinate)
        {
            return fst ? static_cast<float>(clampInt(static_cast<int>((coordinate * 16.0f) + 0.5f), 0, 0xFFFF)) * 0.0625f
                       : coordinate;
        };
        const float rowU = primitive.tu0 + (static_cast<float>(primitive.minX - primitive.originX) + 0.5f) * stepU;
        const float firstU = quantise(rowU);
        const float lastU = quantise(rowU + static_cast<float>(primitive.maxX - primitive.minX) * stepU);
        const float firstV = quantise(primitive.tv0 + (static_cast<float>(primitive.minY - primitive.originY) + 0.5f) * stepV);
        const float lastV = quantise(primitive.tv0 + (static_cast<float>(primitive.maxY - primitive.originY) + 0.5f) * stepV);
        // Fixed-point coordinates come out the same here as there; a divide
        // by Q may round differently.
        p.setSampledCoordinates(firstU, firstV, lastU, lastV, fst ? 0.0f : 1.0f / 64.0f);
    }
    return true;
}

// Draws rows yFirst..yLast of a prepared primitive. Touches only those rows of
// the targets, so different rows can be drawn on different threads.
void GSCpuBackend::Rasterise(const PreparedPrimitive &primitive, int yFirst, int yLast)
{
    if (rasterClassStatsWanted()) [[unlikely]]
    {
        const FastPipeline &p = primitive.pipeline;
        const uint32_t key = (primitive.sprite ? 1u : 0u) | (p.tme ? 2u : 0u) | (p.tme && p.linear ? 4u : 0u) | (p.decoded ? 8u : 0u) |
                             (p.fge ? 16u : 0u) | (p.abe ? 32u : 0u) | (p.alphaTest ? 64u : 0u) | (static_cast<uint32_t>(p.ztst & 3u) << 7) |
                             (p.zmask ? 0x200u : 0u) | (p.tme && p.fst ? 0x400u : 0u) | (primitive.gouraud ? 0x800u : 0u) |
                             (p.tme ? static_cast<uint32_t>(p.tfx & 3u) << 12 : 0u) | (p.tme && p.tcc ? 0x4000u : 0u) |
                             (p.tme ? static_cast<uint32_t>(p.texKind) << 15 : 0u) | (static_cast<uint32_t>(p.frameKind) << 18) |
                             (static_cast<uint32_t>(p.depthKind) << 20) | (p.date ? 0x400000u : 0u) | (p.fbmsk != 0u ? 0x800000u : 0u);
        RasterTally tally;
        const auto start = std::chrono::steady_clock::now();
        RasteriseRows(primitive, yFirst, yLast, &tally);
        const auto elapsed = std::chrono::steady_clock::now() - start;
        addRasterClassSample(key, static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()),
                             tally.pixels, tally.rows, tally.blocks);
        return;
    }
    RasteriseRows(primitive, yFirst, yLast, nullptr);
}

void GSCpuBackend::RasteriseRows(const PreparedPrimitive &primitive, int yFirst, int yLast, RasterTally *tally)
{
    const FastPipeline &p = primitive.pipeline;
    // A primitive that samples what it is drawing needs each pixel stored
    // before the next is sampled. PS2X_GS_SCALAR=1 draws everything that way.
    static const bool scalarOnly = []
    {
        const char *value = std::getenv("PS2X_GS_SCALAR");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    if (scalarOnly || (p.tme && p.textureIsTarget))
    {
        RasteriseRowsScalar(primitive, yFirst, yLast, tally);
        return;
    }

    const GSVertex &v1 = primitive.v[1];
    PixelBlock block;

    if (primitive.sprite)
    {
        if (tally)
        {
            const uint64_t rows = static_cast<uint64_t>(yLast - yFirst + 1);
            const uint64_t width = static_cast<uint64_t>(primitive.maxX - primitive.minX + 1);
            tally->pixels += rows * width;
            tally->rows += rows;
            tally->blocks += rows * ((width + kLanes - 1u) / kLanes);
        }
        for (int i = 0; i < kLanes; ++i)
        {
            block.dx[i] = 0.0f;
            block.z[i] = static_cast<uint32_t>(v1.z);
            block.r[i] = v1.r;
            block.g[i] = v1.g;
            block.b[i] = v1.b;
            block.a[i] = v1.a;
            block.fog[i] = v1.fog;
            block.texU[i] = 0.0f;
            block.texV[i] = 0.0f;
        }
        const float stepU = p.tme ? (primitive.tu1 - primitive.tu0) / primitive.spanX : 0.0f;
        const float stepV = p.tme ? (primitive.tv1 - primitive.tv0) / primitive.spanY : 0.0f;
        const bool fst = p.fst;
        // With FST the coordinates are 12.4 fixed point, as the reference rounds them.
        auto quantise = [fst](float coordinate)
        {
            return fst ? static_cast<float>(clampInt(static_cast<int>((coordinate * 16.0f) + 0.5f), 0, 0xFFFF)) * 0.0625f
                       : coordinate;
        };
        const float rowU = primitive.tu0 + (static_cast<float>(primitive.minX - primitive.originX) + 0.5f) * stepU;
        for (int y = yFirst; y <= yLast; ++y)
        {
            const FastPipeline::Row frameRow = FastPipeline::rowOf(p.frame, y);
            const FastPipeline::Row depthRow = FastPipeline::rowOf(p.depth, y);
            if (p.tme)
            {
                const float texV = quantise(primitive.tv0 + (static_cast<float>(y - primitive.originY) + 0.5f) * stepV);
                for (int i = 0; i < kLanes; ++i)
                    block.texV[i] = texV;
            }
            for (int x = primitive.minX; x <= primitive.maxX; x += kLanes)
            {
                if (p.tme)
                {
                    const int column = x - primitive.minX;
                    for (int i = 0; i < kLanes; ++i)
                        block.texU[i] = quantise(rowU + static_cast<float>(column + i) * stepU);
                }
                p.shade(block, x, std::min(kLanes, primitive.maxX - x + 1), frameRow, depthRow);
            }
        }
        return;
    }

    const GSVertex &v2 = primitive.v[2];
    const float *wa = primitive.wa, *wb = primitive.wb, *wc = primitive.wc, *invWa = primitive.invWa;
    const bool *ownsEdge = primitive.ownsEdge;
    const float fx2 = primitive.fx2, fy2 = primitive.fy2;
    const int minX = primitive.minX, maxX = primitive.maxX;

    const bool gouraud = primitive.gouraud;
    const bool textured = p.tme, fst = p.fst, fogged = p.fge;
    const float texWf = static_cast<float>(p.texW);
    const float texHf = static_cast<float>(p.texH);
    const float stepR = primitive.red.dx, stepG = primitive.green.dx, stepB = primitive.blue.dx, stepA = primitive.alpha.dx;
    const float stepFog = primitive.fog.dx;
    const float stepS = primitive.texS.dx, stepT = primitive.texT.dx, stepQ = primitive.texQ.dx;
    const double stepZ = primitive.zDx;
    for (int i = 0; i < kLanes; ++i)
    {
        block.r[i] = v2.r;
        block.g[i] = v2.g;
        block.b[i] = v2.b;
        block.a[i] = v2.a;
        block.fog[i] = 0;
        block.texU[i] = 0.0f;
        block.texV[i] = 0.0f;
    }

    for (int y = yFirst; y <= yLast; ++y)
    {
        const float dy = (static_cast<float>(y) + 0.5f) - fy2;

        // The pixels of this row whose centres are inside all three edges.
        float rowW[3];
        float lo = static_cast<float>(minX);
        float hi = static_cast<float>(maxX);
        bool empty = false;
        for (int i = 0; i < 3; ++i)
        {
            rowW[i] = wb[i] * dy + wc[i]; // weight at px = fx2
            // Solve w_i = 0 for the pixel index, a pixel generous either way.
            if (wa[i] > 0.0f)
                lo = std::max(lo, fx2 - (rowW[i] * invWa[i]) - 1.5f);
            else if (wa[i] < 0.0f)
                hi = std::min(hi, fx2 - (rowW[i] * invWa[i]) + 0.5f);
            else if (rowW[i] < 0.0f || (rowW[i] == 0.0f && !ownsEdge[i]))
                empty = true;
        }
        if (empty || !(lo <= hi))
            continue;

        int xStart = std::max(minX, static_cast<int>(std::floor(lo)));
        int xEnd = std::min(maxX, static_cast<int>(std::ceil(hi)));
        auto inside = [&](int x)
        {
            const float dx = (static_cast<float>(x) + 0.5f) - fx2;
            const float w0 = wa[0] * dx + rowW[0];
            const float w1 = wa[1] * dx + rowW[1];
            const float w2 = 1.0f - w0 - w1;
            return (w0 > 0.0f || (w0 == 0.0f && ownsEdge[0])) &&
                   (w1 > 0.0f || (w1 == 0.0f && ownsEdge[1])) &&
                   (w2 > 0.0f || (w2 == 0.0f && ownsEdge[2]));
        };
        // Tighten the generous bounds to the first and last covered pixel.
        while (xStart <= xEnd && !inside(xStart))
            ++xStart;
        while (xEnd >= xStart && !inside(xEnd))
            --xEnd;
        if (xStart > xEnd)
            continue;
        if (tally)
        {
            const uint64_t width = static_cast<uint64_t>(xEnd - xStart + 1);
            tally->pixels += width;
            tally->rows += 1u;
            tally->blocks += (width + kLanes - 1u) / kLanes;
        }

        const FastPipeline::Row frameRow = FastPipeline::rowOf(p.frame, y);
        const FastPipeline::Row depthRow = FastPipeline::rowOf(p.depth, y);

        // Attribute values on this row at dx = 0 (vertex 2's column).
        auto onRow = [&](const PreparedPrimitive::Plane &plane)
        {
            return plane.base + plane.dy * dy;
        };
        const float rowR = onRow(primitive.red), rowG = onRow(primitive.green), rowB = onRow(primitive.blue);
        const float rowA = onRow(primitive.alpha), rowFog = onRow(primitive.fog);
        const float rowS = onRow(primitive.texS), rowT = onRow(primitive.texT), rowQ = onRow(primitive.texQ);
        const double rowZ = primitive.zBase + primitive.zDy * static_cast<double>(dy);

        for (int x = xStart; x <= xEnd; x += kLanes)
        {
            for (int i = 0; i < kLanes; ++i)
                block.dx[i] = (static_cast<float>(x + i) + 0.5f) - fx2;
            for (int i = 0; i < kLanes; ++i)
                block.z[i] = static_cast<uint32_t>(rowZ + stepZ * static_cast<double>(block.dx[i]) + 0.5);
            if (gouraud)
            {
                for (int i = 0; i < kLanes; ++i)
                {
                    block.r[i] = clampU8(static_cast<int>(rowR + stepR * block.dx[i]));
                    block.g[i] = clampU8(static_cast<int>(rowG + stepG * block.dx[i]));
                    block.b[i] = clampU8(static_cast<int>(rowB + stepB * block.dx[i]));
                    block.a[i] = clampU8(static_cast<int>(rowA + stepA * block.dx[i]));
                }
            }
            if (textured)
            {
                if (fst)
                {
                    for (int i = 0; i < kLanes; ++i)
                    {
                        block.texU[i] = static_cast<float>(static_cast<uint16_t>(rowS + stepS * block.dx[i])) * 0.0625f;
                        block.texV[i] = static_cast<float>(static_cast<uint16_t>(rowT + stepT * block.dx[i])) * 0.0625f;
                    }
                }
                else
                {
                    // S, T and Q are interpolated; the divide comes after.
                    for (int i = 0; i < kLanes; ++i)
                    {
                        float q = rowQ + stepQ * block.dx[i];
                        if (!(std::fabs(q) > 1.0e-8f))
                            q = 1.0f;
                        const float invQ = 1.0f / q;
                        block.texU[i] = (rowS + stepS * block.dx[i]) * invQ * texWf;
                        block.texV[i] = (rowT + stepT * block.dx[i]) * invQ * texHf;
                    }
                }
            }
            if (fogged)
            {
                for (int i = 0; i < kLanes; ++i)
                    block.fog[i] = clampU8(static_cast<int>(rowFog + stepFog * block.dx[i]));
            }
            p.shade(block, x, std::min(kLanes, xEnd - x + 1), frameRow, depthRow);
        }
    }
}

// One pixel at a time, each stored before the next is sampled.
void GSCpuBackend::RasteriseRowsScalar(const PreparedPrimitive &primitive, int yFirst, int yLast, RasterTally *tally)
{
    const FastPipeline &p = primitive.pipeline;
    const GSVertex &v0 = primitive.v[0];
    const GSVertex &v1 = primitive.v[1];
    if (tally && primitive.sprite)
    {
        const uint64_t rows = static_cast<uint64_t>(yLast - yFirst + 1);
        const uint64_t width = static_cast<uint64_t>(primitive.maxX - primitive.minX + 1);
        tally->pixels += rows * width;
        tally->rows += rows;
        tally->blocks += rows * ((width + kLanes - 1u) / kLanes);
    }

    if (primitive.sprite)
    {
        const uint32_t z = static_cast<uint32_t>(v1.z);
        const int r = v1.r, g = v1.g, b = v1.b, a = v1.a;
        const int fog = v1.fog;
        if (!p.tme)
        {
            for (int y = yFirst; y <= yLast; ++y)
            {
                const FastPipeline::Row frameRow = FastPipeline::rowOf(p.frame, y);
                const FastPipeline::Row depthRow = FastPipeline::rowOf(p.depth, y);
                for (int x = primitive.minX; x <= primitive.maxX; ++x)
                    p.write(x, frameRow, depthRow, z, r, g, b, a, fog);
            }
            return;
        }

        // Texel coordinates step evenly across the sprite; with FST they are
        // 12.4 fixed point, as the reference rounds them.
        const float stepU = (primitive.tu1 - primitive.tu0) / primitive.spanX;
        const float stepV = (primitive.tv1 - primitive.tv0) / primitive.spanY;
        auto quantise = [&](float coordinate)
        {
            return p.fst ? static_cast<float>(clampInt(static_cast<int>((coordinate * 16.0f) + 0.5f), 0, 0xFFFF)) * 0.0625f
                         : coordinate;
        };
        for (int y = yFirst; y <= yLast; ++y)
        {
            const FastPipeline::Row frameRow = FastPipeline::rowOf(p.frame, y);
            const FastPipeline::Row depthRow = FastPipeline::rowOf(p.depth, y);
            const float texV = quantise(primitive.tv0 + (static_cast<float>(y - primitive.originY) + 0.5f) * stepV);
            const float rowU = primitive.tu0 + (static_cast<float>(primitive.minX - primitive.originX) + 0.5f) * stepU;
            for (int x = primitive.minX; x <= primitive.maxX; ++x)
            {
                const float texU = quantise(rowU + static_cast<float>(x - primitive.minX) * stepU);
                int pr = r, pg = g, pb = b, pa = a;
                p.combine(p.sample(texU, texV), pr, pg, pb, pa);
                p.write(x, frameRow, depthRow, z, pr, pg, pb, pa, fog);
            }
        }
        return;
    }

    const GSVertex &v2 = primitive.v[2];
    const float *wa = primitive.wa, *wb = primitive.wb, *wc = primitive.wc;
    const bool *ownsEdge = primitive.ownsEdge;
    const float fx2 = primitive.fx2, fy2 = primitive.fy2;
    const int minX = primitive.minX, maxX = primitive.maxX;

    const bool gouraud = primitive.gouraud;
    const float texWf = static_cast<float>(p.texW);
    const float texHf = static_cast<float>(p.texH);
    const int flatR = v2.r, flatG = v2.g, flatB = v2.b, flatA = v2.a;

    for (int y = yFirst; y <= yLast; ++y)
    {
        const float dy = (static_cast<float>(y) + 0.5f) - fy2;

        // The pixels of this row whose centres are inside all three edges.
        float rowW[3];
        float lo = static_cast<float>(minX);
        float hi = static_cast<float>(maxX);
        bool empty = false;
        for (int i = 0; i < 3; ++i)
        {
            rowW[i] = wb[i] * dy + wc[i]; // weight at px = fx2
            // Solve w_i = 0 for the pixel index, a pixel generous either way.
            if (wa[i] > 0.0f)
                lo = std::max(lo, fx2 - (rowW[i] / wa[i]) - 1.5f);
            else if (wa[i] < 0.0f)
                hi = std::min(hi, fx2 - (rowW[i] / wa[i]) + 0.5f);
            else if (rowW[i] < 0.0f || (rowW[i] == 0.0f && !ownsEdge[i]))
                empty = true;
        }
        if (empty || !(lo <= hi))
            continue;

        int xStart = std::max(minX, static_cast<int>(std::floor(lo)));
        int xEnd = std::min(maxX, static_cast<int>(std::ceil(hi)));
        auto inside = [&](int x)
        {
            const float dx = (static_cast<float>(x) + 0.5f) - fx2;
            const float w0 = wa[0] * dx + rowW[0];
            const float w1 = wa[1] * dx + rowW[1];
            const float w2 = 1.0f - w0 - w1;
            return (w0 > 0.0f || (w0 == 0.0f && ownsEdge[0])) &&
                   (w1 > 0.0f || (w1 == 0.0f && ownsEdge[1])) &&
                   (w2 > 0.0f || (w2 == 0.0f && ownsEdge[2]));
        };
        // Tighten the generous bounds to the first and last covered pixel.
        while (xStart <= xEnd && !inside(xStart))
            ++xStart;
        while (xEnd >= xStart && !inside(xEnd))
            --xEnd;
        if (xStart > xEnd)
            continue;

        if (tally)
        {
            const uint64_t width = static_cast<uint64_t>(xEnd - xStart + 1);
            tally->pixels += width;
            tally->rows += 1u;
            tally->blocks += (width + kLanes - 1u) / kLanes;
        }

        const FastPipeline::Row frameRow = FastPipeline::rowOf(p.frame, y);
        const FastPipeline::Row depthRow = FastPipeline::rowOf(p.depth, y);

        // Attribute values on this row at dx = 0 (vertex 2's column).
        auto onRow = [&](const PreparedPrimitive::Plane &plane)
        {
            return plane.base + plane.dy * dy;
        };
        const float rowR = onRow(primitive.red), rowG = onRow(primitive.green), rowB = onRow(primitive.blue);
        const float rowA = onRow(primitive.alpha), rowFog = onRow(primitive.fog);
        const float rowS = onRow(primitive.texS), rowT = onRow(primitive.texT), rowQ = onRow(primitive.texQ);
        const double rowZ = primitive.zBase + primitive.zDy * static_cast<double>(dy);

        for (int x = xStart; x <= xEnd; ++x)
        {
            const float dx = (static_cast<float>(x) + 0.5f) - fx2;
            const double z = rowZ + primitive.zDx * static_cast<double>(dx);

            int r, g, b, a;
            if (gouraud)
            {
                r = clampU8(static_cast<int>(rowR + primitive.red.dx * dx));
                g = clampU8(static_cast<int>(rowG + primitive.green.dx * dx));
                b = clampU8(static_cast<int>(rowB + primitive.blue.dx * dx));
                a = clampU8(static_cast<int>(rowA + primitive.alpha.dx * dx));
            }
            else
            {
                r = flatR;
                g = flatG;
                b = flatB;
                a = flatA;
            }

            if (p.tme)
            {
                float texU, texV;
                if (p.fst)
                {
                    texU = static_cast<float>(static_cast<uint16_t>(rowS + primitive.texS.dx * dx)) * 0.0625f;
                    texV = static_cast<float>(static_cast<uint16_t>(rowT + primitive.texT.dx * dx)) * 0.0625f;
                }
                else
                {
                    // S, T and Q are interpolated; the divide comes after.
                    float q = rowQ + primitive.texQ.dx * dx;
                    if (!(std::fabs(q) > 1.0e-8f))
                        q = 1.0f;
                    const float invQ = 1.0f / q;
                    texU = (rowS + primitive.texS.dx * dx) * invQ * texWf;
                    texV = (rowT + primitive.texT.dx * dx) * invQ * texHf;
                }
                p.combine(p.sample(texU, texV), r, g, b, a);
            }

            const int fog = p.fge ? clampU8(static_cast<int>(rowFog + primitive.fog.dx * dx)) : 0;
            p.write(x, frameRow, depthRow, static_cast<uint32_t>(z + 0.5), r, g, b, a, fog);
        }
    }
}

// Queues or draws one triangle or sprite. Returns false when the fast path
// cannot handle the state (nothing has been drawn then).
bool GSCpuBackend::DrawFast(const GSPrimitiveBatch &batch, bool sprite)
{
    if (!m_rasterQueue)
        m_rasterQueue.reset(new RasterQueue());
    RasterQueue &queue = *m_rasterQueue;

    PreparedPrimitive primitive;
    if (!(sprite ? PrepareSprite(batch, primitive) : PrepareTriangle(batch, primitive)))
        return false;
    if (primitive.empty())
        return true;
    FastPipeline &p = primitive.pipeline;

    // Primitives wait together only while they draw to the same buffers.
    const GSContext &ctx = batch.state.context;
    if (!queue.pending.empty() &&
        (queue.fbp != ctx.frame.fbp || queue.fbw != ctx.frame.fbw || queue.fpsm != ctx.frame.psm ||
         queue.zbp != ctx.zbuf.zbp || queue.zpsm != ctx.zbuf.psm))
    {
        queue.submit("target");
    }

    // Its texture has to be read after everything that draws into it.
    if (p.tme)
        SubmitPrimitivesWriting(p.texture, p.sampledU0, p.sampledV0, p.sampledU1, p.sampledV1, "texture");

    p.useDecodedTexture(primitive.minX, primitive.minY, primitive.maxX, primitive.maxY);
    MarkWritten(p.frame, primitive.minX, primitive.minY, primitive.maxX, primitive.maxY);
    if (!p.zmask)
        MarkWritten(p.depth, primitive.minX, primitive.minY, primitive.maxX, primitive.maxY);

    // A texture sampled from VRAM while drawing can wait with the others
    // unless it is its own target (every pixel written changes what the next
    // one reads) or needs the shared CLUT conversion: those are drawn now, on
    // this thread.
    const bool live = p.tme && !p.decoded;
    if (live && (p.textureIsTarget || p.texKind >= TextureKind::Index8))
    {
        queue.run("direct");
        Rasterise(primitive, primitive.minY, primitive.maxY);
        return true;
    }

    // What this primitive writes must not be something a queued one still has
    // to sample.
    PageSet writes{};
    addPages(writes, p.frame, primitive.minX, primitive.minY, primitive.maxX, primitive.maxY);
    if (!p.zmask)
        addPages(writes, p.depth, primitive.minX, primitive.minY, primitive.maxX, primitive.maxY);
    if (!queue.pending.empty() && intersects(writes, queue.readLive))
        queue.submit("texture");

    if (queue.pending.empty())
    {
        queue.fbp = ctx.frame.fbp;
        queue.fbw = ctx.frame.fbw;
        queue.fpsm = ctx.frame.psm;
        queue.zbp = ctx.zbuf.zbp;
        queue.zpsm = ctx.zbuf.psm;
    }
    for (size_t i = 0; i < writes.size(); ++i)
        queue.written[i] |= writes[i];
    // A copy that is never decoded again is no reason for a transfer to its
    // pages to wait.
    if (p.tme && !p.decodedIsFixed)
    {
        addPages(queue.read, p.texture, p.sampledU0, p.sampledV0, p.sampledU1, p.sampledV1);
        if (live)
            addPages(queue.readLive, p.texture, p.sampledU0, p.sampledV0, p.sampledU1, p.sampledV1);
    }
    queue.pendingPixels += static_cast<uint64_t>(primitive.maxX - primitive.minX + 1) *
                           static_cast<uint64_t>(primitive.maxY - primitive.minY + 1);
    // The queued copy must not point at this call's draw state.
    p.drawState = nullptr;
    queue.pending.push_back(primitive);

    // Enough for the pool to start on if it has nothing to do; otherwise let
    // the batch grow.
    if (queue.pending.size() >= RasterQueue::kMaxPending)
        queue.submit("full");
    else if (queue.pendingPixels >= RasterQueue::kEagerPixels && !queue.busy())
        queue.submit("idle");
    return true;
}
