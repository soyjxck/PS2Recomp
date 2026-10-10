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
#include <mutex>
#include <thread>

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
    // The decoded copy, when the texture can use one.
    const uint32_t *decoded = nullptr;
    // The pixels the primitive can touch, for telling a texture that is also
    // the render target.
    int boundsX0 = 0, boundsY0 = 0, boundsX1 = 0, boundsY1 = 0;
    bool textureIsTarget = false;
    // The texels the primitive can sample (the whole texture when it wraps).
    int sampledU0 = 0, sampledV0 = 0, sampledU1 = 0, sampledV1 = 0;

    // Narrows the sampled rectangle to texel coordinates [u0, u1] x [v0, v1].
    void setSampledTexels(float u0, float v0, float u1, float v1)
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
        // A texel of margin for bilinear filtering and rounding.
        if (u0 >= 2.0f && u1 < static_cast<float>(texW - 2))
        {
            sampledU0 = static_cast<int>(u0) - 2;
            sampledU1 = static_cast<int>(u1) + 2;
        }
        if (v0 >= 2.0f && v1 < static_cast<float>(texH - 2))
        {
            sampledV0 = static_cast<int>(v0) - 2;
            sampledV1 = static_cast<int>(v1) + 2;
        }
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
                decoded = copy->pixels.data();
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
        u = wrapCoordinate(u, texW, wrapU, minU, maxU);
        v = wrapCoordinate(v, texH, wrapV, minV, maxV);
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
    [[gnu::always_inline]] inline void write(int x, int y, uint32_t z, int r, int g, int b, int a, int fog) const
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
        const uint32_t frameUnit = frame.unit(static_cast<uint32_t>(x), static_cast<uint32_t>(y));

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
            depthUnit = depth.unit(static_cast<uint32_t>(x), static_cast<uint32_t>(y));

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
                auto pick = [](uint8_t sel, int source, int dest)
                {
                    return sel == 0 ? source : (sel == 1 ? dest : 0);
                };
                r = clampU8((((pick(asel, r, dr) - pick(bsel, r, dr)) * alpha) >> 7) + pick(dsel, r, dr));
                g = clampU8((((pick(asel, g, dg) - pick(bsel, g, dg)) * alpha) >> 7) + pick(dsel, g, dg));
                b = clampU8((((pick(asel, b, db) - pick(bsel, b, db)) * alpha) >> 7) + pick(dsel, b, db));
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

    const GSDrawState *drawState = nullptr;
};

bool GSCpuBackend::FastPipeline::setup(GSCpuBackend &owner, const GSDrawState &state)
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

uint64_t GSCpuBackend::HashPages(const std::vector<uint32_t> &pages) const
{
    uint64_t hash = 0xCBF29CE484222325ull;
    for (const uint32_t page : pages)
        hash = hashBytes(m_vram + static_cast<size_t>(page) * GSMem::GS_PAGE_SIZE, GSMem::GS_PAGE_SIZE, hash);
    return hash;
}

const GSCpuBackend::DecodedTexture *GSCpuBackend::DecodedTextureFor(FastPipeline &p)
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

    uint64_t key = mixKey(0x243F6A8885A308D3ull, (static_cast<uint64_t>(p.tbp0) << 32) | (static_cast<uint64_t>(p.tbw) << 16) | p.texPsm);
    key = mixKey(key, (static_cast<uint64_t>(p.texW) << 32) | static_cast<uint64_t>(p.texH));
    if (p.texKind >= TextureKind::Index8)
    {
        // The CLUT entries this texture can reach, as raw words.
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
            clutHash = hashBytes(reinterpret_cast<const uint8_t *>(m_clut.data()), m_clut.size() * sizeof(uint16_t), clutHash);
        }
        key = mixKey(key, clutHash);
    }
    if (p.texKind != TextureKind::Direct32)
    {
        key = mixKey(key, (static_cast<uint64_t>(p.texa.ta0) << 16) | (static_cast<uint64_t>(p.texa.ta1) << 8) | (p.texa.aem ? 1u : 0u));
    }

    // In texture lookups: how soon a second content change marks a texture as
    // volatile, and for how long it is then read from VRAM.
    constexpr uint64_t kVolatileWindow = 30000u;
    constexpr uint64_t kVolatileHold = 300000u;

    const auto [it, inserted] = m_decodedTextures.try_emplace(key);
    DecodedTexture &copy = it->second;
    copy.lastUsed = ++m_decodedUseClock;

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
        if (written && HashPages(copy.pages) != copy.contentHash)
            decode = true;
    }

    if (decode && !inserted)
    {
        // A texture whose contents keep changing is a render target being
        // read back (a feedback or blur pass). Decoding all of it for the few
        // texels a pass reads costs more than reading them from VRAM.
        const bool changedRecently = copy.lastDecoded + kVolatileWindow > m_decodedUseClock;
        copy.lastDecoded = m_decodedUseClock;
        if (changedRecently)
        {
            copy.directUntil = m_decodedUseClock + kVolatileHold;
            copy.pixels.clear();
            copy.pixels.shrink_to_fit();
        }
    }
    if (copy.directUntil > m_decodedUseClock)
    {
        copy.checkedSerial = m_writeSerial;
        return nullptr;
    }
    if (copy.pixels.empty())
        decode = true;
    if (decode)
    {
        copy.pixels.resize(static_cast<size_t>(p.texW) * static_cast<size_t>(p.texH));
        uint32_t *out = copy.pixels.data();
        for (int v = 0; v < p.texH; ++v)
            for (int u = 0; u < p.texW; ++u)
                *out++ = p.fetch(u, v);
        copy.contentHash = HashPages(copy.pages);
    }
    copy.checkedSerial = m_writeSerial;

    // Forget copies that have not been used for a long while (queued
    // primitives hold pointers into them, so only between batches).
    if (m_decodedTextures.size() > 1024u && !HasQueuedPrimitives())
    {
        for (auto entry = m_decodedTextures.begin(); entry != m_decodedTextures.end();)
        {
            if (entry->second.lastUsed + 2048u < m_decodedUseClock)
                entry = m_decodedTextures.erase(entry);
            else
                ++entry;
        }
    }
    return &copy;
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
    bool ownsEdge[3]{};

    // Sprite: its unclipped origin and size, and texel coordinates at the
    // corners.
    int originX = 0, originY = 0;
    float spanX = 1.0f, spanY = 1.0f;
    float tu0 = 0.0f, tv0 = 0.0f, tu1 = 0.0f, tv1 = 0.0f;

    bool empty() const { return maxX < minX || maxY < minY; }
};

// Primitives waiting to be drawn, and the workers that draw them. Rows are
// handed out in bands; within a band primitives are drawn in submission order,
// so the result is the same as drawing them one after another.
struct GSCpuBackend::RasterQueue
{
    static constexpr int kBandHeight = 16;
    // Below this many bounding-box pixels a batch is drawn by the caller alone.
    static constexpr uint64_t kParallelPixels = 24000u;
    static constexpr size_t kMaxPending = 4096u;

    std::vector<PreparedPrimitive> pending;
    uint64_t pendingPixels = 0;
    PageSet written{};  // pages pending primitives write
    PageSet read{};     // pages their textures occupy
    PageSet readLive{}; // ...of those, pages sampled from VRAM while drawing
    // Target of the pending primitives.
    uint32_t fbp = 0, fbw = 0, zbp = 0;
    uint8_t fpsm = 0, zpsm = 0;

    // Worker pool.
    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable wake;
    std::condition_variable done;
    uint64_t generation = 0;
    int active = 0;
    bool quit = false;
    std::atomic<int> nextBand{0};
    int bandCount = 0;
    int firstRow = 0;

    void drawBands()
    {
        for (;;)
        {
            const int band = nextBand.fetch_add(1, std::memory_order_relaxed);
            if (band >= bandCount)
                return;
            const int bandFirst = firstRow + band * kBandHeight;
            const int bandLast = bandFirst + kBandHeight - 1;
            for (const PreparedPrimitive &primitive : pending)
            {
                const int y0 = std::max(primitive.minY, bandFirst);
                const int y1 = std::min(primitive.maxY, bandLast);
                if (y0 <= y1)
                    Rasterise(primitive, y0, y1);
            }
        }
    }

    void workerLoop()
    {
        uint64_t seen = 0;
        std::unique_lock<std::mutex> lock(mutex);
        for (;;)
        {
            wake.wait(lock, [&]
                      { return quit || generation != seen; });
            if (quit)
                return;
            seen = generation;
            lock.unlock();
            drawBands();
            lock.lock();
            if (--active == 0)
                done.notify_one();
        }
    }

    void startWorkers()
    {
        if (!workers.empty())
            return;
        // Leave cores for the EE, the IOP and the window; efficiency cores
        // would only hold a band up.
        unsigned count = std::thread::hardware_concurrency();
        count = count > 3u ? std::min(count - 3u, 6u) : 0u;
        if (const char *value = std::getenv("PS2X_GS_THREADS"))
            count = static_cast<unsigned>(std::max(0, std::atoi(value) - 1));
        for (unsigned i = 0; i < count; ++i)
            workers.emplace_back([this]
                                 { workerLoop(); });
    }

    // PS2X_GS_STATS=1: how the queue is being used, every few thousand batches.
    struct Stats
    {
        uint64_t batches = 0, parallelBatches = 0, primitives = 0, parallelPixels = 0, inlinePixels = 0;
        double parallelSeconds = 0.0, inlineSeconds = 0.0;
        uint64_t reason[6]{};
    } stats;
    const char *nextReason = "other";

    void run(const char *why = nullptr)
    {
        if (pending.empty())
            return;

        static const bool wantStats = []
        {
            const char *value = std::getenv("PS2X_GS_STATS");
            return value != nullptr && value[0] != '\0' && value[0] != '0';
        }();
        if (wantStats)
        {
            static const char *const kReasons[6] = {"target", "texture", "direct", "full", "upload", "other"};
            const char *reason = why ? why : "other";
            for (int i = 0; i < 6; ++i)
                if (std::strcmp(reason, kReasons[i]) == 0)
                    ++stats.reason[i];
            ++stats.batches;
            stats.primitives += pending.size();
            if (pendingPixels >= kParallelPixels)
            {
                ++stats.parallelBatches;
                stats.parallelPixels += pendingPixels;
            }
            else
                stats.inlinePixels += pendingPixels;
            if (stats.batches % 2000u == 0u)
            {
                std::fprintf(stderr, "[gs-queue] batches=%llu (parallel %llu) prims/batch=%.1f bbox pixels: parallel=%lluM inline=%lluM; flushed by",
                             (unsigned long long)stats.batches, (unsigned long long)stats.parallelBatches,
                             static_cast<double>(stats.primitives) / static_cast<double>(stats.batches),
                             (unsigned long long)(stats.parallelPixels / 1000000u), (unsigned long long)(stats.inlinePixels / 1000000u));
                for (int i = 0; i < 6; ++i)
                    std::fprintf(stderr, " %s=%llu", kReasons[i], (unsigned long long)stats.reason[i]);
                std::fprintf(stderr, "; drawing took %.2fs parallel, %.2fs inline\n", stats.parallelSeconds, stats.inlineSeconds);
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
        const auto drawStart = wantStats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const bool parallel = !workers.empty() && pendingPixels >= kParallelPixels;
        if (!parallel)
        {
            for (const PreparedPrimitive &primitive : pending)
                Rasterise(primitive, primitive.minY, primitive.maxY);
        }
        else
        {
            firstRow = top;
            bandCount = (bottom - top) / kBandHeight + 1;
            nextBand.store(0, std::memory_order_relaxed);
            {
                std::lock_guard<std::mutex> lock(mutex);
                active = static_cast<int>(workers.size());
                ++generation;
            }
            wake.notify_all();
            drawBands();
            std::unique_lock<std::mutex> lock(mutex);
            done.wait(lock, [&]
                      { return active == 0; });
        }
        if (wantStats)
        {
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - drawStart).count();
            (parallel ? stats.parallelSeconds : stats.inlineSeconds) += seconds;
        }

        pending.clear();
        pendingPixels = 0;
        written.fill(0u);
        read.fill(0u);
        readLive.fill(0u);
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
    }
};

void GSCpuBackend::RasterQueueDeleter::operator()(RasterQueue *queue) const
{
    delete queue;
}

bool GSCpuBackend::HasQueuedPrimitives() const
{
    return m_rasterQueue && !m_rasterQueue->pending.empty();
}

void GSCpuBackend::FlushPrimitives()
{
    if (m_rasterQueue && !m_rasterQueue->pending.empty())
        m_rasterQueue->run();
}

void GSCpuBackend::FlushPrimitivesTouching(const GSMem::SurfaceLayout &layout, int x0, int y0, int x1, int y1, bool writes,
                                           const char *why)
{
    if (!m_rasterQueue || m_rasterQueue->pending.empty() || !layout.table)
        return;
    // A reader has to wait for pending writes to its pages; a writer also for
    // pending reads of them.
    PageSet pages{};
    addPages(pages, layout, x0, y0, x1, y1);
    if (intersects(pages, m_rasterQueue->written) || (writes && intersects(pages, m_rasterQueue->read)))
        m_rasterQueue->run(why);
}

// Sets `primitive` up for the triangle in `batch`. Returns false when the fast
// path cannot draw it; a primitive that covers nothing comes back empty.
bool GSCpuBackend::PrepareTriangle(const GSPrimitiveBatch &batch, PreparedPrimitive &primitive)
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
        primitive.ownsEdge[i] = wa[i] > 0.0f || (wa[i] == 0.0f && wb[i] > 0.0f);

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
        p.setSampledTexels(lowU, lowV, highU, highV);
    }
    return true;
}

bool GSCpuBackend::PrepareSprite(const GSPrimitiveBatch &batch, PreparedPrimitive &primitive)
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
    p.setSampledTexels(0.0f, 0.0f, 0.0f, 0.0f);

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
    p.setSampledTexels(tu0, tv0, tu1, tv1);
    return true;
}

// Draws rows yFirst..yLast of a prepared primitive. Touches only those rows of
// the targets, so different rows can be drawn on different threads.
void GSCpuBackend::Rasterise(const PreparedPrimitive &primitive, int yFirst, int yLast)
{
    const FastPipeline &p = primitive.pipeline;
    const GSVertex &v0 = primitive.v[0];
    const GSVertex &v1 = primitive.v[1];

    if (primitive.sprite)
    {
        const uint32_t z = static_cast<uint32_t>(v1.z);
        const int r = v1.r, g = v1.g, b = v1.b, a = v1.a;
        const int fog = v1.fog;
        if (!p.tme)
        {
            for (int y = yFirst; y <= yLast; ++y)
                for (int x = primitive.minX; x <= primitive.maxX; ++x)
                    p.write(x, y, z, r, g, b, a, fog);
            return;
        }

        for (int y = yFirst; y <= yLast; ++y)
        {
            const float ty = (static_cast<float>(y - primitive.originY) + 0.5f) / primitive.spanY;
            float texV = primitive.tv0 + (primitive.tv1 - primitive.tv0) * ty;
            if (p.fst)
                texV = static_cast<float>(clampInt(static_cast<int>((texV * 16.0f) + 0.5f), 0, 0xFFFF)) / 16.0f;

            for (int x = primitive.minX; x <= primitive.maxX; ++x)
            {
                const float tx = (static_cast<float>(x - primitive.originX) + 0.5f) / primitive.spanX;
                float texU = primitive.tu0 + (primitive.tu1 - primitive.tu0) * tx;
                if (p.fst)
                    texU = static_cast<float>(clampInt(static_cast<int>((texU * 16.0f) + 0.5f), 0, 0xFFFF)) / 16.0f;

                int pr = r, pg = g, pb = b, pa = a;
                p.combine(p.sample(texU, texV), pr, pg, pb, pa);
                p.write(x, y, z, pr, pg, pb, pa, fog);
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
    const float r0 = v0.r, r1 = v1.r, r2 = v2.r;
    const float g0 = v0.g, g1 = v1.g, g2 = v2.g;
    const float b0 = v0.b, b1 = v1.b, b2 = v2.b;
    const float a0 = v0.a, a1 = v1.a, a2 = v2.a;
    const float f0 = v0.fog, f1 = v1.fog, f2 = v2.fog;
    const float texWf = static_cast<float>(p.texW);
    const float texHf = static_cast<float>(p.texH);
    const float u0 = v0.u, u1 = v1.u, u2 = v2.u;
    const float vv0 = v0.v, vv1 = v1.v, vv2 = v2.v;

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

        for (int x = xStart; x <= xEnd; ++x)
        {
            const float dx = (static_cast<float>(x) + 0.5f) - fx2;
            const float w0 = wa[0] * dx + rowW[0];
            const float w1 = wa[1] * dx + rowW[1];
            const float w2 = 1.0f - w0 - w1;

            const double z = v0.z * w0 + v1.z * w1 + v2.z * w2;

            int r, g, b, a;
            if (gouraud)
            {
                r = clampU8(static_cast<int>(r0 * w0 + r1 * w1 + r2 * w2));
                g = clampU8(static_cast<int>(g0 * w0 + g1 * w1 + g2 * w2));
                b = clampU8(static_cast<int>(b0 * w0 + b1 * w1 + b2 * w2));
                a = clampU8(static_cast<int>(a0 * w0 + a1 * w1 + a2 * w2));
            }
            else
            {
                r = v2.r;
                g = v2.g;
                b = v2.b;
                a = v2.a;
            }

            if (p.tme)
            {
                float texU, texV;
                if (p.fst)
                {
                    texU = static_cast<float>(static_cast<uint16_t>(u0 * w0 + u1 * w1 + u2 * w2)) / 16.0f;
                    texV = static_cast<float>(static_cast<uint16_t>(vv0 * w0 + vv1 * w1 + vv2 * w2)) / 16.0f;
                }
                else
                {
                    // S, T and Q are interpolated; the divide comes after.
                    const float s = v0.s * w0 + v1.s * w1 + v2.s * w2;
                    const float t = v0.t * w0 + v1.t * w1 + v2.t * w2;
                    float q = v0.q * w0 + v1.q * w1 + v2.q * w2;
                    if (!(std::fabs(q) > 1.0e-8f))
                        q = 1.0f;
                    const float invQ = 1.0f / q;
                    texU = s * invQ * texWf;
                    texV = t * invQ * texHf;
                }
                p.combine(p.sample(texU, texV), r, g, b, a);
            }

            const int fog = clampU8(static_cast<int>(f0 * w0 + f1 * w1 + f2 * w2));
            p.write(x, y, static_cast<uint32_t>(z + 0.5), r, g, b, a, fog);
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
        queue.run("target");
    }

    // Its texture has to be read after everything that draws into it.
    if (p.tme)
        FlushPrimitivesTouching(p.texture, p.sampledU0, p.sampledV0, p.sampledU1, p.sampledV1, false, "texture");

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
        queue.run("texture");

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
    if (p.tme)
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

    if (queue.pending.size() >= RasterQueue::kMaxPending)
        queue.run("full");
    return true;
}
