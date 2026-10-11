#ifndef PS2_GS_COMMON_H
#define PS2_GS_COMMON_H

#include "runtime/gs/gs_types.h"
#include <cstdint>

namespace GSInternal
{
static inline uint32_t bitsPerPixel(uint8_t psm)
{
    switch (psm)
    {
    case GS_PSM_CT32:
    case GS_PSM_Z32:
        return 32;
    case GS_PSM_CT24:
    case GS_PSM_Z24:
        return 32;
    case GS_PSM_CT16:
    case GS_PSM_CT16S:
    case GS_PSM_Z16:
    case GS_PSM_Z16S:
        return 16;
    case GS_PSM_T8:
    case GS_PSM_T8H:
        return 8;
    case GS_PSM_T4:
    case GS_PSM_T4HL:
    case GS_PSM_T4HH:
        return 4;
    default:
        return 32;
    }
}

static inline uint32_t fbStride(uint32_t fbw, uint8_t psm)
{
    uint32_t pixelsPerRow = fbw * 64u;
    return pixelsPerRow * (bitsPerPixel(psm) / 8u);
}

static inline uint32_t framePageBaseToBlock(uint32_t fbp)
{
    return fbp << 5u;
}

static inline int clampInt(int v, int lo, int hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static inline uint8_t clampU8(int v)
{
    if (v < 0) return 0;
    if (v > 255) return 255;
    return static_cast<uint8_t>(v);
}

// The first whole pixel at or after a sprite corner: a vertex coordinate (in
// pixels, with its sixteenths) less the drawing offset (in sixteenths). The
// GS draws a sprite from that pixel at its smaller corner up to, not
// including, that pixel at its larger one. A strip from 0 to 63.5 fills 64
// columns; one from 0.5 to 64 leaves column 0 alone.
static inline int gsSpriteEdge(float vertexCoordinate, uint32_t offsetSixteenths)
{
    const int sixteenths = static_cast<int>(vertexCoordinate * 16.0f + 0.5f) - static_cast<int>(offsetSixteenths);
    return (sixteenths + 15) >> 4;
}
}

#endif
