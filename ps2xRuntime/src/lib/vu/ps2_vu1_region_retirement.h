#ifndef PS2_VU1_REGION_RETIREMENT_H
#define PS2_VU1_REGION_RETIREMENT_H

#include <cstdint>
#include <cstring>
#if defined(__aarch64__)
#include <arm_neon.h>
#elif !defined(__aarch64__) && (defined(__FMA__) || (defined(_MSC_VER) && defined(__AVX2__)))
#include <immintrin.h>
#endif

namespace ps2_vu_detail {

inline void retireRegionVector(float *destination, const float *value,
                               const uint64_t *latest, uint64_t sequence,
                               uint8_t laneMask, const uint8_t *firstWrite,
                               uint32_t readyOffset)
{
#if defined(__aarch64__)
    const auto expected = vdupq_n_u64(sequence);
    const auto matches = vcombine_u32(vmovn_u64(vceqq_u64(vld1q_u64(latest), expected)),
                                     vmovn_u64(vceqq_u64(vld1q_u64(latest + 2), expected)));
    uint32_t offsets;
    std::memcpy(&offsets, firstWrite, sizeof(offsets));
    const auto expanded = vmovl_u16(vget_low_u16(vmovl_u8(vcreate_u8(offsets))));
    const auto visible = vcgeq_u32(expanded, vdupq_n_u32(readyOffset));
    const uint32x4_t laneBits = {8u, 4u, 2u, 1u};
    const auto lanes = vtstq_u32(vdupq_n_u32(laneMask), laneBits);
    const auto mask = vandq_u32(vandq_u32(matches, visible), lanes);
    vst1q_f32(destination, vbslq_f32(mask, vld1q_f32(value), vld1q_f32(destination)));
#elif !defined(__aarch64__) && (defined(__FMA__) || (defined(_MSC_VER) && defined(__AVX2__)))
    const __m128i expected = _mm_set1_epi64x(static_cast<long long>(sequence));
    const __m128i low = _mm_cmpeq_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i *>(latest)), expected);
    const __m128i high = _mm_cmpeq_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i *>(latest + 2)), expected);
    const __m128i matches = _mm_castps_si128(
        _mm_shuffle_ps(_mm_castsi128_ps(low), _mm_castsi128_ps(high), _MM_SHUFFLE(2, 0, 2, 0)));
    uint32_t offsets;
    std::memcpy(&offsets, firstWrite, sizeof(offsets));
    const __m128i expanded = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(static_cast<int>(offsets)));
    const __m128i ready = _mm_set1_epi32(static_cast<int>(readyOffset));
    const __m128i visible = _mm_or_si128(_mm_cmpgt_epi32(expanded, ready), _mm_cmpeq_epi32(expanded, ready));
    const __m128i laneBits = _mm_setr_epi32(8, 4, 2, 1);
    const __m128i lanes = _mm_cmpeq_epi32(_mm_and_si128(_mm_set1_epi32(laneMask), laneBits), laneBits);
    const __m128i mask = _mm_and_si128(_mm_and_si128(matches, visible), lanes);
    _mm_storeu_ps(destination, _mm_blendv_ps(_mm_loadu_ps(destination), _mm_loadu_ps(value), _mm_castsi128_ps(mask)));
#else
    for (unsigned lane = 0; lane < 4; ++lane)
        if ((laneMask & (8u >> lane)) != 0u && latest[lane] == sequence &&
            readyOffset <= firstWrite[lane])
            destination[lane] = value[lane];
#endif
}

}
#endif
