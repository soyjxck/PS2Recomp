#ifndef PS2_VU1_FMAC_SSE_H
#define PS2_VU1_FMAC_SSE_H

#include "ps2_vu1_detail.h"

// The x86 twin of ps2_vu1_fmac_neon.h: the same lane tests and the same single
// rounding of a product sum, so the two give the same results and flags. It
// needs FMA (part of every AVX2 part; MSVC's /arch:AVX2 enables it); other
// translation units keep the scalar path.
#if !defined(__aarch64__) && (defined(__FMA__) || (defined(_MSC_VER) && defined(__AVX2__)))
#include <immintrin.h>
#include <cstdint>
#include <cstring>

#define PS2_VU_FMAC_VECTOR 1

namespace ps2_vu_detail {

namespace sse_detail {

PS2_VU_FORCE_INLINE __m128i select(__m128i mask, __m128i yes, __m128i no)
{
    return _mm_or_si128(_mm_and_si128(mask, yes), _mm_andnot_si128(mask, no));
}

// Everything compared below is small and non-negative, so the signed compares
// are the unsigned ones.
PS2_VU_FORCE_INLINE __m128i lessOrEqual(__m128i left, __m128i right)
{
    return _mm_or_si128(_mm_cmplt_epi32(left, right), _mm_cmpeq_epi32(left, right));
}

PS2_VU_FORCE_INLINE __m128i greaterOrEqual(__m128i left, __m128i right)
{
    return _mm_or_si128(_mm_cmpgt_epi32(left, right), _mm_cmpeq_epi32(left, right));
}

// Lanes x, y, z, w of a destination mask (bit 3 is x) as all-ones masks.
PS2_VU_FORCE_INLINE __m128i activeLanes(uint8_t destination)
{
    const __m128i laneBits = _mm_setr_epi32(8, 4, 2, 1);
    return _mm_cmpeq_epi32(_mm_and_si128(_mm_set1_epi32(destination), laneBits), laneBits);
}

// True when some written lane failed its exact-result test.
PS2_VU_FORCE_INLINE bool anyActiveUnsafe(__m128i safe, __m128i active)
{
    const __m128i ok = _mm_or_si128(safe, _mm_andnot_si128(active, _mm_set1_epi32(-1)));
    return _mm_movemask_ps(_mm_castsi128_ps(ok)) != 0xF;
}

// The low byte of each lane, in lane order, as the four lane flags.
PS2_VU_FORCE_INLINE void storeLaneFlags(__m128i flags, uint8_t laneFlags[4])
{
    const __m128i none = _mm_setzero_si128();
    const __m128i packed = _mm_packus_epi16(_mm_packs_epi32(flags, none), none);
    const uint32_t bytes = static_cast<uint32_t>(_mm_cvtsi128_si32(packed));
    std::memcpy(laneFlags, &bytes, sizeof(bytes));
}

}

PS2_VU_FORCE_INLINE __m128 normalizeFmacOperands(__m128 value)
{
    const __m128i bits = _mm_castps_si128(value);
    const __m128i sign = _mm_and_si128(bits, _mm_set1_epi32(static_cast<int>(0x80000000u)));
    const __m128i exponent = _mm_and_si128(bits, _mm_set1_epi32(0x7f800000));
    const __m128i zero = _mm_cmpeq_epi32(exponent, _mm_setzero_si128());
    const __m128i special = _mm_cmpeq_epi32(exponent, _mm_set1_epi32(0x7f800000));
    const __m128i finite = sse_detail::select(special, _mm_or_si128(sign, _mm_set1_epi32(0x7f7fffff)), bits);
    return _mm_castsi128_ps(sse_detail::select(zero, sign, finite));
}

enum class SimpleArithmetic { Add, Subtract, Multiply };

template <SimpleArithmetic operation>
PS2_VU_FORCE_INLINE bool trySimpleArithmeticVector(
    const float *leftInput, const float *rightInput, uint8_t destination,
    float *result, uint8_t laneFlags[4])
{
    const __m128 left = normalizeFmacOperands(_mm_loadu_ps(leftInput));
    const __m128 right = normalizeFmacOperands(_mm_loadu_ps(rightInput));
    __m128 value;
    if constexpr (operation == SimpleArithmetic::Add)
        value = _mm_add_ps(left, right);
    else if constexpr (operation == SimpleArithmetic::Subtract)
        value = _mm_sub_ps(left, right);
    else
        value = _mm_mul_ps(left, right);
    const __m128i bits = _mm_castps_si128(value);
    const __m128i magnitudeMask = _mm_set1_epi32(0x7fffffff);
    const __m128i magnitude = _mm_and_si128(bits, magnitudeMask);
    const __m128i leftBits = _mm_castps_si128(left);
    const __m128i rightBits = _mm_castps_si128(right);
    const __m128i none = _mm_setzero_si128();
    const __m128i leftZero = _mm_cmpeq_epi32(_mm_and_si128(leftBits, magnitudeMask), none);
    const __m128i rightZero = _mm_cmpeq_epi32(_mm_and_si128(rightBits, magnitudeMask), none);
    __m128i exactZero;
    if constexpr (operation == SimpleArithmetic::Multiply)
        exactZero = _mm_or_si128(leftZero, rightZero);
    else
    {
        const __m128i opposing = operation == SimpleArithmetic::Add
            ? _mm_xor_si128(rightBits, _mm_set1_epi32(static_cast<int>(0x80000000u))) : rightBits;
        exactZero = _mm_or_si128(_mm_and_si128(leftZero, rightZero), _mm_cmpeq_epi32(leftBits, opposing));
    }
    const __m128i zero = _mm_cmpeq_epi32(magnitude, none);
    const __m128i normal = _mm_and_si128(_mm_cmpgt_epi32(magnitude, _mm_set1_epi32(0x00800000)),
                                         _mm_cmplt_epi32(magnitude, _mm_set1_epi32(0x7f7fffff)));
    const __m128i active = sse_detail::activeLanes(destination);
    // Boundary results retain the scalar exact-result checks, including
    // tiny exact results rounded to zero/MIN_NORMAL and overflow to FLT_MAX.
    const __m128i safe = _mm_or_si128(normal, _mm_and_si128(zero, exactZero));
    if (sse_detail::anyActiveUnsafe(safe, active))
        return false;
    const __m128i flags = _mm_and_si128(_mm_or_si128(_mm_slli_epi32(_mm_srli_epi32(bits, 31), 1),
                                                     _mm_and_si128(zero, _mm_set1_epi32(1))), active);
    sse_detail::storeLaneFlags(flags, laneFlags);
    _mm_storeu_ps(result, value);
    return true;
}

template <bool subtract>
PS2_VU_FORCE_INLINE bool tryProductSumVector(
    const float *leftInput, const float *rightInput, const float *accInput,
    uint8_t destination, float *result, uint8_t laneFlags[4], uint32_t &extraSticky)
{
    const __m128 left = normalizeFmacOperands(_mm_loadu_ps(leftInput));
    const __m128 right = normalizeFmacOperands(_mm_loadu_ps(rightInput));
    const __m128 acc = normalizeFmacOperands(_mm_loadu_ps(accInput));
    const __m128 value = subtract ? _mm_fnmadd_ps(left, right, acc) : _mm_fmadd_ps(left, right, acc);
    const __m128i valueBits = _mm_castps_si128(value);
    const __m128i leftBits = _mm_castps_si128(left);
    const __m128i rightBits = _mm_castps_si128(right);
    const __m128i exponentMask = _mm_set1_epi32(0xff);
    const __m128i resultExponent = _mm_and_si128(_mm_srli_epi32(valueBits, 23), exponentMask);
    const __m128i leftExponent = _mm_and_si128(_mm_srli_epi32(leftBits, 23), exponentMask);
    const __m128i rightExponent = _mm_and_si128(_mm_srli_epi32(rightBits, 23), exponentMask);
    const __m128i productBound = _mm_add_epi32(leftExponent, rightExponent);
    const __m128i none = _mm_setzero_si128();
    const __m128i zeroProduct = _mm_or_si128(_mm_cmpeq_epi32(leftExponent, none),
                                             _mm_cmpeq_epi32(rightExponent, none));
    const __m128i productSafe = _mm_and_si128(
        sse_detail::lessOrEqual(productBound, _mm_set1_epi32(379)),
        sse_detail::lessOrEqual(productBound, _mm_add_epi32(resultExponent, _mm_set1_epi32(146))));
    const __m128i resultSafe = _mm_and_si128(sse_detail::greaterOrEqual(resultExponent, _mm_set1_epi32(2)),
                                             sse_detail::lessOrEqual(resultExponent, _mm_set1_epi32(253)));
    const __m128i magnitudeMask = _mm_set1_epi32(0x7fffffff);
    const __m128i zeroResult = _mm_cmpeq_epi32(_mm_and_si128(valueBits, magnitudeMask), none);
    const __m128i active = sse_detail::activeLanes(destination);
    const __m128i safe = _mm_or_si128(_mm_and_si128(resultSafe, _mm_or_si128(zeroProduct, productSafe)),
                                      _mm_and_si128(zeroResult, zeroProduct));
    if (sse_detail::anyActiveUnsafe(safe, active))
        return false;

    // The accepted exponent bound excludes product overflow. A tiny product
    // is exact zero only when one normalized multiplicand is zero.
    const __m128 product = _mm_mul_ps(left, right);
    const __m128i productMagnitude = _mm_and_si128(_mm_castps_si128(product), magnitudeMask);
    const __m128i tiny = _mm_cmplt_epi32(productMagnitude, _mm_set1_epi32(0x00800000));
    const __m128i sign = _mm_slli_epi32(_mm_srli_epi32(_mm_xor_si128(leftBits, rightBits), 31), 1);
    const __m128i zero = _mm_and_si128(tiny, _mm_set1_epi32(1));
    const __m128i underflow = _mm_and_si128(_mm_andnot_si128(zeroProduct, tiny), _mm_set1_epi32(4));
    const __m128i sticky = _mm_and_si128(_mm_or_si128(sign, _mm_or_si128(zero, underflow)), active);
    __m128i folded = _mm_or_si128(sticky, _mm_srli_si128(sticky, 8));
    folded = _mm_or_si128(folded, _mm_srli_si128(folded, 4));
    extraSticky = static_cast<uint32_t>(_mm_cvtsi128_si32(folded));

    const __m128i current = _mm_and_si128(_mm_or_si128(_mm_slli_epi32(_mm_srli_epi32(valueBits, 31), 1),
                                                       _mm_and_si128(zeroResult, _mm_set1_epi32(1))), active);
    sse_detail::storeLaneFlags(current, laneFlags);
    _mm_storeu_ps(result, value);
    return true;
}

}
#endif

#ifndef PS2_VU_FMAC_VECTOR
#define PS2_VU_FMAC_VECTOR 0
#endif

#endif
