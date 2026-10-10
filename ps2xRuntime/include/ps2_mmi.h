#ifndef PS2_MMI_H
#define PS2_MMI_H

// R5900 multimedia instructions that are more than a single host intrinsic:
// the ones that go through the 128-bit HI/LO pair, and the lane permutations.
// Semantics follow the EE Core instruction manual. Lanes are numbered from the
// least significant end: word 0 is bits 0-31, halfword 0 is bits 0-15.
//
// HI and LO are 128 bits wide. R5900Context keeps bits 0-63 in hi/lo and bits
// 64-127 in hi1/lo1 (the registers MULT1/DIV1 use).
//
// Included by ps2_runtime_macros.h.

#include "ps2_runtime.h"

#include <cstdint>
#include <cstring>

namespace ps2mmi
{
    struct Words
    {
        uint32_t w[4];
    };

    struct Halves
    {
        uint16_t h[8];
    };

    inline Words words(__m128i value)
    {
        Words out;
        _mm_storeu_si128(reinterpret_cast<__m128i *>(out.w), value);
        return out;
    }

    inline Halves halves(__m128i value)
    {
        Halves out;
        _mm_storeu_si128(reinterpret_cast<__m128i *>(out.h), value);
        return out;
    }

    inline __m128i vec(const Words &value)
    {
        return _mm_loadu_si128(reinterpret_cast<const __m128i *>(value.w));
    }

    inline __m128i vec(const Halves &value)
    {
        return _mm_loadu_si128(reinterpret_cast<const __m128i *>(value.h));
    }

    inline __m128i vec(uint64_t low, uint64_t high)
    {
        return _mm_set_epi64x(static_cast<long long>(high), static_cast<long long>(low));
    }

    inline Words loWords(const R5900Context *ctx)
    {
        Words out;
        std::memcpy(out.w, &ctx->lo, 8);
        std::memcpy(out.w + 2, &ctx->lo1, 8);
        return out;
    }

    inline Words hiWords(const R5900Context *ctx)
    {
        Words out;
        std::memcpy(out.w, &ctx->hi, 8);
        std::memcpy(out.w + 2, &ctx->hi1, 8);
        return out;
    }

    inline void setLo(R5900Context *ctx, const Words &value)
    {
        std::memcpy(&ctx->lo, value.w, 8);
        std::memcpy(&ctx->lo1, value.w + 2, 8);
    }

    inline void setHi(R5900Context *ctx, const Words &value)
    {
        std::memcpy(&ctx->hi, value.w, 8);
        std::memcpy(&ctx->hi1, value.w + 2, 8);
    }

    inline uint64_t signExtend32(uint32_t value)
    {
        return static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(value)));
    }

    // PMFHI, PMFLO, PMTHI, PMTLO: the whole 128-bit register.
    inline __m128i pmfhi(const R5900Context *ctx) { return vec(ctx->hi, ctx->hi1); }
    inline __m128i pmflo(const R5900Context *ctx) { return vec(ctx->lo, ctx->lo1); }

    inline void pmthi(R5900Context *ctx, __m128i rs)
    {
        uint64_t parts[2];
        _mm_storeu_si128(reinterpret_cast<__m128i *>(parts), rs);
        ctx->hi = parts[0];
        ctx->hi1 = parts[1];
    }

    inline void pmtlo(R5900Context *ctx, __m128i rs)
    {
        uint64_t parts[2];
        _mm_storeu_si128(reinterpret_cast<__m128i *>(parts), rs);
        ctx->lo = parts[0];
        ctx->lo1 = parts[1];
    }

    // PMULTW/PMADDW/PMSUBW and PMULTUW/PMADDUW: 32x32 products of words 0 and
    // 2. Each 64-bit result goes to rd's doubleword; its halves, sign extended,
    // go to the matching doublewords of LO and HI. `accumulate` is 0 to store
    // the product, +1 to add it to HI:LO, -1 to subtract it.
    inline __m128i multiplyWords(R5900Context *ctx, __m128i rs, __m128i rt, bool isSigned, int accumulate)
    {
        const Words a = words(rs);
        const Words b = words(rt);
        uint64_t lo[2] = {ctx->lo, ctx->lo1};
        uint64_t hi[2] = {ctx->hi, ctx->hi1};
        uint64_t result[2];
        for (int lane = 0; lane < 2; ++lane)
        {
            const uint32_t x = a.w[lane * 2];
            const uint32_t y = b.w[lane * 2];
            const uint64_t product = isSigned
                                         ? static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(x)) * static_cast<int64_t>(static_cast<int32_t>(y)))
                                         : static_cast<uint64_t>(x) * static_cast<uint64_t>(y);
            const uint64_t acc = ((hi[lane] & 0xFFFFFFFFull) << 32) | (lo[lane] & 0xFFFFFFFFull);
            const uint64_t value = accumulate > 0 ? acc + product : (accumulate < 0 ? acc - product : product);
            lo[lane] = signExtend32(static_cast<uint32_t>(value));
            hi[lane] = signExtend32(static_cast<uint32_t>(value >> 32));
            result[lane] = value;
        }
        ctx->lo = lo[0];
        ctx->lo1 = lo[1];
        ctx->hi = hi[0];
        ctx->hi1 = hi[1];
        return vec(result[0], result[1]);
    }

    // PDIVW / PDIVUW: words 0 and 2; quotients to LO, remainders to HI.
    inline void divideWords(R5900Context *ctx, __m128i rs, __m128i rt, bool isSigned)
    {
        const Words a = words(rs);
        const Words b = words(rt);
        uint64_t lo[2];
        uint64_t hi[2];
        for (int lane = 0; lane < 2; ++lane)
        {
            const uint32_t x = a.w[lane * 2];
            const uint32_t y = b.w[lane * 2];
            if (!isSigned)
            {
                if (y != 0u)
                {
                    lo[lane] = signExtend32(x / y);
                    hi[lane] = signExtend32(x % y);
                }
                else
                {
                    lo[lane] = ~0ull;
                    hi[lane] = signExtend32(x);
                }
                continue;
            }

            const int32_t sx = static_cast<int32_t>(x);
            const int32_t sy = static_cast<int32_t>(y);
            if (x == 0x80000000u && y == 0xFFFFFFFFu)
            {
                lo[lane] = signExtend32(0x80000000u);
                hi[lane] = 0u;
            }
            else if (sy != 0)
            {
                lo[lane] = signExtend32(static_cast<uint32_t>(sx / sy));
                hi[lane] = signExtend32(static_cast<uint32_t>(sx % sy));
            }
            else
            {
                lo[lane] = sx < 0 ? 1ull : ~0ull;
                hi[lane] = signExtend32(x);
            }
        }
        ctx->lo = lo[0];
        ctx->lo1 = lo[1];
        ctx->hi = hi[0];
        ctx->hi1 = hi[1];
    }

    // PDIVBW: all four words of rs divided by halfword 0 of rt.
    inline void pdivbw(R5900Context *ctx, __m128i rs, __m128i rt)
    {
        const Words a = words(rs);
        const int32_t divisor = static_cast<int16_t>(halves(rt).h[0]);
        Words lo;
        Words hi;
        for (int i = 0; i < 4; ++i)
        {
            const int32_t x = static_cast<int32_t>(a.w[i]);
            if (a.w[i] == 0x80000000u && divisor == -1)
            {
                lo.w[i] = 0x80000000u;
                hi.w[i] = 0u;
            }
            else if (divisor != 0)
            {
                lo.w[i] = static_cast<uint32_t>(x / divisor);
                hi.w[i] = static_cast<uint32_t>(x % divisor);
            }
            else
            {
                lo.w[i] = x < 0 ? 1u : 0xFFFFFFFFu;
                hi.w[i] = a.w[i];
            }
        }
        setLo(ctx, lo);
        setHi(ctx, hi);
    }

    // PMULTH/PMADDH/PMSUBH: eight 16x16 products. Products 0, 1, 4, 5 go to
    // LO words 0-3 and products 2, 3, 6, 7 to HI words 0-3; rd collects the
    // even ones (LO.w0, HI.w0, LO.w2, HI.w2).
    inline __m128i multiplyHalves(R5900Context *ctx, __m128i rs, __m128i rt, int accumulate)
    {
        const Halves a = halves(rs);
        const Halves b = halves(rt);
        Words lo = loWords(ctx);
        Words hi = hiWords(ctx);
        for (int i = 0; i < 8; ++i)
        {
            const uint32_t product = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(a.h[i])) *
                                                           static_cast<int32_t>(static_cast<int16_t>(b.h[i])));
            uint32_t &slot = ((i & 2) != 0 ? hi : lo).w[((i & 4) != 0 ? 2 : 0) + (i & 1)];
            slot = accumulate > 0 ? slot + product : (accumulate < 0 ? slot - product : product);
        }
        setLo(ctx, lo);
        setHi(ctx, hi);
        return vec(Words{{lo.w[0], hi.w[0], lo.w[2], hi.w[2]}});
    }

    // PHMADH/PHMSBH: for each pair of halfwords, the product of the odd ones
    // plus (or minus) the product of the even ones.
    inline __m128i multiplyHalfPairs(R5900Context *ctx, __m128i rs, __m128i rt, bool subtract)
    {
        const Halves a = halves(rs);
        const Halves b = halves(rt);
        Words lo;
        Words hi;
        for (int pair = 0; pair < 4; ++pair)
        {
            const uint32_t odd = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(a.h[pair * 2 + 1])) *
                                                       static_cast<int32_t>(static_cast<int16_t>(b.h[pair * 2 + 1])));
            const uint32_t even = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(a.h[pair * 2])) *
                                                        static_cast<int32_t>(static_cast<int16_t>(b.h[pair * 2])));
            Words &target = (pair & 1) != 0 ? hi : lo;
            const int base = (pair & 2) != 0 ? 2 : 0;
            target.w[base] = subtract ? odd - even : odd + even;
            target.w[base + 1] = subtract ? ~odd : odd; // undefined on hardware
        }
        setLo(ctx, lo);
        setHi(ctx, hi);
        return vec(Words{{lo.w[0], hi.w[0], lo.w[2], hi.w[2]}});
    }

    // PSLLVW/PSRLVW/PSRAVW: words 0 and 2 of rt shifted by the low five bits
    // of the same words of rs; each result is sign extended to a doubleword.
    inline __m128i shiftWordsVariable(__m128i rs, __m128i rt, int kind)
    {
        const Words amount = words(rs);
        const Words value = words(rt);
        uint64_t result[2];
        for (int lane = 0; lane < 2; ++lane)
        {
            const uint32_t x = value.w[lane * 2];
            const uint32_t count = amount.w[lane * 2] & 0x1Fu;
            const uint32_t shifted = kind == 0 ? x << count
                                               : (kind == 1 ? x >> count
                                                            : static_cast<uint32_t>(static_cast<int32_t>(x) >> count));
            result[lane] = signExtend32(shifted);
        }
        return vec(result[0], result[1]);
    }
    inline __m128i psllvw(__m128i rs, __m128i rt) { return shiftWordsVariable(rs, rt, 0); }
    inline __m128i psrlvw(__m128i rs, __m128i rt) { return shiftWordsVariable(rs, rt, 1); }
    inline __m128i psravw(__m128i rs, __m128i rt) { return shiftWordsVariable(rs, rt, 2); }

    // Lane permutations.
    // PINTH: low halfwords of rt interleaved with the high halfwords of rs.
    inline __m128i pinth(__m128i rs, __m128i rt)
    {
        return _mm_unpacklo_epi16(rt, _mm_srli_si128(rs, 8));
    }

    // PINTEH: even halfwords of rt (even slots) and of rs (odd slots).
    inline __m128i pinteh(__m128i rs, __m128i rt)
    {
        return _mm_or_si128(_mm_and_si128(rt, _mm_set1_epi32(0x0000FFFF)), _mm_slli_epi32(rs, 16));
    }

    // PEXEH: halfwords 0 and 2 of each doubleword swapped.
    inline __m128i pexeh(__m128i rt)
    {
        return _mm_shufflehi_epi16(_mm_shufflelo_epi16(rt, _MM_SHUFFLE(3, 0, 1, 2)), _MM_SHUFFLE(3, 0, 1, 2));
    }

    // PEXCH: halfwords 1 and 2 of each doubleword swapped.
    inline __m128i pexch(__m128i rt)
    {
        return _mm_shufflehi_epi16(_mm_shufflelo_epi16(rt, _MM_SHUFFLE(3, 1, 2, 0)), _MM_SHUFFLE(3, 1, 2, 0));
    }

    // PREVH: halfwords reversed within each doubleword.
    inline __m128i prevh(__m128i rt)
    {
        return _mm_shufflehi_epi16(_mm_shufflelo_epi16(rt, _MM_SHUFFLE(0, 1, 2, 3)), _MM_SHUFFLE(0, 1, 2, 3));
    }

    // PEXEW: words 0 and 2 swapped. PEXCW: words 1 and 2 swapped.
    // PROT3W: words 0-2 rotated down (w0 <- w1 <- w2 <- w0), word 3 kept.
    inline __m128i pexew(__m128i rt) { return _mm_shuffle_epi32(rt, _MM_SHUFFLE(3, 0, 1, 2)); }
    inline __m128i pexcw(__m128i rt) { return _mm_shuffle_epi32(rt, _MM_SHUFFLE(3, 1, 2, 0)); }
    inline __m128i prot3w(__m128i rt) { return _mm_shuffle_epi32(rt, _MM_SHUFFLE(3, 0, 2, 1)); }

    // PABSW/PABSH: the most negative value saturates to the most positive.
    inline __m128i pabsw(__m128i rt)
    {
        const __m128i magnitude = _mm_abs_epi32(rt);
        return _mm_sub_epi32(magnitude, _mm_srli_epi32(magnitude, 31));
    }

    inline __m128i pabsh(__m128i rt)
    {
        const __m128i magnitude = _mm_abs_epi16(rt);
        return _mm_sub_epi16(magnitude, _mm_srli_epi16(magnitude, 15));
    }

    // PMFHL: HI and LO, interleaved, into rd.
    inline __m128i pmfhlLw(const R5900Context *ctx)
    {
        const Words lo = loWords(ctx);
        const Words hi = hiWords(ctx);
        return vec(Words{{lo.w[0], hi.w[0], lo.w[2], hi.w[2]}});
    }

    inline __m128i pmfhlUw(const R5900Context *ctx)
    {
        const Words lo = loWords(ctx);
        const Words hi = hiWords(ctx);
        return vec(Words{{lo.w[1], hi.w[1], lo.w[3], hi.w[3]}});
    }

    inline __m128i pmfhlSlw(const R5900Context *ctx)
    {
        const Words lo = loWords(ctx);
        const Words hi = hiWords(ctx);
        uint64_t result[2];
        for (int lane = 0; lane < 2; ++lane)
        {
            const int64_t value = static_cast<int64_t>((static_cast<uint64_t>(hi.w[lane * 2]) << 32) | lo.w[lane * 2]);
            if (value >= 0x7FFFFFFFll)
                result[lane] = 0x7FFFFFFFull;
            else if (value <= -0x80000000ll)
                result[lane] = 0xFFFFFFFF80000000ull;
            else
                result[lane] = signExtend32(lo.w[lane * 2]);
        }
        return vec(result[0], result[1]);
    }

    inline __m128i pmfhlLh(const R5900Context *ctx)
    {
        const Halves lo = halves(vec(loWords(ctx)));
        const Halves hi = halves(vec(hiWords(ctx)));
        return vec(Halves{{lo.h[0], lo.h[2], hi.h[0], hi.h[2], lo.h[4], lo.h[6], hi.h[4], hi.h[6]}});
    }

    inline uint16_t saturateToHalf(uint32_t value)
    {
        const int32_t signedValue = static_cast<int32_t>(value);
        if (signedValue > 0x7FFF)
            return 0x7FFFu;
        if (signedValue < -0x8000)
            return 0x8000u;
        return static_cast<uint16_t>(value);
    }

    inline __m128i pmfhlSh(const R5900Context *ctx)
    {
        const Words lo = loWords(ctx);
        const Words hi = hiWords(ctx);
        return vec(Halves{{saturateToHalf(lo.w[0]), saturateToHalf(lo.w[1]), saturateToHalf(hi.w[0]), saturateToHalf(hi.w[1]),
                           saturateToHalf(lo.w[2]), saturateToHalf(lo.w[3]), saturateToHalf(hi.w[2]), saturateToHalf(hi.w[3])}});
    }

    // PMTHL.LW: the inverse of PMFHL.LW; the odd words of HI and LO are kept.
    inline void pmthlLw(R5900Context *ctx, __m128i rs)
    {
        const Words value = words(rs);
        Words lo = loWords(ctx);
        Words hi = hiWords(ctx);
        lo.w[0] = value.w[0];
        hi.w[0] = value.w[1];
        lo.w[2] = value.w[2];
        hi.w[2] = value.w[3];
        setLo(ctx, lo);
        setHi(ctx, hi);
    }
}

#endif // PS2_MMI_H
