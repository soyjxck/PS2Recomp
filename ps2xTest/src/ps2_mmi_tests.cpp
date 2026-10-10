#include "MiniTest.h"
#include "ps2_runtime_macros.h"

#include <cfloat>
#include <cstdint>
#include <cstring>

// Lane-level checks of the MMI helpers and of the VU0/FPU arithmetic helpers
// that generated code calls. Expected values are written out from the EE Core
// instruction manual's definitions, lane 0 first.

namespace
{
    __m128i H(uint16_t a, uint16_t b, uint16_t c, uint16_t d, uint16_t e, uint16_t f, uint16_t g, uint16_t h)
    {
        return ps2mmi::vec(ps2mmi::Halves{{a, b, c, d, e, f, g, h}});
    }

    __m128i W(uint32_t a, uint32_t b, uint32_t c, uint32_t d)
    {
        return ps2mmi::vec(ps2mmi::Words{{a, b, c, d}});
    }

    bool eq(__m128i a, __m128i b)
    {
        return _mm_movemask_epi8(_mm_cmpeq_epi8(a, b)) == 0xFFFF;
    }
}

void register_ps2_mmi_tests()
{
    MiniTest::Case("PS2Mmi", [](TestCase &tc)
                   {
    tc.Run("MMI, VU0 macro and FPU helpers match the instruction manual", [](TestCase &t) {
        const __m128i rt = H(0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7);
        const __m128i rs = H(0xB0, 0xB1, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7);
        t.IsTrue(eq(ps2mmi::pinth(rs, rt), H(0xA0, 0xB4, 0xA1, 0xB5, 0xA2, 0xB6, 0xA3, 0xB7)), "eq(ps2mmi::pinth(rs, rt), H(0xA0, 0xB4, 0xA1, 0xB5, 0xA2, 0xB6, 0xA3, 0xB7))");
        t.IsTrue(eq(ps2mmi::pinteh(rs, rt), H(0xA0, 0xB0, 0xA2, 0xB2, 0xA4, 0xB4, 0xA6, 0xB6)), "eq(ps2mmi::pinteh(rs, rt), H(0xA0, 0xB0, 0xA2, 0xB2, 0xA4, 0xB4, 0xA6, 0xB6))");
        t.IsTrue(eq(ps2mmi::pexeh(rt), H(0xA2, 0xA1, 0xA0, 0xA3, 0xA6, 0xA5, 0xA4, 0xA7)), "eq(ps2mmi::pexeh(rt), H(0xA2, 0xA1, 0xA0, 0xA3, 0xA6, 0xA5, 0xA4, 0xA7))");
        t.IsTrue(eq(ps2mmi::pexch(rt), H(0xA0, 0xA2, 0xA1, 0xA3, 0xA4, 0xA6, 0xA5, 0xA7)), "eq(ps2mmi::pexch(rt), H(0xA0, 0xA2, 0xA1, 0xA3, 0xA4, 0xA6, 0xA5, 0xA7))");
        t.IsTrue(eq(ps2mmi::prevh(rt), H(0xA3, 0xA2, 0xA1, 0xA0, 0xA7, 0xA6, 0xA5, 0xA4)), "eq(ps2mmi::prevh(rt), H(0xA3, 0xA2, 0xA1, 0xA0, 0xA7, 0xA6, 0xA5, 0xA4))");
        const __m128i w = W(0x100, 0x101, 0x102, 0x103);
        t.IsTrue(eq(ps2mmi::pexew(w), W(0x102, 0x101, 0x100, 0x103)), "eq(ps2mmi::pexew(w), W(0x102, 0x101, 0x100, 0x103))");
        t.IsTrue(eq(ps2mmi::pexcw(w), W(0x100, 0x102, 0x101, 0x103)), "eq(ps2mmi::pexcw(w), W(0x100, 0x102, 0x101, 0x103))");
        t.IsTrue(eq(ps2mmi::prot3w(w), W(0x101, 0x102, 0x100, 0x103)), "eq(ps2mmi::prot3w(w), W(0x101, 0x102, 0x100, 0x103))");
        t.IsTrue(eq(ps2mmi::pabsw(W(0x80000000u, 0xFFFFFFFFu, 5, 0)), W(0x7FFFFFFFu, 1, 5, 0)), "eq(ps2mmi::pabsw(W(0x80000000u, 0xFFFFFFFFu, 5, 0)), W(0x7FFFFFFFu, 1, 5, 0))");
        t.IsTrue(eq(ps2mmi::pabsh(H(0x8000, 0xFFFF, 5, 0, 0x8001, 1, 2, 3)), H(0x7FFF, 1, 5, 0, 0x7FFF, 1, 2, 3)), "eq(ps2mmi::pabsh(H(0x8000, 0xFFFF, 5, 0, 0x8001, 1, 2, 3)), H(0x7FFF, 1, 5, 0, 0x7FFF, 1, 2, 3))");

        // Existing macros the generator still uses.
        t.IsTrue(eq(PS2_PEXTLW(W(0xB0, 0xB1, 0xB2, 0xB3), W(0xA0, 0xA1, 0xA2, 0xA3)), W(0xA0, 0xB0, 0xA1, 0xB1)), "eq(PS2_PEXTLW(W(0xB0, 0xB1, 0xB2, 0xB3), W(0xA0, 0xA1, 0xA2, 0xA3)), W(0xA0, 0xB0, 0xA1, 0xB1))");
        t.IsTrue(eq(PS2_PEXTUW(W(0xB0, 0xB1, 0xB2, 0xB3), W(0xA0, 0xA1, 0xA2, 0xA3)), W(0xA2, 0xB2, 0xA3, 0xB3)), "eq(PS2_PEXTUW(W(0xB0, 0xB1, 0xB2, 0xB3), W(0xA0, 0xA1, 0xA2, 0xA3)), W(0xA2, 0xB2, 0xA3, 0xB3))");
        t.IsTrue(eq(PS2_PCPYLD(W(0xB0, 0xB1, 0xB2, 0xB3), W(0xA0, 0xA1, 0xA2, 0xA3)), W(0xA0, 0xA1, 0xB0, 0xB1)), "eq(PS2_PCPYLD(W(0xB0, 0xB1, 0xB2, 0xB3), W(0xA0, 0xA1, 0xA2, 0xA3)), W(0xA0, 0xA1, 0xB0, 0xB1))");
        t.IsTrue(eq(PS2_PPACW(W(0xB0, 0xB1, 0xB2, 0xB3), W(0xA0, 0xA1, 0xA2, 0xA3)), W(0xA0, 0xA2, 0xB0, 0xB2)), "eq(PS2_PPACW(W(0xB0, 0xB1, 0xB2, 0xB3), W(0xA0, 0xA1, 0xA2, 0xA3)), W(0xA0, 0xA2, 0xB0, 0xB2))");
        t.IsTrue(eq(PS2_PPACH(rs, rt), H(0xA0, 0xA2, 0xA4, 0xA6, 0xB0, 0xB2, 0xB4, 0xB6)), "eq(PS2_PPACH(rs, rt), H(0xA0, 0xA2, 0xA4, 0xA6, 0xB0, 0xB2, 0xB4, 0xB6))");
        t.IsTrue(eq(PS2_PEXTLH(rs, rt), H(0xA0, 0xB0, 0xA1, 0xB1, 0xA2, 0xB2, 0xA3, 0xB3)), "eq(PS2_PEXTLH(rs, rt), H(0xA0, 0xB0, 0xA1, 0xB1, 0xA2, 0xB2, 0xA3, 0xB3))");
        t.IsTrue(eq(PS2_PEXTUH(rs, rt), H(0xA4, 0xB4, 0xA5, 0xB5, 0xA6, 0xB6, 0xA7, 0xB7)), "eq(PS2_PEXTUH(rs, rt), H(0xA4, 0xB4, 0xA5, 0xB5, 0xA6, 0xB6, 0xA7, 0xB7))");
        // QFSRV: (rs:rt) >> sa bits, low 128.
        t.IsTrue(eq(PS2_QFSRV(W(0xB0, 0xB1, 0xB2, 0xB3), W(0xA0, 0xA1, 0xA2, 0xA3), 32), W(0xA1, 0xA2, 0xA3, 0xB0)), "eq(PS2_QFSRV(W(0xB0, 0xB1, 0xB2, 0xB3), W(0xA0, 0xA1, 0xA2, 0xA3), 32), W(0xA1, 0xA2, 0xA3, 0xB0))");
        t.IsTrue(eq(PS2_QFSRV(W(0xB0, 0xB1, 0xB2, 0xB3), W(0xA0, 0xA1, 0xA2, 0xA3), 96), W(0xA3, 0xB0, 0xB1, 0xB2)), "eq(PS2_QFSRV(W(0xB0, 0xB1, 0xB2, 0xB3), W(0xA0, 0xA1, 0xA2, 0xA3), 96), W(0xA3, 0xB0, 0xB1, 0xB2))");
        t.IsTrue(eq(PS2_QFSRV(W(0x000000B0, 0, 0, 0), W(0, 0, 0, 0x12345678), 8), W(0, 0, 0x78000000, 0xB0123456)), "eq(PS2_QFSRV(W(0x000000B0, 0, 0, 0), W(0, 0, 0, 0x12345678), 8), W(0, 0, 0x78000000, 0xB0123456))");

        R5900Context ctx{};
        // Variable shifts: rt shifted by rs, words 0 and 2, sign extended.
        t.IsTrue(eq(ps2mmi::psllvw(W(4, 99, 31, 99), W(0x08000001, 7, 1, 7)), W(0x80000010, 0xFFFFFFFF, 0x80000000, 0xFFFFFFFF)), "eq(ps2mmi::psllvw(W(4, 99, 31, 99), W(0x08000001, 7, 1, 7)), W(0x80000010, 0xFFFFFFFF, 0x80000000, 0xFFFFFFFF))");
        t.IsTrue(eq(ps2mmi::psrlvw(W(4, 99, 0, 99), W(0x80000000, 7, 0x80000000, 7)), W(0x08000000, 0, 0x80000000, 0xFFFFFFFF)), "eq(ps2mmi::psrlvw(W(4, 99, 0, 99), W(0x80000000, 7, 0x80000000, 7)), W(0x08000000, 0, 0x80000000, 0xFFFFFFFF))");
        t.IsTrue(eq(ps2mmi::psravw(W(4, 99, 36, 99), W(0x80000000, 7, 0x40000000, 7)), W(0xF8000000, 0xFFFFFFFF, 0x04000000, 0)), "eq(ps2mmi::psravw(W(4, 99, 36, 99), W(0x80000000, 7, 0x40000000, 7)), W(0xF8000000, 0xFFFFFFFF, 0x04000000, 0))");

        // PMULTW: -3 * 5 and 0x40000000 * 8.
        __m128i rd = ps2mmi::multiplyWords(&ctx, W((uint32_t)-3, 77, 0x40000000, 77), W(5, 88, 8, 88), true, 0);
        t.IsTrue(eq(rd, W(0xFFFFFFF1, 0xFFFFFFFF, 0, 2)), "eq(rd, W(0xFFFFFFF1, 0xFFFFFFFF, 0, 2))");
        t.IsTrue(ctx.lo == 0xFFFFFFFFFFFFFFF1ull && ctx.hi == 0xFFFFFFFFFFFFFFFFull && ctx.lo1 == 0 && ctx.hi1 == 2, "ctx.lo == 0xFFFFFFFFFFFFFFF1ull && ctx.hi == 0xFFFFFFFFFFFFFFFFull && ctx.lo1 == 0 && ctx.hi1 == 2");
        // PMADDW on top of that: += 20 and += 1.
        rd = ps2mmi::multiplyWords(&ctx, W(4, 0, 1, 0), W(5, 0, 1, 0), true, 1);
        t.IsTrue(eq(rd, W(5, 0, 1, 2)), "eq(rd, W(5, 0, 1, 2))");
        t.IsTrue(ctx.lo == 5 && ctx.hi == 0 && ctx.lo1 == 1 && ctx.hi1 == 2, "ctx.lo == 5 && ctx.hi == 0 && ctx.lo1 == 1 && ctx.hi1 == 2");
        // PMULTUW: 0xFFFFFFFF * 2 is unsigned.
        rd = ps2mmi::multiplyWords(&ctx, W(0xFFFFFFFF, 0, 3, 0), W(2, 0, 3, 0), false, 0);
        t.IsTrue(eq(rd, W(0xFFFFFFFE, 1, 9, 0)), "eq(rd, W(0xFFFFFFFE, 1, 9, 0))");
        t.IsTrue(ctx.lo == 0xFFFFFFFFFFFFFFFEull && ctx.hi == 1 && ctx.lo1 == 9 && ctx.hi1 == 0, "ctx.lo == 0xFFFFFFFFFFFFFFFEull && ctx.hi == 1 && ctx.lo1 == 9 && ctx.hi1 == 0");

        // PDIVW: -7 / 2 and 9 / 0.
        ps2mmi::divideWords(&ctx, W((uint32_t)-7, 0, 9, 0), W(2, 0, 0, 0), true);
        t.IsTrue(ctx.lo == (uint64_t)-3ll && ctx.hi == (uint64_t)-1ll && ctx.lo1 == ~0ull && ctx.hi1 == 9, "ctx.lo == (uint64_t)-3ll && ctx.hi == (uint64_t)-1ll && ctx.lo1 == ~0ull && ctx.hi1 == 9");
        ps2mmi::divideWords(&ctx, W(7, 0, 9, 0), W(2, 0, 0, 0), false);
        t.IsTrue(ctx.lo == 3 && ctx.hi == 1 && ctx.lo1 == ~0ull && ctx.hi1 == 9, "ctx.lo == 3 && ctx.hi == 1 && ctx.lo1 == ~0ull && ctx.hi1 == 9");

        // PMULTH: products 0..7 land in LO.w0, LO.w1, HI.w0, HI.w1, LO.w2, LO.w3, HI.w2, HI.w3.
        rd = ps2mmi::multiplyHalves(&ctx, H(1, 2, 3, 4, 5, 6, 7, (uint16_t)-8), H(10, 10, 10, 10, 10, 10, 10, 10), 0);
        t.IsTrue(eq(ps2mmi::pmflo(&ctx), W(10, 20, 50, 60)), "eq(ps2mmi::pmflo(&ctx), W(10, 20, 50, 60))");
        t.IsTrue(eq(ps2mmi::pmfhi(&ctx), W(30, 40, 70, (uint32_t)-80)), "eq(ps2mmi::pmfhi(&ctx), W(30, 40, 70, (uint32_t)-80))");
        t.IsTrue(eq(rd, W(10, 30, 50, 70)), "eq(rd, W(10, 30, 50, 70))");
        rd = ps2mmi::multiplyHalves(&ctx, H(1, 1, 1, 1, 1, 1, 1, 1), H(1, 1, 1, 1, 1, 1, 1, 1), 1);
        t.IsTrue(eq(rd, W(11, 31, 51, 71)), "eq(rd, W(11, 31, 51, 71))");
        rd = ps2mmi::multiplyHalves(&ctx, H(1, 1, 1, 1, 1, 1, 1, 1), H(2, 2, 2, 2, 2, 2, 2, 2), -1);
        t.IsTrue(eq(rd, W(9, 29, 49, 69)), "eq(rd, W(9, 29, 49, 69))");
        // PHMADH / PHMSBH.
        rd = ps2mmi::multiplyHalfPairs(&ctx, H(1, 2, 3, 4, 5, 6, 7, 8), H(10, 100, 10, 100, 10, 100, 10, 100), false);
        t.IsTrue(eq(rd, W(210, 430, 650, 870)), "eq(rd, W(210, 430, 650, 870))");
        rd = ps2mmi::multiplyHalfPairs(&ctx, H(1, 2, 3, 4, 5, 6, 7, 8), H(10, 100, 10, 100, 10, 100, 10, 100), true);
        t.IsTrue(eq(rd, W(190, 370, 550, 730)), "eq(rd, W(190, 370, 550, 730))");

        // PMFHL.
        ps2mmi::pmtlo(&ctx, W(0x11, 0x12, 0x13, 0x14));
        ps2mmi::pmthi(&ctx, W(0x21, 0x22, 0x23, 0x24));
        t.IsTrue(eq(ps2mmi::pmfhlLw(&ctx), W(0x11, 0x21, 0x13, 0x23)), "eq(ps2mmi::pmfhlLw(&ctx), W(0x11, 0x21, 0x13, 0x23))");
        t.IsTrue(eq(ps2mmi::pmfhlUw(&ctx), W(0x12, 0x22, 0x14, 0x24)), "eq(ps2mmi::pmfhlUw(&ctx), W(0x12, 0x22, 0x14, 0x24))");
        t.IsTrue(eq(ps2mmi::pmfhlLh(&ctx), H(0x11, 0x12, 0x21, 0x22, 0x13, 0x14, 0x23, 0x24)), "eq(ps2mmi::pmfhlLh(&ctx), H(0x11, 0x12, 0x21, 0x22, 0x13, 0x14, 0x23, 0x24))");
        ps2mmi::pmtlo(&ctx, W(0x12345, (uint32_t)-70000, 3, 4));
        ps2mmi::pmthi(&ctx, W(5, 6, 0x7FFFFFFF, 0x80000000));
        t.IsTrue(eq(ps2mmi::pmfhlSh(&ctx), H(0x7FFF, 0x8000, 5, 6, 3, 4, 0x7FFF, 0x8000)), "eq(ps2mmi::pmfhlSh(&ctx), H(0x7FFF, 0x8000, 5, 6, 3, 4, 0x7FFF, 0x8000))");
        ps2mmi::pmtlo(&ctx, W(0x80000000, 0, 0x10, 0));
        ps2mmi::pmthi(&ctx, W(0, 0, 0xFFFFFFFF, 0));
        t.IsTrue(eq(ps2mmi::pmfhlSlw(&ctx), W(0x7FFFFFFF, 0, 0x80000000, 0xFFFFFFFF)), "eq(ps2mmi::pmfhlSlw(&ctx), W(0x7FFFFFFF, 0, 0x80000000, 0xFFFFFFFF))");
        ps2mmi::pmthlLw(&ctx, W(1, 2, 3, 4));
        t.IsTrue(eq(ps2mmi::pmflo(&ctx), W(1, 0, 3, 0)) && eq(ps2mmi::pmfhi(&ctx), W(2, 0, 4, 0)), "eq(ps2mmi::pmflo(&ctx), W(1, 0, 3, 0)) && eq(ps2mmi::pmfhi(&ctx), W(2, 0, 4, 0))");

        // VU0 flags: dest xyzw, result (-1, 0, 3, -0).
        uint32_t mac = 0xDEAD; uint16_t status = 0;
        ps2Vu0SetFlags(_mm_set_ps(-0.0f, 3.0f, 0.0f, -1.0f), 0xF, mac, status);
        t.IsTrue(mac == ((0x8u | 0x1u) << 4 | (0x4u | 0x1u)), "mac == ((0x8u | 0x1u) << 4 | (0x4u | 0x1u))");
        t.IsTrue(status == (0x3 | (0x3 << 6)), "status == (0x3 | (0x3 << 6))");
        ps2Vu0SetFlags(_mm_set_ps(-0.0f, 3.0f, 0.0f, -1.0f), 0x2, mac, status); // z only
        t.IsTrue(mac == 0 && status == (0x3 << 6), "mac == 0 && status == (0x3 << 6)");
        t.IsTrue(ps2Vu0Div(1.0f, 0.0f) == FLT_MAX && ps2Vu0Div(-1.0f, 0.0f) == -FLT_MAX && ps2Vu0Div(6.0f, 3.0f) == 2.0f, "ps2Vu0Div(1.0f, 0.0f) == FLT_MAX && ps2Vu0Div(-1.0f, 0.0f) == -FLT_MAX && ps2Vu0Div(6.0f, 3.0f) == 2.0f");
        t.IsTrue(ps2Vu0Rsqrt(6.0f, -9.0f) == 2.0f && ps2Vu0Sqrt(-16.0f) == 4.0f, "ps2Vu0Rsqrt(6.0f, -9.0f) == 2.0f && ps2Vu0Sqrt(-16.0f) == 4.0f");
        t.IsTrue(FPU_DIV_S(0.0f, 0.0f) == FLT_MAX && FPU_DIV_S(-2.0f, 0.0f) == -FLT_MAX && FPU_DIV_S(1.0f, 4.0f) == 0.25f, "FPU_DIV_S(0.0f, 0.0f) == FLT_MAX && FPU_DIV_S(-2.0f, 0.0f) == -FLT_MAX && FPU_DIV_S(1.0f, 4.0f) == 0.25f");
        t.IsTrue(FPU_RSQRT_S(8.0f, -4.0f) == 4.0f, "FPU_RSQRT_S(8.0f, -4.0f) == 4.0f");
        float big; uint32_t bigBits = 0x7FFFFFFF; std::memcpy(&big, &bigBits, 4);
        t.IsTrue(FPU_MAX_S(big, 1.0f) != 1.0f && FPU_MIN_S(big, 1.0f) == 1.0f, "FPU_MAX_S(big, 1.0f) != 1.0f && FPU_MIN_S(big, 1.0f) == 1.0f");
        // VU MAX/MINI: sign-and-magnitude order over every bit pattern, and the
        // result is one of the operands. Lanes: (0x7FFFFFFF vs 5), (0xFFFFFFFF
        // vs -3), (-1 vs -2), (+0 vs -0).
        {
            const __m128i a = W(0x7FFFFFFFu, 0xFFFFFFFFu, 0xBF800000u, 0x00000000u);
            const __m128i b = W(0x40A00000u, 0xC0400000u, 0xC0000000u, 0x80000000u);
            const __m128 fa = _mm_castsi128_ps(a), fb = _mm_castsi128_ps(b);
            t.IsTrue(eq(_mm_castps_si128(ps2VuMax(fa, fb)), W(0x7FFFFFFFu, 0xC0400000u, 0xBF800000u, 0x00000000u)), "ps2VuMax picks the greater bit pattern");
            t.IsTrue(eq(_mm_castps_si128(ps2VuMax(fb, fa)), W(0x7FFFFFFFu, 0xC0400000u, 0xBF800000u, 0x00000000u)), "ps2VuMax does not depend on operand order");
            t.IsTrue(eq(_mm_castps_si128(ps2VuMini(fa, fb)), W(0x40A00000u, 0xFFFFFFFFu, 0xC0000000u, 0x80000000u)), "ps2VuMini picks the lesser bit pattern");
            t.IsTrue(eq(_mm_castps_si128(ps2VuMini(fb, fa)), W(0x40A00000u, 0xFFFFFFFFu, 0xC0000000u, 0x80000000u)), "ps2VuMini does not depend on operand order");
        }
        __m128 r = ps2Vu0RInit(0.0f);
        for (int i = 0; i < 5; ++i) r = ps2Vu0RNext(r);
        const float rf = _mm_cvtss_f32(r);
        t.IsTrue(rf >= 1.0f && rf < 2.0f, "rf >= 1.0f && rf < 2.0f");

    });
                   });
}
