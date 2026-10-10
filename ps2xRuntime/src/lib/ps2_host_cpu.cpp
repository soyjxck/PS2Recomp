#include "ps2_host_cpu.h"

#include <cstdint>

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__AVX2__) || defined(__FMA__))
#define PS2X_HOST_CPU_NEEDS_AVX2 1
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#else
#define PS2X_HOST_CPU_NEEDS_AVX2 0
#endif

namespace
{
#if PS2X_HOST_CPU_NEEDS_AVX2
    void cpuid(uint32_t leaf, uint32_t subleaf, uint32_t out[4])
    {
#if defined(_MSC_VER)
        int regs[4];
        __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
        for (int i = 0; i < 4; ++i)
            out[i] = static_cast<uint32_t>(regs[i]);
#else
        unsigned int a = 0, b = 0, c = 0, d = 0;
        if (!__get_cpuid_count(leaf, subleaf, &a, &b, &c, &d))
            a = b = c = d = 0;
        out[0] = a;
        out[1] = b;
        out[2] = c;
        out[3] = d;
#endif
    }

    // XCR0: whether the OS saves the XMM and YMM state. Only valid once
    // CPUID reports OSXSAVE.
    uint64_t xgetbv0()
    {
#if defined(_MSC_VER)
        return _xgetbv(0);
#else
        uint32_t low = 0, high = 0;
        __asm__ volatile("xgetbv" : "=a"(low), "=d"(high) : "c"(0));
        return (static_cast<uint64_t>(high) << 32) | low;
#endif
    }
#endif
}

const char *ps2xHostCpuProblem()
{
#if PS2X_HOST_CPU_NEEDS_AVX2
    uint32_t regs[4];
    cpuid(0, 0, regs);
    const uint32_t maxLeaf = regs[0];
    if (maxLeaf < 7)
        return "this build needs a CPU with AVX2 and FMA (Intel Haswell, AMD Zen or later); "
               "rebuild with -DPS2X_X86_AVX2=OFF for this processor";
    cpuid(1, 0, regs);
    const bool fma = (regs[2] & (1u << 12)) != 0;
    const bool osxsave = (regs[2] & (1u << 27)) != 0;
    const bool avx = (regs[2] & (1u << 28)) != 0;
    cpuid(7, 0, regs);
    const bool avx2 = (regs[1] & (1u << 5)) != 0;
    if (!fma || !avx || !avx2)
        return "this build needs a CPU with AVX2 and FMA (Intel Haswell, AMD Zen or later); "
               "rebuild with -DPS2X_X86_AVX2=OFF for this processor";
    if (!osxsave || (xgetbv0() & 0x6u) != 0x6u)
        return "this build needs AVX2, which the operating system has not enabled for programs "
               "(XMM and YMM state are not saved); rebuild with -DPS2X_X86_AVX2=OFF";
#endif
    return nullptr;
}
