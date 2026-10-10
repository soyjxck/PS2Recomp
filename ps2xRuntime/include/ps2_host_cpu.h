#ifndef PS2_HOST_CPU_H
#define PS2_HOST_CPU_H

// Why this build cannot run on the host CPU, or nullptr when it can. x86-64
// builds compiled for AVX2 and FMA (the default, PS2X_X86_AVX2) die with an
// illegal instruction on older processors; main calls this first to say so
// instead. Integer code only, so the check itself runs anywhere.
const char *ps2xHostCpuProblem();

#endif
