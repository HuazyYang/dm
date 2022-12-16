#include "dm_simd_config.isph"

#if DM_REAL_AS_DOUBLE

export
void print_double_isa() {
#ifdef ISPC_TARGET_SSE2
  print("[dm_simd.double] use SSE2 ISA, lane width: %\n", programCount);
#endif
#ifdef ISPC_TARGET_SSE4
  print("[dm_simd.double] use SSE4 ISA, lane width: %\n", programCount);
#endif

#ifdef ISPC_TARGET_AVX
  print("[dm_simd.double] use AVX ISA, lane width: %\n", programCount);
#endif

#ifdef ISPC_TARGET_AVX2
  print("[dm_simd.double] use AVX2 ISA, lane width: %\n", programCount);
#endif
}

#elif DM_REAL_AS_FLOAT

export
void print_float_isa() {
#ifdef ISPC_TARGET_SSE2
  print("[dm_simd.float] use SSE2 ISA, lane width: %\n", programCount);
#endif
#ifdef ISPC_TARGET_SSE4
  print("[dm_simd.float] use SSE4 ISA, lane width: %\n", programCount);
#endif

#ifdef ISPC_TARGET_AVX
  print("[dm_simd.float] use AVX ISA, lane width: %\n", programCount);
#endif

#ifdef ISPC_TARGET_AVX2
  print("[dm_simd.float] use AVX2 ISA, lane width: %\n", programCount);
#endif
}

#endif
