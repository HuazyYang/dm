# dm
 ## About
 dm is a numeric algorithm library featuring instructional level optimization using [Intel ISPC](https://ispc.github.io/).

 As so far, the following instructional parallel algorithm routine is provided:
  + Parallel matrix multiplication, which accelerate the C/C++ implement 2.2x;
  + square matrix LUP decompose and matrix inverse, which accelerate 1.2x compare to C/C++ implement;
  + matrix inplace tranpose alogrithm, which shall be 1.0x faster than C/C++ implement, althrough have not tested yet.

 More parallel implement will be included in future.

 ## Optimization note
 - dm use implicit CPU L1 cache line size of 64B per way, 8 ways per group to perform matrix sub-block subdivision strategy in some implementations(such as matrix multiplication and inplace transpose);
 - We link multiple instruction set object files into a one unified library and use Intel ISPC runtime to perfrom runtime code selection based on runtime most advanced IA set available(from AVX2 down to SSE2);