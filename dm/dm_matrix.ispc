#include "dm_simd_config.isph"

extern
void REAL_FN(mat_transpose)(
  uniform real data[],
  uniform int m,
  uniform int n
);

export
void REAL_FN(mat_mul)(
  uniform const real A[],
  uniform const real B[],
  uniform int m,
  uniform int n,
  uniform int p,
  uniform real C[]
) {
#define MUL_A_BLOCK_ROW   4

#if MUL_A_BLOCK_ROW == 4
  uniform int count;
  uniform int ips[programCount], ibase[programCount];
  real r[programCount];

  // use subdivision block A[4x4] * B[*x4]
  for(int i = programIndex; i < m; i += programCount) {
    count = packed_store_active(ibase, i * n);
    ips[programIndex] = i;
    foreach(j = 0 ... p) {

      unmasked {
        for(uniform int h = 0; h < count; ++h)
          r[h] = 0.0;
      }

      for(uniform int k = 0; k < n; ++k) {
        real b = B[k*p+j];
        for(uniform int h = 0; h < count; ++h)
          r[h] += A[ibase[h] + k] * b;
      }
      for(uniform int h = 0; h < count; ++h)
        C[ips[h]*p+j] = r[h];
    }
  }
#elif MUL_A_BLOCK_ROW == 2
  // use subdivision block A[2x*] * B[*x4]
  uniform int count;
  uniform int ips[programCount], ibase[programCount];
  real r[programCount >> 1];

  for(int i = (programIndex % 4); i < m; i += (programCount >> 1)) {
    if(programIndex < 4) {
      count = packed_store_active(ibase, i*n);
      ips[programIndex] = i;
    }
    foreach (j = 0 ... p) {

      unmasked {
        for (uniform int h = 0; h < count; ++h)
          r[h] = 0.0;
      }

      for (uniform int k = 0; k < n; ++k) {
        real b = B[k * p + j];
        for (uniform int h = 0; h < count; ++h)
          r[h] += A[ibase[h] + k] * b;
      }
      for (uniform int h = 0; h < count; ++h)
        C[ips[h] * p + j] = r[h];
    }
  }
#else
  // use subdivision block A[1x*] * B[*x4]
  uniform int ibase;
  for (uniform int i = 0; i < m; ++i) {
    ibase = i * n;
    foreach (j = 0 ... p) {
      real r = 0.0;
      for (uniform int k = 0; k < n; ++k) {
        real b = B[k * p + j];
        r += A[ibase + k] * b;
      }
      C[i * p + j] = r;
    }
  }
#endif
}

export
void REAL_FN(mat_LUP_restore)(
  uniform const real LU[],
  uniform const int P[],
  uniform int n,
  uniform real R[]) {
  

  for(uniform int i = 0; i < n; ++i)
    for(uniform int j = 0; j < n; ++j) {

   uniform real a0;
    if (i < j) {
      real a = 0.0;
      foreach(k = 0 ... i)
        a += LU[i * n + k] * LU[k * n + j];
      a0 = reduce_add(a) + LU[i * n + j];
    } else {
      real a = 0.0;
      foreach(k = 0 ... j)
        a += LU[i * n + k] * LU[k * n + j];
      
      a0 = reduce_add(a);
      if (i == j)
        a0 += LU[j * n + j];
      else
        a0 += LU[i * n + j] * LU[j * n + j];
    }

    R[P[i]*n+j] = a0;
  }
}

export
uniform int REAL_FN(mat_LUP_decompose)(
  uniform const real M[],
  uniform int n,
  uniform real LU[],
  uniform int P[]
) {

  uniform const int count = n * n;
  foreach(i = 0 ... count) LU[i] = M[i];

  foreach(i = 0 ... n+1) P[i] = i;

  uniform int rows[programCount];

  for(uniform int i = 0; i < n - 1; ++i) {
    real p = 0.0;
    int row = i;
    foreach(j = i ... n) {
      real q = abs(LU[j*n + i]);
      if(q > p) {
        p = q;
        row = j;
      }
    }

    uniform real p_ = 0.0;
    foreach_unique(val in p) {
      if(val > p_) {
        p_ = val;
        packed_store_active(rows, row);
      }
    }

    if(p_ == 0.0)
      return -1;

    uniform const int row_ = rows[0];

    if(row_ != i) {
      uniform int tmp = P[i];
      P[i] = P[row_];
      P[row_] = tmp;

      foreach(j = 0 ... n) {
        real tmp2 = LU[i*n+j];
        LU[i*n+j] = LU[row_*n+j];
        LU[row_*n+j] = tmp2;
      }
      P[n] += 1;
    }

    uniform const real u = LU[i*n+i];
    for(uniform int j = i+1; j < n; ++j) {
      uniform real l = LU[j*n+i] / u;
      LU[j*n+i] = l;
      foreach(k = i+1 ... n)
        LU[j*n+k] -= LU[i*n+k] * l;
    }
  }

  return 0;
}

export
void REAL_FN(mat_LUP_solve)(
  uniform const real LU[],
  uniform const int P[],
  uniform const real b[],
  uniform int n,
  uniform real x[]
) {
  // forward substitute
  for(uniform int i = 0; i < n; ++i) {
    x[i] = b[P[i]];
    real a = 0.0;
    foreach(j = 0 ... i)
      a -= LU[i*n+j] * x[j];
    x[i] += reduce_add(a);
  }

  // backward substitute
  for(uniform int i = n - 1; i >= 0; --i) {
    real a = 0.0;
    foreach(j = i+1 ... n)
      a -= LU[i*n+j] * x[j];
    x[i] = (x[i] + reduce_add(a)) / LU[i*n+i];
  }
}

static
void mat_LUP_solve_one(
  uniform const real LU[],
  uniform const int P[],
  uniform int n,
  uniform int idx,
  uniform real x[]
) {
  // forward substitute
  for(uniform int i = 0; i < n; ++i) {
    x[i] = (real)(P[i] == idx);
    real a = 0.0;
    foreach(j = 0 ... i)
      a -= LU[i*n+j] * x[j];
    x[i] += reduce_add(a);
  }

  // backward substitute
  for(uniform int i = n - 1; i >= 0; --i) {
    real a = 0.0;
    foreach(j = i+1 ... n)
      a -= LU[i*n+j] * x[j];
    x[i] = (x[i] + reduce_add(a)) / LU[i*n+i];
  }
}

export
uniform int REAL_FN(mat_LUP_solve_inverse)(
  uniform const real M[],
  uniform int n,
  uniform real invM[]
) {
  uniform bool use_stack_alloc = 
    ((n+1)*sizeof(uniform int) + (n*n) * sizeof(uniform real)) < DM_SIMD_ALLOCA_THRESHOLD;

  uniform int *uniform P;
  if(use_stack_alloc) P = (uniform int *uniform)alloca((n+1)*sizeof(uniform int));
  else P = uniform new uniform int[n+1];

  uniform int ret = REAL_FN(mat_LUP_decompose)(M, n, invM, P);
  if(ret < 0) {
    if(!use_stack_alloc) delete[] P;

    return ret;
  }

  uniform const int count = n * n;
  uniform real *uniform x;
  if(use_stack_alloc) x = (uniform real *uniform)alloca(count * sizeof(uniform real));
  else x = uniform new uniform real[count];

  for(uniform int i = 0; i < n; ++i)
    mat_LUP_solve_one(invM, P, n, i, x + i * n);

  REAL_FN(mat_transpose)(x, n, n);

  memcpy(invM, x, count * sizeof(uniform real));

  if(!use_stack_alloc) {
    delete[] x;
    delete[] P;
  }

  return 0;
}