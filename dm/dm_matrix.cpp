#include "dm_matrix.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include "dm_matrix_double_ispc.h"
#include "dm_matrix_float_ispc.h"
#include "dm_transpose_double_ispc.h"
#include "dm_transpose_float_ispc.h"

namespace dm {

template<typename real>
void _mat_mul(const real *A, const real *B, int m , int n, int p, real *C) {
  // for (int i = 0; i < m; ++i) {
  //   int ibase = i*n;
  //   int ibase2 = i*p;
  //   for (int j = 0; j < p; ++j) {
  //     real c = real(0);
  //     for (int k = 0; k < n; ++k)
  //       c += A[ibase + k] * B[k * p + j];
  //     C[ibase2 + j] = c;
  //   }
  // }

  real As[8][8], Bs[8][8], Cs[8][8];

  for(int i = 0; i < m; i += 8) {
    int u = std::min(8, m - i);

    for(int j = 0; j < p; j += 8) {
      int w = std::min(8, p - j);

      memset(Cs, 0, sizeof(Cs));

      for(int k = 0; k < n; k += 8) {
        int v = std::min(8, n - k);

        for(int u2 = 0; u2 < u; ++u2)
          for(int v2 = 0; v2 < v; ++v2)
            As[u2][v2] = A[(i+u2) * n + (k+v2)];

        for(int v2 = 0; v2 < v; ++v2)
          for(int w2 = 0; w2 < w; ++w2)
            Bs[v2][w2] = B[(k+v2) * p + (j+w2)];

        for(int u2 = 0; u2 < u; ++u2)
          for(int v2 = 0; v2 < v; ++v2) {
            double a = As[u2][v2];
            for(int w2 = 0; w2 < w; ++w2)
              Cs[u2][w2] += a * Bs[v2][w2];
          }
      }

      for(int u2 = 0; u2 < u; ++u2)
        for(int w2 = 0; w2 < w; ++w2)
          C[(i+u2)*p+(j+w2)] = Cs[u2][w2];
    }
  }
}

template<>
void mat_mul<float>(const float *A, const float *B, int m, int n, int p, float *C) {
  return _mat_mul(A, B, m, n, p, C);
}
template <>
void mat_mul<double>(const double *A, const double *B, int m, int n, int p, double *C) {
  return _mat_mul(A, B, m, n, p, C);
}

template<>
void simd::mat_mul<float>(const float *A, const float *B, int m, int n, int p, float *C) {
  return ispc::fmat_mul(A, B, m, n, p, C);
}
template<>
void simd::mat_mul<double>(const double *A, const double *B, int m, int n, int p, double *C) {
  return ispc::dmat_mul(A, B, m, n, p, C);
}

template<typename real>
void _mat_LUP_restore(const real *LU, const int *P, int n, real *R) {

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      real a = real(0);
      if (i < j) {
        for (int k = 0; k < i; ++k)
          a += LU[i * n + k] * LU[k * n + j];
        a += LU[i * n + j];
      } else {
        for (int k = 0; k < j; ++k)
          a += LU[i * n + k] * LU[k * n + j];
        if (i == j)
          a += LU[j * n + j];
        else
          a += LU[i * n + j] * LU[j * n + j];
      }
      R[P[i] * n + j] = a;
    }
  }
}

template <> void mat_LUP_restore<float>(const float *LU, const int *P, int n, float *R) {
  return _mat_LUP_restore(LU, P, n, R);
}
template <> void mat_LUP_restore<double>(const double *LU, const int *P, int n, double *R) {
  return _mat_LUP_restore(LU, P, n, R);
}
template <> void simd::mat_LUP_restore<float>(const float *LU, const int *P, int n, float *R) {
  return ispc::fmat_LUP_restore(LU, P, n, R);
}
template <> void simd::mat_LUP_restore<double>(const double *LU, const int *P, int n, double *R) {
  return ispc::dmat_LUP_restore(LU, P, n, R);
}

template <typename real>
int _mat_LUP_decompose(const real *M, int n, real *LU, int *P) {
  int count = n * n;
  for (int i = 0; i < count; ++i)
    LU[i] = M[i];

  for (int i = 0; i < n+1; ++i)
    P[i] = i;

  for (int i = 0; i < n - 1; ++i) {
    real p = real(0);
    int row  = i;
    for (int j = i; j < n; ++j) {
      real q = std::abs(LU[j * n + i]);
      if (q > p) {
        p   = q;
        row = j;
      }
    }

    if (p == real(0))
      return -1;

    if (row != i) {
      int tmp = P[i];
      P[i]    = P[row];
      P[row]  = tmp;
      for (int j = 0; j < n; ++j) {
        real tmp2     = LU[i * n + j];
        LU[i * n + j]   = LU[row * n + j];
        LU[row * n + j] = tmp2;
      }
      P[n] += 1;
    }

    real u = LU[i * n + i];
    for (int j = i + 1; j < n; ++j) {
      real l      = LU[j * n + i] / u;
      LU[j * n + i] = l;
      for (int k = i + 1; k < n; ++k)
        LU[j * n + k] -= LU[i * n + k] * l;
    }
  }
  return 0;
}

template<>
int mat_LUP_decompose<float>(const float *M, int n, float *LU, int *P) {
  return _mat_LUP_decompose(M, n, LU, P);
}
template <> int mat_LUP_decompose<double>(const double *M, int n, double *LU, int *P) {
  return _mat_LUP_decompose(M, n, LU, P);
}
template <> int simd::mat_LUP_decompose<float>(const float *M, int n, float *LU, int *P) {
  return ispc::fmat_LUP_decompose(M, n, LU, P);
}
template <> int simd::mat_LUP_decompose<double>(const double *M, int n, double *LU, int *P) {
  return ispc::dmat_LUP_decompose(M, n, LU, P);
}

template<typename real>
void _mat_LUP_solve(const real *LU, const int *P, const real *b, int n, real *x) {
  // forward substitute
  for (int i = 0; i < n; ++i) {
    x[i] = b[P[i]];
    real a = real(0);
    for (int j = 0; j < i; ++j)
      a = a - LU[i * n + j] * x[j];
    x[i] += a;
  }

  // backward substitute
  for (int i = n - 1; i >= 0; --i) {
    real a = real(0);
    for (int j = i + 1; j < n; ++j)
      a = a - LU[i * n + j] * x[j];
    x[i] = (x[i] + a) / LU[i * n + i];
  }
}

template<>
void mat_LUP_solve<float>(const float *LU, int *P, const float *b, int n, float *x) {
  return _mat_LUP_solve(LU, P, b, n, x);
}
template <> void mat_LUP_solve<double>(const double *LU, int *P, const double *b, int n, double *x) {
  return _mat_LUP_solve(LU, P, b, n, x);
}
template <> void simd::mat_LUP_solve<float>(const float *LU, int *P, const float *b, int n, float *x) {
  return ispc::fmat_LUP_solve(LU, P, b, n, x);
}
template <> void simd::mat_LUP_solve<double>(const double *LU, int *P, const double *b, int n, double *x) {
  return ispc::dmat_LUP_solve(LU, P, b, n, x);
}

template<typename real>
void _mat_LUP_solve_one(const real *LU, const int *P, int n, int idx, real *x) {
  // forward substitute
  for (int i = 0; i < n; ++i) {
    x[i]     = real(P[i] == idx);
    real a = real(0);
    for (int j = 0; j < i; ++j)
      a = a - LU[i * n + j] * x[j];
    x[i] += a;
  }

  // backward substitute
  for (int i = n - 1; i >= 0; --i) {
    real a = real(0);
    for (int j = i + 1; j < n; ++j)
      a = a - LU[i * n + j] * x[j];
    x[i] = (x[i] + a) / LU[i * n + i];
  }
}

template<typename real>
int _mat_LUP_solve_inverse(const real *M, int n, real *invM) {

  int *P = new int[n+1];

  int ret = _mat_LUP_decompose(M, n, invM, P);
  if (ret < 0) {
    delete[] P;
    return ret;
  }

  const int count = n * n;
  real *x            = new real[count];

  for (int i = 0; i < n; ++i)
    _mat_LUP_solve_one(invM, P, n, i, x + i * n);

  simd::mat_transpose(x, n, n);

  memcpy(invM, x, count * sizeof(real));
  delete[] x;
  delete[] P;
  return 0;
}

template<>
int mat_LUP_solve_inverse<float>(const float *M, int n, float *invM) {
  return _mat_LUP_solve_inverse(M, n, invM);
}
template<>
int mat_LUP_solve_inverse<double>(const double *M, int n, double *invM) {
  return _mat_LUP_solve_inverse(M, n, invM);
}
template <> int simd::mat_LUP_solve_inverse<float>(const float *M, int n, float *invM) {
  return ispc::fmat_LUP_solve_inverse(M, n, invM);
}
template <> int simd::mat_LUP_solve_inverse<double>(const double *M, int n, double *invM) {
  return ispc::dmat_LUP_solve_inverse(M, n, invM);
}

template <> void simd::mat_transpose<float>(float *M, int m, int n) {
  ispc::fmat_transpose(M, m, n);
}
template<>
void simd::mat_transpose<double>(double *M, int m, int n) {
  ispc::dmat_transpose(M, m, n);
}

}