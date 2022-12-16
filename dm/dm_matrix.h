#ifndef DM_MATRIX
#define DM_MATRIX
#include <type_traits>

namespace dm {

template<typename real>
using enable_real_t = std::enable_if_t<std::is_same_v<real, float> || std::is_same_v<real, real>>;

template<typename real, typename = enable_real_t<real>> void mat_mul(const real *A, const real *B, int m, int n, int p, real *C);
// P's dimension must be no less than n+1
template<typename real, typename = enable_real_t<real>> void mat_LUP_restore(const real *LP, const int *P, int n, real *R);
template<typename real, typename = enable_real_t<real>> int mat_LUP_decompose(const real *M, int n, real *LU, int *P);
template<typename real, typename = enable_real_t<real>> void mat_LUP_solve(const real *LU, int *P, const real *b, int n, real *x);
template<typename real, typename = enable_real_t<real>> int mat_LUP_solve_inverse(const real *M, int n, real *invM);

namespace simd {
template<typename real, typename = enable_real_t<real>> void mat_mul(const real *A, const real *B, int m, int n, int p, real *C);
template<typename real, typename = enable_real_t<real>> void mat_transpose(real *M, int m, int n);
template<typename real, typename = enable_real_t<real>> void mat_LUP_restore(const real *LU, const int *P, int n, real *R);
template<typename real, typename = enable_real_t<real>> int mat_LUP_decompose(const real *M, int n, real *LU, int *P);
template<typename real, typename = enable_real_t<real>> void mat_LUP_solve(const real *LU, int *P, const real *b, int n, real *x);
template<typename real, typename = enable_real_t<real>> int mat_LUP_solve_inverse(const real *M, int n, real *invM);

};

};



#endif /* DM_MATRIX */
