#include "pdf.h"
#include "dm/dm_matrix.h"
#include "dm/miscs.h"

#include <gtest/gtest.h>

#define VERIFY_RESULT 0

TEST(dm, dummy) {
  dm::print_double_isa();
  dm::print_float_isa();

  uniform_int_pdf<int> dim_pdf{1, 1};
  uniform_real_pdf<double> pdf{1.0, 1.0};

  EXPECT_EQ(dim_pdf(), 1);
  EXPECT_EQ(pdf(), 1.0);
}

template <typename T> void mat_mul(const T *A, const T *B, int m, int n, int p, T *C) {
  for (int i = 0; i < m; ++i) {
    int ibase  = i * n;
    int ibase2 = i * p;
    for (int j = 0; j < p; ++j) {
      T c = T(0);
      for (int k = 0; k < n; ++k)
        c += A[ibase + k] * B[k * p + j];
      C[ibase2 + j] = c;
    }
  }
}

TEST(dm, mat_mul) {
  auto test_case = [](auto tol) {
    using real = decltype(tol);

    uniform_int_pdf<int> dim_pdf{1, 200};
    uniform_real_pdf<real> pdf{-1.0, 1.0};
    std::vector<real> A, B, C;
    std::vector<real> C2;

    for (int i = 0; i < 100; ++i) {
      int m = dim_pdf();
      int n = dim_pdf();
      int p = dim_pdf();

      A.resize(m * n);
      B.resize(n * p);
      C.resize(m * p);

      for (auto &v : A)
        v = pdf();

      for (auto &v : B)
        v = pdf();

      dm::mat_mul(A.data(), B.data(), m, n, p, C.data());

      // int count = m * p;
      // C2.resize(C.size());
      // mat_mul(A.data(), B.data(), m, n, p, C2.data());
      // for(int i = 0; i < count; ++i)
      //   EXPECT_NEAR(C[i], C2[i], 1.0e-10);
    }
  };

  test_case(0.f);
  test_case(0.0);
}

TEST(dm_simd, mat_mul) {

  auto test_case = [](auto tol) {
    using real = decltype(tol);

    uniform_int_pdf<int> dim_pdf{1, 200};
    uniform_real_pdf<real> pdf{-1.0, 1.0};
    std::vector<real> A, B, C;
    std::vector<real> C2;

    for (int i = 0; i < 100; ++i) {
      int m = dim_pdf();
      int n = dim_pdf();
      int p = dim_pdf();

      A.resize(m * n);
      B.resize(n * p);
      C.resize(m * p);

      for (auto &v : A)
        v = pdf();

      for (auto &v : B)
        v = pdf();

      dm::simd::mat_mul(A.data(), B.data(), m, n, p, C.data());

      // int count = m * p;
      // C2.resize(C.size());
      // mat_mul(A.data(), B.data(), m, n, p, C2.data());
      // for(int i = 0; i < count; ++i)
      //   EXPECT_NEAR(C[i], C2[i], 1.0e-10);
    }
  };

  test_case(0.f);
  test_case(0.0);
}

TEST(dm, mat_LUP_decompose) {

  auto test_case = [](auto tol) {
    using real = decltype(tol);

    uniform_int_pdf<int> dim_pdf{64, 128};
    uniform_real_pdf<real> pdf{-1.0, 1.0};
    std::vector<real> M;
    std::vector<real> LU;
    std::vector<real> R;
    std::vector<int> P;

    for (int i = 0; i < 100; ++i) {
      int n = dim_pdf();

      int count = n * n;
      M.resize(count);
      LU.resize(count);
      R.resize(count);
      P.resize(n + 1);

      for (auto &v : M)
        v = pdf();

      dm::mat_LUP_decompose(M.data(), n, LU.data(), P.data());

#if VERIFY_RESULT
      dm::mat_LUP_restore(LU.data(), P.data(), n, R.data());

      for (int j = 0; j < count; ++j)
        EXPECT_NEAR(R[j], M[j], tol);
#endif
    }
  };

  test_case(1.0e-5f);
  test_case(1.0e-8);
}

TEST(dm_simd, mat_LUP_decompose) {

  auto test_case = [](auto tol) {
    using real = decltype(tol);
    uniform_int_pdf<int> dim_pdf{64, 128};
    uniform_real_pdf<real> pdf{-1.0, 1.0};
    std::vector<real> M;
    std::vector<real> LU;
    std::vector<real> R;
    std::vector<int> P;

    for (int i = 0; i < 100; ++i) {
      int n = dim_pdf();

      int count = n * n;
      M.resize(count);
      LU.resize(count);
      R.resize(count);
      P.resize(n + 1);

      for (auto &v : M)
        v = pdf();

      dm::simd::mat_LUP_decompose(M.data(), n, LU.data(), P.data());

#if VERIFY_RESULT
      dm::simd::mat_LUP_restore(LU.data(), P.data(), n, R.data());

      for (int j = 0; j < count; ++j)
        EXPECT_NEAR(R[j], M[j], tol);
#endif
    }
  };
  test_case(1.0e-5f);
  test_case(1.0e-8);
}

TEST(dm, mat_LUP_solve_inverse) {
  auto test_case = [](auto tol) {
    using real = decltype(tol);
    uniform_int_pdf<int> dim_pdf{16, 128};
    uniform_real_pdf<real> pdf{-1.0, 1.0};
    std::vector<real> M;
    std::vector<real> invM;
    std::vector<real> I;

    for (int i = 0; i < 100; ++i) {
      int n = dim_pdf();

      int count = n * n;
      M.resize(count);
      invM.resize(count);
      I.resize(count);

      for (auto &v : M)
        v = pdf();

      dm::mat_LUP_solve_inverse(M.data(), n, invM.data());

#if VERIFY_RESULT
      dm::simd::mat_mul(M.data(), invM.data(), n, n, n, I.data());

      for (int j = 0; j < count; ++j)
        EXPECT_NEAR(I[j], (j % (n + 1)) == 0, tol);
#endif
    }
  };
  // test_case(2.0e-4f);
  test_case(1.0e-10);
}

TEST(dm_simd, mat_LUP_solve_inverse) {
  auto test_case = [](auto tol) {
    using real = decltype(tol);
    uniform_int_pdf<int> dim_pdf{16, 128};
    uniform_real_pdf<real> pdf{-1.0, 1.0};
    std::vector<real> M;
    std::vector<real> invM;
    std::vector<real> I;

    for (int i = 0; i < 100; ++i) {
      int n = dim_pdf();

      int count = n * n;
      M.resize(count);
      invM.resize(count);
      I.resize(count);

      for (auto &v : M)
        v = pdf();

      dm::simd::mat_LUP_solve_inverse(M.data(), n, invM.data());

#if VERIFY_RESULT
      dm::simd::mat_mul(M.data(), invM.data(), n, n, n, I.data());

      for (int j = 0; j < count; ++j)
        EXPECT_NEAR(I[j], (j % (n + 1)) == 0, tol);
#endif
    }
  };
  // test_case(2.0e-4f);
  test_case(1.0e-10);
}

TEST(dm_simd, mat_transpose) {
  auto test_case = [](auto tol) {
    using real = decltype(tol);
    uniform_int_pdf<int> dim_pdf{1, 1000};

    std::vector<real> M;

    for (int h = 0; h < 100; ++h) {
      int m = dim_pdf();
      int n = dim_pdf();
      // m = 8;
      // n = 4;
      int count = m * n;
      M.resize(count);
      for (int i = 0; i < count; ++i)
        M[i] = (i % n) * m + i / n;

      dm::simd::mat_transpose(M.data(), m, n);

#if VERIFY_RESULT
      for (int i = 0; i < count; ++i)
        EXPECT_EQ(M[i], i);
#endif
    }
  };
  test_case(0.f);
  test_case(0.0);
}

int main() {
  ::testing::InitGoogleTest();
  RUN_ALL_TESTS();
}