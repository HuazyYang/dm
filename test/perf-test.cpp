#include <iostream>
#include <benchmark/benchmark.h>
#include "pdf.h"

BENCHMARK_MAIN();

class dm_perf: public benchmark::Fixture {
public:
  void setUp(const benchmark::State &st) {
    uniform_int_pdf<int> dim_pdf(st.range_x(), st.range_y());

    int m = dim_pdf();
    int n = dim_pdf();

    uniform_real_pdf<double> pdf{-1.0, 1.0};

    samples.resize(m * n);
    for(auto &v : samples) v = pdf();
  }

  void tearDown(const benchmark::State &state) {
    samples.clear();
  }
  int m, n;
  std::vector<double> samples;
};

BENCHMARK_DEFINE_F(dm_perf, dmat_mul) (benchmark::State &state)  {

  while(state.KeepRunning()) {
    
  }
}
BENCHMARK_REGISTER_F(dm_perf, dmat_mul)->Iterations(100);