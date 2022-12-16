
find_package(GTest CONFIG REQUIRED)
find_package(benchmark CONFIG REQUIRED)

set(TEST_SOURCES
  main.cpp
  pdf.cpp
)

add_executable(
  DMTest
  ${TEST_SOURCES}
)
target_link_libraries(DMTest PRIVATE  dm_algorithm GTest::gtest GTest::gtest_main)


add_executable(
  DMPerfTest
  perf-test.cpp
)
target_link_libraries(DMPerfTest PRIVATE dm_algorithm benchmark::benchmark benchmark::benchmark_main)