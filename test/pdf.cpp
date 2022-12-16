#include "pdf.h"

unsigned long long g_RandomSeed = [] {
  std::random_device rdev;
  return rdev();
} ();

unsigned long long get_random_seed() {
  return g_RandomSeed;
}