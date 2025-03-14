/*
g++ -O3 tbb_sort.cpp -ltbb
- Fujitsu compiler is not supported
*/

#include <iostream>
#include <vector>
#include <algorithm>
#include <tbb/parallel_sort.h>
#include <random>
#include <chrono>

int main(int argc, char **argv)
{
  uint64_t n = 1024 * 1024 * 32;
  std::vector<float> data(n);

  int seed = 100;
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  std::mt19937 mt(seed);
  for(size_t i = 0; i < data.size(); i++) data[i] = dist(mt);

  // std::sort
  auto start = std::chrono::high_resolution_clock::now();
  std::sort(data.begin(), data.end());
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "std::sort " << elapsed.count() << " sec\n";

  mt.seed(seed);
  for(size_t i = 0; i < data.size(); i++) data[i] = dist(mt);

  // tbb::parallel_sort
  start = std::chrono::high_resolution_clock::now();
  tbb::parallel_sort(data.begin(), data.end());
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "tbb::parallel_sort " << elapsed.count() << " sec\n";

  return 0;
}
