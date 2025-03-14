/*
g++ -O3 std_sort.cpp -ltbb
- Fujitsu compiler is not supported
*/

#include <iostream>
#include <vector>
#include <algorithm>
#include <execution>
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

  // std::sort seq
  start = std::chrono::high_resolution_clock::now();
  std::sort(std::execution::seq, data.begin(), data.end());
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "std::sort seq " << elapsed.count() << " sec\n";

  mt.seed(seed);
  for(size_t i = 0; i < data.size(); i++) data[i] = dist(mt);

  // std::sort unseq
  start = std::chrono::high_resolution_clock::now();
  std::sort(std::execution::unseq, data.begin(), data.end());
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "std::sort unseq " << elapsed.count() << " sec\n";

  mt.seed(seed);
  for(size_t i = 0; i < data.size(); i++) data[i] = dist(mt);

  // std::sort par
  start = std::chrono::high_resolution_clock::now();
  std::sort(std::execution::par, data.begin(), data.end());
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "std::sort par " << elapsed.count() << " sec\n";

  mt.seed(seed);
  for(size_t i = 0; i < data.size(); i++) data[i] = dist(mt);

  // std::sort par_unseq
  start = std::chrono::high_resolution_clock::now();
  std::sort(std::execution::par_unseq, data.begin(), data.end());
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "std::sort par_unseq " << elapsed.count() << " sec\n";

  return 0;
}
