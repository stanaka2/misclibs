/*
g++ -O3 -fopenmp hnswlib_sample.cpp
*/

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "hnswlib/hnswlib.h"

struct particle {
  float x, y, z;
  int id;
};

int main()
{
  constexpr int ndim = 3;                  // 次元数
                                           // constexpr int max_elements = 1024 * 1024 * 64; // 最大データ数
  constexpr int max_elements = 1024 * 100; // 最大データ数
  constexpr int M = 16;                    // 近傍ノード数
  constexpr int ef_construction = 50;      // インデックス構築時の探索深さ
  constexpr int ef_search = 50;            // クエリ検索時の探索深さ（大きいほど精度向上）

  std::vector<particle> ptcls(max_elements);

  for(size_t i = 0; i < ptcls.size(); i++) {
    ptcls[i].x = static_cast<float>(rand()) / RAND_MAX;
    ptcls[i].y = static_cast<float>(rand()) / RAND_MAX;
    ptcls[i].z = static_cast<float>(rand()) / RAND_MAX;
    ptcls[i].id = static_cast<int>(i);
  }

  // hnswlib::L2Space space(ndim);
  hnswlib::L2Space space(ndim);
  hnswlib::HierarchicalNSW<float> index(&space, max_elements, M, ef_construction);

  auto t_start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for
  for(size_t i = 0; i < ptcls.size(); i++) {
    float data[3] = {ptcls[i].x, ptcls[i].y, ptcls[i].z};
    index.addPoint(data, ptcls[i].id);
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> t_elapsed = t_end - t_start;
  std::cout << "Index built with " << ptcls.size() << " particles. Elapsed time: " << t_elapsed.count() << " [s]\n";

  float query_pt[3] = {0.5f, 0.5f, 0.5f};

  constexpr int k = 15;
  t_start = std::chrono::high_resolution_clock::now();
  auto results = index.searchKnn(query_pt, k);
  t_end = std::chrono::high_resolution_clock::now();
  t_elapsed = t_end - t_start;

  std::cout << "Found " << results.size() << " nearest neighbors:\n";
  //  for(const auto &res : results) {
  //    std::cout << "Index: " << res.first << ", Distance^2: " << res.second << "\n";
  //  }
  std::cout << "Elapsed time: " << t_elapsed.count() << " [s]\n";

  return 0;
}
