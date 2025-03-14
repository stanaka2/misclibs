/*
g++ -O3 -fopenmp nanoflann_sample.cpp
*/

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

#include "nanoflann.hpp"

struct particle {
  float x, y, z;
  int id;
};

struct PointCloud {
  std::vector<particle> &ptcls;
  PointCloud(std::vector<particle> &p) : ptcls(p) {}
  inline size_t kdtree_get_point_count() const { return ptcls.size(); }

  inline float kdtree_get_pt(const size_t idx, const size_t dim) const
  {
    if(dim == 0) return ptcls[idx].x;
    else if(dim == 1) return ptcls[idx].y;
    else return ptcls[idx].z;
  }

  template <class BBOX>
  bool kdtree_get_bbox(BBOX &) const
  {
    return false;
  }
};

int main(int argc, char **argv)
{
  constexpr int ndim = 3;
  //  constexpr int max_leaf = 10;
  constexpr int max_leaf = 50;
  //  constexpr int max_leaf = 100;

  constexpr int build_threads = 0;
  // constexpr int build_threads = 1;
  // const int build_threads = omp_get_max_threads();

  std::vector<particle> ptcls(1024 * 1024 * 32);

  int seed = 100;
  std::mt19937 mt(seed);
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  for(size_t i = 0; i < ptcls.size(); i++) {
    ptcls[i].x = dist(mt);
    ptcls[i].y = dist(mt);
    ptcls[i].z = dist(mt);
    ptcls[i].id = static_cast<int>(i);
  }

  PointCloud cloud(ptcls);

  // construct a kd-tree index:
  using my_kd_tree_t =
      nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud>, PointCloud, ndim>;

  auto t_start = std::chrono::high_resolution_clock::now();

  nanoflann::KDTreeSingleIndexAdaptorParams tree_params;
  tree_params.leaf_max_size = max_leaf;
  tree_params.n_thread_build = build_threads;

  // build tree index
  my_kd_tree_t index(ndim, cloud, tree_params);

  auto t_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> t_elapsed = t_end - t_start;
  std::cout << "KD-Tree built with " << ptcls.size() << " particles. Elapsed time: " << t_elapsed.count() << " [s]\n";

  std::vector<nanoflann::ResultItem<uint32_t, float>> matches;
  nanoflann::SearchParameters search_params;
  search_params.sorted = false;

  float query_pt[3] = {0.5f, 0.5f, 0.5f};

  // radiusSearch
  // Ensure that the input is the squared distance, not the actual distance.
  float radius = 0.1f;

  t_start = std::chrono::high_resolution_clock::now();
  const uint32_t num_found = index.radiusSearch(query_pt, radius * radius, matches, search_params);
  std::cout << "Found " << num_found << " particles within radius " << radius << ":\n";

  t_end = std::chrono::high_resolution_clock::now();
  t_elapsed = t_end - t_start;
  std::cout << "Elapsed time: " << t_elapsed.count() << " [s]\n";

#if 0
  for(const auto &m : matches) {
    std::cout << "Index: " << m.first << ", Distance^2: " << m.second << '\n';
  }
#endif

  // knnSearch
  const size_t k = 15;
  uint32_t out_indices[k];
  float out_distances_sq[k];

  t_start = std::chrono::high_resolution_clock::now();
  const size_t nMatches = index.knnSearch(query_pt, k, out_indices, out_distances_sq);
  std::cout << "Found " << nMatches << " nearest neighbors:\n";

  t_end = std::chrono::high_resolution_clock::now();
  t_elapsed = t_end - t_start;
  std::cout << "Elapsed time: " << t_elapsed.count() << " [s]\n";

  for(size_t i = 0; i < nMatches; i++) {
    std::cout << "Index: " << out_indices[i] << ", Distance^2: " << out_distances_sq[i] << "\n";
  }

  return 0;
}
