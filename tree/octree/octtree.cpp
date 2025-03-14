/*
g++ -O3 -fopenmp octtree.cpp
*/

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

#include "octtree.hpp"

struct particle {
  float x, y, z;
  int id;
};

struct PointCloud {
  std::vector<particle> &ptcls;
  PointCloud(std::vector<particle> &p) : ptcls(p) {}
  inline size_t octtree_get_point_count() const { return ptcls.size(); }

  inline float octtree_get_pt(const size_t idx, const size_t dim) const
  {
    if(dim == 0) return ptcls[idx].x;
    else if(dim == 1) return ptcls[idx].y;
    else return ptcls[idx].z;
  }
};

int main(int argc, char **argv)
{
  constexpr int max_leaf = 50;
  //  constexpr int max_leaf = 100;
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
  using my_octtree_t = octtree::Octree<float, PointCloud>;
  my_octtree_t octree;

  auto t_start = std::chrono::high_resolution_clock::now();

  octree.index(cloud, max_leaf);

  auto t_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> t_elapsed = t_end - t_start;
  std::cout << "OctTree built with " << ptcls.size() << " particles. Elapsed time: " << t_elapsed.count() << " [s]\n";

  float query_pt[3] = {0.5f, 0.5f, 0.5f};
  float radius = 0.1f;
  std::vector<size_t> radiusResults;
  octree.radiusSearch(query_pt, radius, radiusResults);
  std::cout << "radiusSearch: nMatches " << radiusResults.size() << " points within radius " << radius << "\n";

  const size_t k = 15;
  std::vector<size_t> knnIndices;
  std::vector<float> knnDists;
  size_t nMatches = octree.knnSearch(query_pt, k, knnIndices, knnDists);
  std::cout << "Found " << nMatches << " nearest neighbors:\n";

  t_end = std::chrono::high_resolution_clock::now();
  t_elapsed = t_end - t_start;
  std::cout << "Elapsed time: " << t_elapsed.count() << " [s]\n";

  for(size_t i = 0; i < nMatches; i++) {
    std::cout << "Index: " << knnIndices[i] << ", Distance^2: " << knnDists[i] << "\n";
  }

  return 0;
}
