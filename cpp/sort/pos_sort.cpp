#include <algorithm>
#include <stdint.h>
#include <omp.h>

struct target {
  double xpos, ypos, zpos;
};

#define para_qsort_THRESHOLD (5000)
// #define para_qsort_THRESHOLD (2)

/* ### median ### */
template <typename T>
static inline T median(const T a, const T b, const T c)
{
  return std::max(std::min(a, b), std::min(std::max(a, b), c));
}

template <typename T>
static inline void qsort_partitioning_axis(T p[], int &left, int &right, double pivot, const int axis)
{
  int ii = left;
  int jj = right;

  /* partitioning */
  if(axis == 0) {
    while(1) {
      while(p[ii].xpos < pivot) ii++;
      while(pivot < p[jj].xpos) jj--;
      if(ii >= jj) break;
      std::swap(p[ii], p[jj]);
      ii++;
      jj--;
    }

  } else if(axis == 1) {
    while(1) {
      while(p[ii].ypos < pivot) ii++;
      while(pivot < p[jj].ypos) jj--;
      if(ii >= jj) break;
      std::swap(p[ii], p[jj]);
      ii++;
      jj--;
    }

  } else {
    while(1) {
      while(p[ii].zpos < pivot) ii++;
      while(pivot < p[jj].zpos) jj--;
      if(ii >= jj) break;
      std::swap(p[ii], p[jj]);
      ii++;
      jj--;
    }
  }

  left = ii;
  right = jj;
}

template <typename T>
static void single_qsort_axis(T p[], int left, int right, const int axis)
{
  if(left < right) {
    int ii = left, jj = right;
    double pivot;
    if(axis == 0) pivot = median(p[ii].xpos, p[jj].xpos, p[(ii + jj) / 2].xpos);
    else if(axis == 1) pivot = median(p[ii].ypos, p[jj].ypos, p[(ii + jj) / 2].ypos);
    else pivot = median(p[ii].zpos, p[jj].zpos, p[(ii + jj) / 2].zpos);

    qsort_partitioning_axis(p, ii, jj, pivot, axis);
    single_qsort_axis(p, left, ii - 1, axis);
    single_qsort_axis(p, jj + 1, right, axis);
  }
}

template <typename T, typename U>
static void para_qsort_internal_axis(U p[], int left, int right, const int axis)
{
  int length = right - left;
  if(length < para_qsort_THRESHOLD) {
    single_qsort_axis(p, left, right, axis);
    return;
  }

  int ii = left, jj = right;
  double pivot;
  if(axis == 0) pivot = median(p[ii].xpos, p[jj].xpos, p[(ii + jj) / 2].xpos);
  else if(axis == 1) pivot = median(p[ii].ypos, p[jj].ypos, p[(ii + jj) / 2].ypos);
  else pivot = median(p[ii].zpos, p[jj].zpos, p[(ii + jj) / 2].zpos);

  qsort_partitioning_axis(p, ii, jj, pivot, axis);

#pragma omp task
  para_qsort_internal_axis(p, left, jj, axis);
#pragma omp task
  para_qsort_internal_axis(p, ii, right, axis);
}

/* ### para_qsort_axis ### */
template <typename T, typename U>
static void para_qsort_axis(U p[], int left, int right, const int axis)
{
  if(omp_in_parallel() != 0) {
    single_qsort_axis(p, left, right, axis);
    return;
  }

#pragma omp parallel
  {
#pragma omp single nowait
    {
      para_qsort_internal_axis(p, left, right, axis);
    }
  }
}

#if 1

#include <vector>
#include <random>
#include "../utils.hpp"

int main(int argc, char **argv)
{
  int axis = atol(argv[1]);

  std::vector<target> targets(300);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);

  for(auto &t : targets) {
    t.xpos = dis(gen);
    t.ypos = dis(gen);
    t.zpos = dis(gen);
  }

  for(int i = 0; i < targets.size(); i++) {
    auto &t = targets[i];
    std::cout << i << " pos: " << t.xpos << ", " << t.ypos << ", " << t.zpos << std::endl;
  }

  single_qsort_axis(targets.data(), 0, targets.size() - 1, axis);

  std::cout << "sorted" << std::endl;
  for(int i = 0; i < targets.size(); i++) {
    auto &t = targets[i];
    std::cout << i << " pos: " << t.xpos << ", " << t.ypos << ", " << t.zpos << std::endl;
  }
}

#endif
