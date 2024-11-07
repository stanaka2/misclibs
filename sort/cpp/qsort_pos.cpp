#include <algorithm>
#include <type_traits>
#include <cstdint>
#include <omp.h>

// support two type position member
struct target_xyz {
  int id;
  double xpos, ypos, zpos;
};

struct target_pos {
  int id;
  double pos[3];
};

#define para_sort_THRESHOLD (5000)
// #define para_sort_THRESHOLD (2)

template <typename T, typename = void>
struct has_xpos : std::false_type {
};

template <typename T>
struct has_xpos<T, std::void_t<decltype(std::declval<T>().xpos)>> : std::true_type {
};

template <typename T, typename = void>
struct has_pos_array : std::false_type {
};

template <typename T>
struct has_pos_array<T, std::void_t<decltype(std::declval<T>().pos[0])>> : std::true_type {
};

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
  if constexpr(has_xpos<T>::value) {
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

  } else if constexpr(has_pos_array<T>::value) {
    while(1) {
      while(p[ii].pos[axis] < pivot) ii++;
      while(pivot < p[jj].pos[axis]) jj--;
      if(ii >= jj) break;
      std::swap(p[ii], p[jj]);
      ii++;
      jj--;
    }

  } else {
    static_assert(has_xpos<T>::value || has_pos_array<T>::value, "T must have either xpos/ypos/zpos or pos[3]");
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
    if constexpr(has_xpos<T>::value) {
      if(axis == 0) pivot = median(p[ii].xpos, p[jj].xpos, p[(ii + jj) / 2].xpos);
      else if(axis == 1) pivot = median(p[ii].ypos, p[jj].ypos, p[(ii + jj) / 2].ypos);
      else pivot = median(p[ii].zpos, p[jj].zpos, p[(ii + jj) / 2].zpos);
    } else if constexpr(has_pos_array<T>::value) {
      pivot = median(p[ii].pos[axis], p[jj].pos[axis], p[(ii + jj) / 2].pos[axis]);
    } else {
      static_assert(has_xpos<T>::value || has_pos_array<T>::value, "T must have either xpos/ypos/zpos or pos[3]");
    }

    qsort_partitioning_axis(p, ii, jj, pivot, axis);
    single_qsort_axis(p, left, ii - 1, axis);
    single_qsort_axis(p, jj + 1, right, axis);
  }
}

template <typename T, typename U>
static void para_qsort_internal_axis(U p[], int left, int right, const int axis)
{
  int length = right - left;
  if(length < para_sort_THRESHOLD) {
    single_qsort_axis(p, left, right, axis);
    return;
  }

  int ii = left, jj = right;
  double pivot;
  if constexpr(has_xpos<T>::value) {
    if(axis == 0) pivot = median(p[ii].xpos, p[jj].xpos, p[(ii + jj) / 2].xpos);
    else if(axis == 1) pivot = median(p[ii].ypos, p[jj].ypos, p[(ii + jj) / 2].ypos);
    else pivot = median(p[ii].zpos, p[jj].zpos, p[(ii + jj) / 2].zpos);
  } else if constexpr(has_pos_array<T>::value) {
    pivot = median(p[ii].pos[axis], p[jj].pos[axis], p[(ii + jj) / 2].pos[axis]);
  } else {
    static_assert(has_xpos<T>::value || has_pos_array<T>::value, "T must have either xpos/ypos/zpos or pos[3]");
  }

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
#include "../../utils/utils.hpp"

int main(int argc, char **argv)
{
  int axis = atol(argv[1]);

#if 1
  std::vector<target_xyz> targets(300);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);

  int a = 0;
  for(auto &t : targets) {
    t.id = a++;
    t.xpos = dis(gen);
    t.ypos = dis(gen);
    t.zpos = dis(gen);
  }

  for(int i = 0; i < targets.size(); i++) {
    auto &t = targets[i];
    std::cout << i << " " << t.id << " pos: " << t.xpos << ", " << t.ypos << ", " << t.zpos << std::endl;
  }

  single_qsort_axis(targets.data(), 0, targets.size() - 1, axis);

  std::cout << "sorted" << std::endl;
  for(int i = 0; i < targets.size(); i++) {
    auto &t = targets[i];
    std::cout << i << " " << t.id << " pos: " << t.xpos << ", " << t.ypos << ", " << t.zpos << std::endl;
  }

#else

  std::vector<target_pos> targets(300);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);

  int a = 0;
  for(auto &t : targets) {
    t.id = a++;
    t.pos[0] = dis(gen);
    t.pos[1] = dis(gen);
    t.pos[2] = dis(gen);
  }

  for(int i = 0; i < targets.size(); i++) {
    auto &t = targets[i];
    std::cout << i << " " << t.id << " pos: " << t.pos[0] << ", " << t.pos[1] << ", " << t.pos[2] << std::endl;
  }

  single_qsort_axis(targets.data(), 0, targets.size() - 1, axis);

  std::cout << "sorted" << std::endl;
  for(int i = 0; i < targets.size(); i++) {
    auto &t = targets[i];
    std::cout << i << " " << t.id << " pos: " << t.pos[0] << ", " << t.pos[1] << ", " << t.pos[2] << std::endl;
  }

#endif
}

#endif
