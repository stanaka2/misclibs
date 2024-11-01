#include <stdint.h>
#include <omp.h>

#define para_qsort_THRESHOLD (2000)
// #define para_qsort_THRESHOLD (2)

/* ### median ### */
static inline int64_t median_i64(const int64_t a, const int64_t b, const int64_t c)
{
  if(a > b) {
    if(b > c) return b;
    else if(c > a) return a;
    else return c;

  } else {
    if(a > c) return a;
    else if(c > b) return b;
    else return c;
  }
}

/* ### qsort_partitioning ### */
static inline void qsort_partitioning_i64(int a[], int64_t key[], int *left, int *right, int64_t pivot)
{
  int ii, jj;
  ii = *left;
  jj = *right;

  /* partitioning */
  while(1) {
    while(key[ii] < pivot) ii++;
    while(pivot < key[jj]) jj--;
    if(ii >= jj) break;

    const int64_t tmpk = key[ii];
    const int tmpa = a[ii];

    key[ii] = key[jj];
    key[jj] = tmpk;

    a[ii] = a[jj];
    a[jj] = tmpa;

    ii++;
    jj--;
  }

  *left = ii;
  *right = jj;
}

/* ### single_qsort ### */
static void single_qsort_i64(int a[], int64_t key[], int left, int right)
{
  if(left < right) {
    int ii = left, jj = right;

    int64_t pivot = median(key[ii], key[jj], key[(ii + jj) / 2]);
    qsort_partitioning_i64(a, key, &ii, &jj, pivot);

    single_qsort_i64(a, key, left, ii - 1);
    single_qsort_i64(a, key, jj + 1, right);
  }
}

/* ### para_qsort_internal ### */
static void para_qsort_internal_i64(int a[], int64_t key[], int left, int right)
{
  int length = right - left;
  if(length < para_qsort_THRESHOLD) {
    single_qsort_i64(a, key, left, right);
    return;
  }

  int ii = left, jj = right;
  const int64_t pivot = median(key[ii], key[jj], key[(ii + jj) / 2]);
  qsort_partitioning_i64(a, key, &ii, &jj, pivot);
#pragma omp task
  para_qsort_internal_i64(a, key, left, jj);
#pragma omp task
  para_qsort_internal_i64(a, key, ii, right);
}

/* ### para_qsort ### */
static void para_qsort_i64(int a[], int64_t key[], int left, int right)
{
  if(omp_in_parallel() != 0) {
    single_qsort_i64(a, key, left, right);
    return;
  }

#pragma omp parallel
  {
#pragma omp single nowait
    {
      para_qsort_internal_i64(a, key, left, right);
    }
  }
}

#if 1
#include <stdio.h>
#include <stdlib.h>

#include <random>
#include "../utils.hpp"

#if 1

int main(int argc, char **argv)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist_p(1, 100);
  std::uniform_int_distribution<> dist_d(0, 3);

  std::vector<int> p(300);
  std::vector<int64_t> d(300);

  for(int i = 0; i < p.size(); ++i) {
    p[i] = dist_p(gen);
    d[i] = dist_d(gen);
  }

  std::cout << "Before sorting:" << std::endl;
  printVec(p, "Array p");
  printVec(d, "Array d");

  // single_qsort_key(p, d, 0, p.size()-1);
  para_qsort_key(p.data(), d.data(), 0, p.size() - 1);
  //  para_qsort_i64(p.data(), d.data(), 0, p.size()-1);

  std::cout << "After sorting:" << std::endl;
  printVec(p, "Array p");
  printVec(d, "Array d");
}
#else

int main(int argc, char **argv)
{
  int axis = atol(argv[1]);

  std::vector<target> targets(30);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);

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
#endif
