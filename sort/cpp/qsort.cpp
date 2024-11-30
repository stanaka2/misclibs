#include <algorithm>
#include <cstdint>
#include <omp.h>

#define para_sort_THRESHOLD (5000)
// #define para_sort_THRESHOLD (4)

/* ### median ### */
template <typename T>
static inline T median(const T a, const T b, const T c)
{
  return std::max(std::min(a, b), std::min(std::max(a, b), c));
}

/* ### qsort_partitioning ### */
template <typename T>
static inline void qsort_partitioning(T a[], int &left, int &right, T pivot)
{
  int ii = left;
  int jj = right;

  /* partitioning */
  while(1) {
    while(a[ii] < pivot) ii++;
    while(pivot < a[jj]) jj--;
    if(ii >= jj) break;
    std::swap(a[ii], a[jj]);
    ii++;
    jj--;
  }

  left = ii;
  right = jj;
}

/* ### single_qsort ### */
template <typename T>
static void single_qsort(T a[], int left, int right)
{
  if(left < right) {
    int ii = left, jj = right;
    const T pivot = median(a[ii], a[jj], a[(ii + jj) / 2]);

    qsort_partitioning(a, ii, jj, pivot);
    single_qsort(a, left, ii - 1);
    single_qsort(a, jj + 1, right);
  }
}

/* ### para_qsort_internal ### */
template <typename T>
static void para_qsort_internal(T a[], int left, int right)
{
  int length = right - left;
  if(length < para_sort_THRESHOLD) {
    single_qsort(a, left, right);
    return;
  }

  int ii = left, jj = right;
  const T pivot = median(a[ii], a[jj], a[(ii + jj) / 2]);

  qsort_partitioning(a, ii, jj, pivot);

#pragma omp task
  para_qsort_internal(a, left, jj);
#pragma omp task
  para_qsort_internal(a, ii, right);
}

/* ### para_qsort ### */
template <typename T>
static void para_qsort(T a[], int left, int right)
{
  if(omp_in_parallel() != 0) {
    single_qsort(a, left, right);
    return;
  }

#pragma omp parallel
  {
#pragma omp single nowait
    {
      para_qsort_internal(a, left, right);
    }
  }
}

#if 1
#include <vector>
#include <random>
#include "../../utils/utils.hpp"

int main(int argc, char **argv)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist_p(1, 100);
  std::vector<int> p(300);

  for(int i = 0; i < p.size(); ++i) {
    p[i] = dist_p(gen);
  }

  std::cout << "Before sorting:" << std::endl;
  printVec(p, "Array p");

  single_qsort(p.data(), 0, p.size() - 1);
  // para_qsort(p.data(), 0, p.size() - 1);

  std::cout << "After sorting:" << std::endl;
  printVec(p, "Array p");
}
#endif
