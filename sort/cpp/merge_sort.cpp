#include <vector>
#include <iostream>
#include <omp.h>

// #define para_sort_THRESHOLD (5000)
#define para_sort_THRESHOLD (4)

template <typename T>
static inline void merge(T a[], int left, int mid, int right)
{
  T temp[right - left + 1]; // use stack memory
  int i = left;
  int j = mid + 1;
  int k = 0;

  while(i <= mid && j <= right) {
    if(a[i] <= a[j]) {
      temp[k++] = a[i++];
    } else {
      temp[k++] = a[j++];
    }
  }

  while(i <= mid) temp[k++] = a[i++];
  while(j <= right) temp[k++] = a[j++];

  for(i = left, k = 0; i <= right; ++i, ++k) {
    a[i] = temp[k];
  }
}

template <typename T>
void merge_inplace(T a[], int left, int mid, int right)
{
  int i = left;
  int j = mid + 1;

  while(i <= mid && j <= right) {
    if(a[i] <= a[j]) {
      i++;
    } else {
      T temp = a[j];
      int idx = j;

      while(idx != i) {
        a[idx] = a[idx - 1];
        idx--;
      }
      a[i] = temp;

      mid++;
      i++;
      j++;
    }
  }
}

template <typename T>
void single_merge_sort(T a[], int left, int right)
{
  if(left < right) {
    int mid = left + (right - left) / 2;
    single_merge_sort(a, left, mid);
    single_merge_sort(a, mid + 1, right);
    merge(a, left, mid, right);
  }
}

template <typename T>
void single_merge_sort_inplace(T a[], int left, int right)
{
  if(left < right) {
    int mid = left + (right - left) / 2;
    single_merge_sort_inplace(a, left, mid);
    single_merge_sort_inplace(a, mid + 1, right);
    merge_inplace(a, left, mid, right);
  }
}

template <typename T>
void para_merge_sort_internal(T a[], int left, int right)
{
  int length = right - left;
  if(length < para_sort_THRESHOLD) {
    single_merge_sort(a, left, right);
    return;
  }

  if(left < right) {
    int mid = left + (right - left) / 2;
#pragma omp task
    para_merge_sort_internal(a, left, mid);
#pragma omp task
    para_merge_sort_internal(a, mid + 1, right);

#pragma omp taskwait
    merge(a, left, mid, right);
  }
}

template <typename T>
void para_merge_sort(T a[], int left, int right)
{
  if(omp_in_parallel() != 0) {
    single_merge_sort(a, left, right);
    return;
  }

#pragma omp parallel
  {
#pragma omp single nowait
    {
      para_merge_sort_internal(a, left, right);
    }
  }
}

template <typename T>
void para_merge_sort_inplace_internal(T a[], int left, int right)
{
  int length = right - left;
  if(length < para_sort_THRESHOLD) {
    single_merge_sort_inplace(a, left, right);
    return;
  }

  if(left < right) {
    int mid = left + (right - left) / 2;
#pragma omp task
    para_merge_sort_inplace_internal(a, left, mid);
#pragma omp task
    para_merge_sort_inplace_internal(a, mid + 1, right);

#pragma omp taskwait
    merge_inplace(a, left, mid, right);
  }
}

template <typename T>
void para_merge_sort_inplace(T a[], int left, int right)
{
  if(omp_in_parallel() != 0) {
    single_merge_sort_inplace(a, left, right);
    return;
  }

#pragma omp parallel
  {
#pragma omp single nowait
    {
      para_merge_sort_inplace_internal(a, left, right);
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

  // single_merge_sort(p.data(), 0, p.size() - 1);
  // single_merge_sort_inplace(p.data(), 0, p.size() - 1);
  // para_merge_sort(p.data(), 0, p.size() - 1);
  para_merge_sort_inplace(p.data(), 0, p.size() - 1);

  std::cout << "After sorting:" << std::endl;
  printVec(p, "Array p");
}
#endif
