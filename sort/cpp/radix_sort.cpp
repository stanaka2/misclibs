#include <cstdint>
#include <limits>
#include <algorithm>
#include <cstring>

// https://stackoverflow.com/questions/42303108/how-can-i-use-radix-sort-for-an-array-of-float-numbers

inline uint32_t float_to_uint32(float value)
{
  uint32_t result;
  std::memcpy(&result, &value, sizeof(uint32_t));
  result = (result) ^ (((~(result) >> 31) - 1) | 0x80000000);
  return result;
}

inline float uint32_to_float(uint32_t value)
{
  value = (value) ^ ((((value) >> 31) - 1) | 0x80000000);
  float result;
  std::memcpy(&result, &value, sizeof(uint32_t));
  return result;
}

inline uint64_t double_to_uint64(double value)
{
  uint64_t result;
  std::memcpy(&result, &value, sizeof(uint64_t));
  result = (result) ^ (((~(result) >> 63) - 1) | 0x8000000000000000);
  return result;
}

inline double uint64_to_double(uint64_t value)
{
  value = (value) ^ ((((value) >> 63) - 1) | 0x8000000000000000);
  double result;
  std::memcpy(&result, &value, sizeof(uint64_t));
  return result;
}

template <typename T>
void radix_sort(T a[], const size_t left, const size_t right)
{
  if(right < left) return;
  const size_t length = right - left + 1;

  if constexpr(std::is_unsigned<T>::value || std::is_same<T, uint64_t>::value) {
    // Unsigned integer sorting
    const int bits = sizeof(T) * 8;
    const int radix = 256;
    T output[length];
    for(int shift = 0; shift < bits; shift += 8) {
      int count[radix] = {0};
      for(size_t i = 0; i < length; i++) count[(a[left + i] >> shift) & 0xFF]++;
      for(int i = 1; i < radix; i++) count[i] += count[i - 1];
      for(int i = length - 1; i >= 0; i--) output[--count[(a[left + i] >> shift) & 0xFF]] = a[left + i];
      for(size_t i = 0; i < length; i++) a[left + i] = output[i];
    }

  } else if constexpr(std::is_integral<T>::value && std::is_signed<T>::value) {
    // Signed integer sorting by treating as unsigned with offset
    using U = std::make_unsigned_t<T>;
    U t[length];
    const size_t shift = (sizeof(T) * 8 - 1);
    for(size_t i = 0; i < length; i++) t[i] = static_cast<U>(a[left + i] ^ (U(1) << shift));
    radix_sort(t, 0, length - 1);
    for(size_t i = 0; i < length; i++) a[left + i] = static_cast<T>(t[i] ^ (U(1) << shift));

  } else if constexpr(std::is_same<T, float>::value) {
    // Sort float by converting to uint32
    uint32_t t[length];
    for(size_t i = 0; i < length; i++) t[i] = float_to_uint32(a[left + i]);
    radix_sort(t, 0, length - 1);
    for(size_t i = 0; i < length; i++) a[left + i] = uint32_to_float(t[i]);

  } else if constexpr(std::is_same<T, double>::value) {
    // Sort double by converting to uint64
    uint64_t t[length];
    for(size_t i = 0; i < length; i++) t[i] = double_to_uint64(a[left + i]);
    radix_sort(t, 0, length - 1);
    for(size_t i = 0; i < length; i++) a[left + i] = uint64_to_double(t[i]);
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
  std::uniform_int_distribution<> dist_p(-10, 100);

#if 0
  std::vector<double> p(300);
  for(int i = 0; i < p.size(); ++i) {
    p[i] = dist_p(gen) * 0.1234;
  }
#else
  std::vector<int64_t> p(300);
  for(int i = 0; i < p.size(); ++i) {
    p[i] = dist_p(gen);
  }
#endif

  std::cout << "Before sorting:" << std::endl;
  printVec(p, "Array p");

  radix_sort(p.data(), 0, p.size() - 1);

  std::cout << "After sorting:" << std::endl;
  printVec(p, "Array p");

  return 0;
}
#endif
