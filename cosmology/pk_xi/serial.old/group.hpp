#pragma once

#include <cstdint>
#include <algorithm>
#include <numeric>
#include <vector>
#include <random>

#include "cosm_tools.hpp"

#if 1
struct group {
  float xpos, ypos, zpos;
  float mass;
  int block_id; // for jackknife
};

#else
struct group {
  float xpos, ypos, zpos;
  float xvel, yvel, zvel;
  float mass;
  float pot;
  int block_id; // for jackknife
};
#endif

using glist = std::vector<group>;

class groupcatalog
{
public:
  double lbox = -1.0; // box size in Mpc/h
  double Om = 0.3;    // matter density parameter
  double Ol = 0.7;    // dark energy density parameter

  std::vector<uint64_t> select_idx;

  groupcatalog() = default;

  template <typename T>
  T wrap01(T);

  void select_all(uint64_t);
  void select_random_sampling(uint64_t, double = 1.0);

  template <typename T>
  void select_range(const std::vector<T> &, T, T);

  template <typename T>
  glist set_base_grp(const T &);

  template <typename T, typename U>
  void apply_RSD_shift(const T &, const U &, const std::string &, glist &);

  template <typename T, typename U>
  void apply_Gred_shift(const T &, const U &, const std::string &, glist &);

  glist set_random_group(uint64_t, int = 10);

private:
  template <typename T>
  std::vector<uint64_t> initial_select(const std::vector<T> &, T, T);
  template <typename T>
  std::vector<uint64_t> refine_select(const std::vector<T> &, T, T, const std::vector<uint64_t> &);

  void ensure_selection(uint64_t);
};

template <typename T>
T groupcatalog::wrap01(T x)
{
  return x - std::floor(x);
}

void groupcatalog::select_all(uint64_t n_total)
{
  select_idx.resize(n_total);
  std::iota(select_idx.begin(), select_idx.end(), 0);
  std::cerr << "# selected all: " << n_total << " ~ " << (int)(std::pow((double)n_total, 1.0 / 3.0)) << "^3\n";
}

void groupcatalog::select_random_sampling(uint64_t n_total, double ratio)
{
  uint64_t n_sample = n_total * ratio;

  select_idx.resize(n_total);
  std::iota(select_idx.begin(), select_idx.end(), 0);

  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(select_idx.begin(), select_idx.end(), g);

  select_idx.resize(n_sample);
  std::sort(select_idx.begin(), select_idx.end());
  std::cerr << "# select random: " << n_sample << " ~ " << (int)(std::pow((double)n_sample, 1.0 / 3.0)) << "^3\n";
}

template <typename T>
void groupcatalog::select_range(const std::vector<T> &data, T s1, T s2)
{
  if(select_idx.empty()) {
    select_idx = initial_select(data, s1, s2);
  } else {
    select_idx = refine_select(data, s1, s2, select_idx);
  }
}

template <typename T>
std::vector<uint64_t> groupcatalog::initial_select(const std::vector<T> &data, T s1, T s2)
{
  uint64_t n = data.size();
  std::vector<uint64_t> idx;

  std::cerr << "# full data: " << n << " ~ " << (int)(std::pow((double)n, 1.0 / 3.0)) << "^3\n";

#pragma omp parallel
  {
    std::vector<uint64_t> local_idx;
#pragma omp for schedule(static)
    for(uint64_t i = 0; i < n; i++) {
      if(s1 <= data[i] && data[i] <= s2) local_idx.push_back(i);
    }
#pragma omp critical
    idx.insert(idx.end(), local_idx.begin(), local_idx.end());
  }

  std::sort(idx.begin(), idx.end());
  std::cerr << "# selection data: " << idx.size() << " ~ " << (int)(std::pow((double)idx.size(), 1.0 / 3.0)) << "^3\n";
  return idx;
}

template <typename T>
std::vector<uint64_t> groupcatalog::refine_select(const std::vector<T> &data, T s1, T s2,
                                                  const std::vector<uint64_t> &base_idx)
{
  uint64_t n = base_idx.size();
  std::vector<uint64_t> idx;

#pragma omp parallel
  {
    std::vector<uint64_t> local_idx;
#pragma omp for schedule(static)
    for(uint64_t j = 0; j < n; j++) {
      auto i = base_idx[j];
      if(s1 <= data[i] && data[i] <= s2) local_idx.push_back(i);
    }
#pragma omp critical
    idx.insert(idx.end(), local_idx.begin(), local_idx.end());
  }

  std::sort(idx.begin(), idx.end());
  std::cerr << "# re-selection: " << idx.size() << " ~ " << (int)(std::pow((double)idx.size(), 1.0 / 3.0)) << "^3\n";
  return idx;
}

void groupcatalog::ensure_selection(uint64_t n_total)
{
  if(select_idx.empty()) {
    select_all(n_total);
  }
}

template <typename T>
glist groupcatalog::set_base_grp(const T &data)
{
  // Array of Structs (AoS) or flat array
  // data are data[i].pos or pos[i]
  constexpr bool is_aos = std::is_class_v<typename T::value_type>;
  uint64_t n = is_aos ? data.size() : (data.size() / 3);
  ensure_selection(n);

  uint64_t m = select_idx.size();
  glist grp(m);

#pragma omp parallel for
  for(uint64_t j = 0; j < m; j++) {
    uint64_t i = select_idx[j];
    if constexpr(is_aos) {
      grp[j].xpos = wrap01(data[i].pos[0] / lbox);
      grp[j].ypos = wrap01(data[i].pos[1] / lbox);
      grp[j].zpos = wrap01(data[i].pos[2] / lbox);
    } else {
      grp[j].xpos = wrap01(data[3 * i + 0] / lbox);
      grp[j].ypos = wrap01(data[3 * i + 1] / lbox);
      grp[j].zpos = wrap01(data[3 * i + 2] / lbox);
    }
  }

  return grp;
}

template <typename T, typename U>
void groupcatalog::apply_RSD_shift(const T &data, const U &alist, const std::string &los_axis, glist &grp)
{
  constexpr bool vel_is_aos = std::is_class_v<typename T::value_type>;
  constexpr bool a_is_scalar = std::is_arithmetic_v<U>;

  uint64_t n = select_idx.size();

  if(los_axis == "x") {
#pragma omp parallel for
    for(uint64_t j = 0; j < n; j++) {
      uint64_t i = select_idx[j];
      double a_val;
      if constexpr(a_is_scalar) a_val = alist;
      else a_val = alist[i];

      auto factor = 1.0 / (a_val * Ha(a_val, Om, Ol)) / lbox;

      if constexpr(vel_is_aos) {
        grp[j].xpos = wrap01(grp[j].xpos + data[i].velx * factor);
      } else {
        grp[j].xpos = wrap01(grp[j].xpos + data[3 * i] * factor);
      }
    }
  } else if(los_axis == "y") {
#pragma omp parallel for
    for(uint64_t j = 0; j < n; j++) {
      uint64_t i = select_idx[j];
      double a_val;
      if constexpr(a_is_scalar) a_val = alist;
      else a_val = alist[i];

      auto factor = 1.0 / (a_val * Ha(a_val, Om, Ol)) / lbox;

      if constexpr(vel_is_aos) {
        grp[j].ypos = wrap01(grp[j].ypos + data[i].vely * factor);
      } else {
        grp[j].ypos = wrap01(grp[j].ypos + data[3 * i + 1] * factor);
      }
    }
  } else if(los_axis == "z") {
#pragma omp parallel for
    for(uint64_t j = 0; j < n; j++) {
      uint64_t i = select_idx[j];
      double a_val;
      if constexpr(a_is_scalar) a_val = alist;
      else a_val = alist[i];

      auto factor = 1.0 / (a_val * Ha(a_val, Om, Ol)) / lbox;

      if constexpr(vel_is_aos) {
        grp[j].zpos = wrap01(grp[j].zpos + data[i].velz * factor);
      } else {
        grp[j].zpos = wrap01(grp[j].zpos + data[3 * i + 2] * factor);
      }
    }
  }
}

template <typename T, typename U>
void groupcatalog::apply_Gred_shift(const T &data, const U &alist, const std::string &los_axis, glist &grp)
{
  constexpr bool pot_is_aos = std::is_class_v<typename T::value_type>;
  constexpr bool a_is_scalar = std::is_arithmetic_v<U>;

  uint64_t n = select_idx.size();

  if(los_axis == "x") {
#pragma omp parallel for
    for(uint64_t j = 0; j < n; j++) {
      uint64_t i = select_idx[j];
      double a_val;
      if constexpr(a_is_scalar) a_val = alist;
      else a_val = alist[i];

      auto factor = cspeed / (a_val * Ha(a_val, Om, Ol)) / lbox;

      if constexpr(pot_is_aos) {
        grp[j].xpos = wrap01(grp[j].xpos - data[i].pot * factor);
      } else {
        grp[j].xpos = wrap01(grp[j].xpos - data[i] * factor);
      }
    }
  } else if(los_axis == "y") {
#pragma omp parallel for
    for(uint64_t j = 0; j < n; j++) {
      uint64_t i = select_idx[j];
      double a_val;
      if constexpr(a_is_scalar) a_val = alist;
      else a_val = alist[i];

      auto factor = cspeed / (a_val * Ha(a_val, Om, Ol)) / lbox;

      if constexpr(pot_is_aos) {
        grp[j].ypos = wrap01(grp[j].ypos - data[i].pot[i] * factor);
      } else {
        grp[j].ypos = wrap01(grp[j].ypos - data[i] * factor);
      }
    }
  } else if(los_axis == "z") {
#pragma omp parallel for
    for(uint64_t j = 0; j < n; j++) {
      uint64_t i = select_idx[j];
      double a_val;
      if constexpr(a_is_scalar) a_val = alist;
      else a_val = alist[i];

      auto factor = cspeed / (a_val * Ha(a_val, Om, Ol)) / lbox;

      if constexpr(pot_is_aos) {
        grp[j].zpos = wrap01(grp[j].zpos - data[i].pot * factor);
      } else {
        grp[j].zpos = wrap01(grp[j].zpos - data[i] * factor);
      }
    }
  }
}

glist groupcatalog::set_random_group(uint64_t nrand, int seed)
{
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  glist rand(nrand);
  for(uint64_t i = 0; i < nrand; i++) {
    rand[i].xpos = dist(rng);
    rand[i].ypos = dist(rng);
    rand[i].zpos = dist(rng);
  }

  std::cerr << "# set random group pos: " << nrand << " ~ " << (int)(std::pow((double)nrand, 1.0 / 3.0)) << "^3\n";
  return rand;
}
