#pragma once

#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cassert>

#include <fftw3.h>
#include "group.hpp"
#include "powerspec.hpp"
#include "util.hpp"

class correlation : public powerspec
{
public:
  int nr = 100;
  double rmin, rmax;
  std::vector<float> rcen; // bin center

  double rmin2, rmax2;    // r, r^2 base
  double lin_dr, lin_dr2; // r, r^2 base

  /* radius of searching cell : (ndiv_1d+2)^3 */
  // int ndiv_1d = 1;
  int ndiv_1d = 4;

  int jk_type = 0; /* 0: spaced-block, 1:shuffle-block */
  int jk_block, jk_level;
  int jk_dd, jk_dd2, nblock;

  bool symmetry = true; // true: count each pair once (i<j only)
                        // false: count both directions

  int nmu;
  double mumin, mumax, dmu;
  std::vector<float> mucen; // bin center

  int nsperp, nspara;
  double sperp_min, sperp_max, dsperp;
  double spara_min, spara_max, dspara;
  std::vector<float> sperpcen, sparacen; // bin center

  // for position xi
  // support Landy SD, Szalay AS 1993, Apj
  std::string estimator = "ideal"; // "ideal", "RR", "LS"
  bool use_LS = false;
  int64_t nrand_factor = 1;
  std::vector<double> dd_pair, dr_pair, dr2_pair, rr_pair;                          // size[nr]
  std::vector<std::vector<double>> dd_pair_jk, dr_pair_jk, dr2_pair_jk, rr_pair_jk; // size[block][nr]
  std::vector<double> xi, xi_ave, xi_sd, xi_se;                                     // size[nr]
  std::vector<std::vector<double>> xi_jk;                                           // size[block][nr]

  std::vector<int64_t> block_start, block_start2, block_end, block_end2;
  std::vector<int64_t> rand_block_start, rand_block_start2, rand_block_end, rand_block_end2;

  void set_rbin(double, double, int, double, bool = true);
  void check_rbin();
  void set_smubin(double, double, int, double, double, int, double);
  void set_spspbin(double, double, int, double, double, int, double);

  template <typename T>
  void calc_xi(T &);
  template <typename T>
  void calc_xi(T &, T &);
  template <typename T>
  void calc_xi_ifft(T &, T &);
  template <typename T>
  void calc_xi_ifft(T &, T &, T &);

  template <typename T>
  void calc_xi_smu(T &);
  template <typename T>
  void calc_xi_spsp(T &);

  void output_xi(std::string);
  template <typename T>
  void output_xi(std::string, T &);
  void output_xi_smu(std::string);
  void output_xi_spsp(std::string);

private:
  template <typename T>
  int get_r_index(T);
  template <typename T>
  int get_r2_index(T);
  template <typename T>
  int get_ir_form_dr(T, T, T);
  template <typename T>
  int get_imu_form_dr(T, T, T);
  template <typename T>
  int get_cell_index(T, int);

  template <typename T>
  void shuffle_data(T &, const int = 10);

  void calc_ideal_pair(uint64_t);
  void calc_ideal_smu_pair(uint64_t);
  void calc_ideal_spsp_pair(uint64_t);
  void calc_xi_from_pair(uint64_t);
  void calc_xi_from_pair(uint64_t, uint64_t);

  template <typename G, typename C>
  std::vector<double> calc_pair(const double, G &, C &, int, int, int, const std::string);
  template <typename G, typename C>
  std::vector<double> calc_pair(const double, G &, G &, C &, C &, int, int, int, const std::string);

  template <typename G, typename C>
  std::vector<double> calc_pair_smu(const double, G &, C &, int, int, int, const std::string);
  template <typename G, typename C>
  std::vector<double> calc_pair_smu(const double, G &, G &, C &, C &, int, int, int, const std::string);

  template <typename G, typename C>
  std::vector<double> calc_pair_spsp(const double, G &, C &, int, int, int, const std::string);
  template <typename G, typename C>
  std::vector<double> calc_pair_spsp(const double, G &, G &, C &, C &, int, int, int, const std::string);

  template <typename T>
  void calc_xi_impl(T &);
  template <typename T>
  void calc_xi_smu_impl(T &);
  template <typename T>
  void calc_xi_spsp_impl(T &);

  template <typename T>
  void calc_xi_impl(T &, T &);
  template <typename T>
  void calc_xi_LS_impl(T &);
  template <typename T>
  void calc_xi_LS_impl(T &, T &);
  template <typename T>
  void calc_xi_jk_impl(T &);
  template <typename T>
  void calc_xi_jk_impl(T &, T &);
  template <typename T>
  void calc_xi_jk_LS_impl(T &);
  template <typename T>
  void calc_xi_jk_LS_impl(T &, T &);
  template <typename T>
  void calc_xi_ifft_impl(T &, T &);
  template <typename T>
  void calc_xi_ifft_impl(T &, T &, T &);
  template <typename T>
  void calc_xi_jk_ifft_impl(T &, T &);
  template <typename T>
  void calc_xi_jk_ifft_impl(T &, T &, T &);

  template <typename T>
  void sort_jk_block(T &);
  template <typename T>
  void sort_jk_rand_block(T &);
  template <typename T>
  void set_block_edge_id(T &);
  template <typename T>
  void set_block_edge_id(T &, T &);
  template <typename T>
  void set_rand_block_edge_id(T &);
  template <typename T>
  void set_rand_block_edge_id(T &, T &);

  void set_block_edge_id_shuffle(uint64_t);
  void set_block_edge_id_shuffle(uint64_t, uint64_t);
  void set_rand_block_edge_id_shuffle(uint64_t);
  void set_rand_block_edge_id_shuffle(uint64_t, uint64_t);

  template <typename T>
  void resample_jk(T &);
  template <typename T>
  void resample_jk(T &, T &);
  template <typename T>
  void resample_jk_LS(T &, T &);
  template <typename T>
  void resample_jk_LS(T &, T &, T &, T &);

  void calc_jk_xi_average();
  void calc_jk_xi_error();
};

void correlation::set_rbin(double _rmin, double _rmax, int _nr, double _lbox, bool _log_scale)
{
  nr = _nr;
  lbox = _lbox;
  rmin = _rmin / lbox;
  rmax = _rmax / lbox;
  log_scale = _log_scale;

  rmin2 = rmin * rmin;
  rmax2 = rmax * rmax;

  rcen.assign(nr, 0.0);

  if(log_scale) {
    if(rmin < 1e-10) rmin = 1e-10;
    ratio = pow(rmax / rmin, 1.0 / (double)(nr));
    logratio = log(ratio);
    logratio2 = 2 * logratio;
    for(int ir = 0; ir < nr; ir++) rcen[ir] = rmin * pow(ratio, ir + 0.5);

  } else {
    lin_dr = (rmax - rmin) / (double)(nr);
    for(int ir = 0; ir < nr; ir++) rcen[ir] = rmin + lin_dr * (ir + 0.5);
  }
}

void correlation::check_rbin()
{
  for(int ir = 0; ir < nr; ir++) std::cerr << ir << " " << rcen[ir] << " " << get_r_index(rcen[ir]) << "\n";
}

void correlation::set_smubin(double _rmin, double _rmax, int _nr, double _mumin, double _mumax, int _nmu, double _lbox)
{
  log_scale = false;
  lbox = _lbox;

  nr = _nr;
  rmin = _rmin / lbox;
  rmax = _rmax / lbox;
  rmin2 = rmin * rmin;
  rmax2 = rmax * rmax;

  rcen.assign(nr, 0.0);
  lin_dr = (rmax - rmin) / (double)(nr);
  for(int ir = 0; ir < nr; ir++) rcen[ir] = rmin + lin_dr * (ir + 0.5);

  nmu = _nmu;
  mumax = _mumax;
  mumin = _mumin;

  mucen.assign(nmu, 0.0);
  dmu = (mumax - mumin) / (double)nmu;
  for(int imu = 0; imu < nmu; imu++) mucen[imu] = mumin + (imu + 0.5) * dmu;
}

void correlation::set_spspbin(double _sperp_min, double _sperp_max, int _nsperp, double _spara_min, double _spara_max,
                              int _nspara, double _lbox)
{
  /*
  rmin < sperp < rmax
  and
  -rmax < spara < rmax  or  0 < spara < rmax
  */

  log_scale = false;
  lbox = _lbox;

  // r is required as well
  rmin = _sperp_min / lbox;
  rmax = _sperp_max / lbox;
  rmin2 = rmin * rmin;
  rmax2 = rmax * rmax;

  nsperp = _nsperp;
  sperp_min = rmin;
  sperp_max = rmax;
  dsperp = (sperp_max - sperp_min) / (double)(nsperp);
  sperpcen.assign(nsperp, 0.0);
  for(int ir = 0; ir < nsperp; ir++) sperpcen[ir] = sperp_min + dsperp * (ir + 0.5);

  nspara = _nspara;
  spara_min = _spara_min / lbox;
  spara_max = _spara_max / lbox;
  dspara = (spara_max - spara_min) / (double)(nspara);
  sparacen.assign(nspara, 0.0);
  for(int ir = 0; ir < nspara; ir++) sparacen[ir] = spara_min + dspara * (ir + 0.5);
}

template <typename T>
int correlation::get_r_index(T r)
{
  if(r < rmin || r >= rmax) return -1;
  if(log_scale) return std::floor(log(r / rmin) / logratio);
  else return std::floor((r - rmin) / lin_dr);
  return -1;
}

template <typename T>
int correlation::get_r2_index(T r2)
{
  if(r2 < rmin2 || r2 >= rmax2) return -1;
  if(log_scale) return std::floor(log(r2 / rmin2) / logratio2);
  else return std::floor((std::sqrt(r2) - rmin) / lin_dr);
  return -1;
}

template <typename T>
int correlation::get_ir_form_dr(T dx, T dy, T dz)
{
  /*
  dx = (dx > 0.5 ? dx - 1.e0 : dx);
  dy = (dy > 0.5 ? dy - 1.e0 : dy);
  dz = (dz > 0.5 ? dz - 1.e0 : dz);
  dx = (dx < -0.5 ? dx + 1.e0 : dx);
  dy = (dy < -0.5 ? dy + 1.e0 : dy);
  dz = (dz < -0.5 ? dz + 1.e0 : dz);
  */
  dx -= std::nearbyint(dx); // assumes unit box (Lbox = 1)
  dy -= std::nearbyint(dy);
  dz -= std::nearbyint(dz);

  const double dr2 = dx * dx + dy * dy + dz * dz;
  int ir = get_r2_index(dr2);
  return ir;
}

template <typename T>
int correlation::get_imu_form_dr(T dx, T dy, T dz)
{
  dx -= std::nearbyint(dx);
  dy -= std::nearbyint(dy);
  dz -= std::nearbyint(dz);
  const double dr2 = dx * dx + dy * dy + dz * dz;
  double mu = dz / std::sqrt(dr2);
  int imu = std::floor((mu - mumin) / dmu);
  return (imu >= 0 && imu < nmu) ? imu : -1;
}

template <typename T>
int correlation::get_cell_index(T pos, int nc)
{
  int idx = static_cast<int>(pos * nc);
  if(idx < 0) idx += nc;
  if(idx >= nc) idx -= nc;
  return idx;
}

template <typename T>
void correlation::shuffle_data(T &grp, const int seed)
{
  uint64_t ngrp = grp.size();
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<uint64_t> dist(0, ngrp - 1);

  for(uint64_t i = 0; i < ngrp; i++) {
    uint64_t irand = dist(rng);
    std::swap(grp[i], grp[irand]);
  }
}

template <typename T>
void correlation::sort_jk_block(T &grp)
{
  uint64_t ngrp = grp.size();

#pragma omp parallel for
  for(uint64_t i = 0; i < ngrp; i++) {
    const int ix = get_cell_index(grp[i].xpos, jk_level);
    const int iy = get_cell_index(grp[i].ypos, jk_level);
    const int iz = get_cell_index(grp[i].zpos, jk_level);
    grp[i].block_id = iz + jk_level * (iy + jk_level * ix);
  }
  std::sort(grp.begin(), grp.end(), [](const group &a, const group &b) { return a.block_id < b.block_id; });
}

template <typename T>
void correlation::sort_jk_rand_block(T &rand)
{
  uint64_t nrand = rand.size();

#pragma omp parallel for
  for(uint64_t i = 0; i < nrand; i++) {
    const int ix = get_cell_index(rand[i].xpos, jk_level);
    const int iy = get_cell_index(rand[i].ypos, jk_level);
    const int iz = get_cell_index(rand[i].zpos, jk_level);
    rand[i].block_id = iz + jk_level * (iy + jk_level * ix);
  }
  std::sort(rand.begin(), rand.end(), [](const group &a, const group &b) { return a.block_id < b.block_id; });
}

template <typename T>
void correlation::set_block_edge_id(T &grp)
{
  uint64_t ngrp = grp.size();
  block_start.assign(nblock, -1);
  block_end.assign(nblock, -1);

  for(uint64_t i = 0; i < ngrp; i++) {
    auto bid = grp[i].block_id;
    if(block_start[bid] == -1) {
      block_start[bid] = i;
    }
    block_end[bid] = i + 1;
  }
}

template <typename T>
void correlation::set_block_edge_id(T &grp1, T &grp2)
{
  uint64_t ngrp1 = grp1.size();
  uint64_t ngrp2 = grp2.size();

  block_start.assign(nblock, -1);
  block_end.assign(nblock, -1);
  block_start2.assign(nblock, -1);
  block_end2.assign(nblock, -1);

  for(uint64_t i = 0; i < ngrp1; i++) {
    auto bid = grp1[i].block_id;
    if(block_start[bid] == -1) {
      block_start[bid] = i;
    }
    block_end[bid] = i + 1;
  }

  for(uint64_t i = 0; i < ngrp2; i++) {
    auto bid = grp2[i].block_id;
    if(block_start2[bid] == -1) {
      block_start2[bid] = i;
    }
    block_end2[bid] = i + 1;
  }
}

template <typename T>
void correlation::set_rand_block_edge_id(T &rand)
{
  uint64_t nrand = rand.size();

  rand_block_start.assign(nblock, -1);
  rand_block_end.assign(nblock, -1);

  for(uint64_t i = 0; i < nrand; i++) {
    auto bid = rand[i].block_id;
    if(rand_block_start[bid] == -1) {
      rand_block_start[bid] = i;
    }
    rand_block_end[bid] = i + 1;
  }
}

template <typename T>
void correlation::set_rand_block_edge_id(T &rand1, T &rand2)
{
  uint64_t nrand1 = rand1.size();
  uint64_t nrand2 = rand2.size();

  rand_block_start.assign(nblock, -1);
  rand_block_end.assign(nblock, -1);
  rand_block_start2.assign(nblock, -1);
  rand_block_end2.assign(nblock, -1);

  for(uint64_t i = 0; i < nrand1; i++) {
    auto bid = rand1[i].block_id;
    if(rand_block_start[bid] == -1) {
      rand_block_start[bid] = i;
    }
    rand_block_end[bid] = i + 1;
  }

  for(uint64_t i = 0; i < nrand2; i++) {
    auto bid = rand2[i].block_id;
    if(rand_block_start2[bid] == -1) {
      rand_block_start2[bid] = i;
    }
    rand_block_end2[bid] = i + 1;
  }
}

void correlation::set_block_edge_id_shuffle(uint64_t ngrp)
{
  jk_dd = ngrp / jk_block;
  nblock = (int)ceil(ngrp / jk_dd);

  block_start.assign(nblock, -1);
  block_end.assign(nblock, -1);

  int64_t length = (ngrp - jk_dd) / (nblock - 1);
  for(int iblock = 0; iblock < nblock; iblock++) {
    block_start[iblock] = iblock * length;
    block_end[iblock] = block_start[iblock] + jk_dd;
  }
}

void correlation::set_block_edge_id_shuffle(uint64_t ngrp1, uint64_t ngrp2)
{
  jk_dd = ngrp1 / jk_block;
  jk_dd2 = ngrp2 / jk_block;

  block_start.assign(nblock, -1);
  block_end.assign(nblock, -1);
  block_start2.assign(nblock, -1);
  block_end2.assign(nblock, -1);

  int64_t length1 = (ngrp1 - jk_dd) / (nblock - 1);
  int64_t length2 = (ngrp2 - jk_dd2) / (nblock - 1);

  for(int iblock = 0; iblock < nblock; iblock++) {
    block_start[iblock] = iblock * length1;
    block_end[iblock] = block_start[iblock] + jk_dd;

    block_start2[iblock] = iblock * length2;
    block_end2[iblock] = block_start2[iblock] + jk_dd2;
  }
}

void correlation::set_rand_block_edge_id_shuffle(uint64_t nrand)
{
  jk_dd = nrand / jk_block;
  rand_block_start.assign(nblock, -1);
  rand_block_end.assign(nblock, -1);

  int64_t length = (nrand - jk_dd) / (nblock - 1);
  for(int iblock = 0; iblock < nblock; iblock++) {
    rand_block_start[iblock] = iblock * length;
    rand_block_end[iblock] = rand_block_start[iblock] + jk_dd;
  }
}

void correlation::set_rand_block_edge_id_shuffle(uint64_t nrand1, uint64_t nrand2)
{
  jk_dd = nrand1 / jk_block;
  jk_dd2 = nrand2 / jk_block;

  rand_block_start.assign(nblock, -1);
  rand_block_end.assign(nblock, -1);
  rand_block_start2.assign(nblock, -1);
  rand_block_end2.assign(nblock, -1);

  int64_t length1 = (nrand1 - jk_dd) / (nblock - 1);
  int64_t length2 = (nrand2 - jk_dd2) / (nblock - 1);

  for(int iblock = 0; iblock < nblock; iblock++) {
    rand_block_start[iblock] = iblock * length1;
    rand_block_end[iblock] = rand_block_start[iblock] + jk_dd;

    rand_block_start2[iblock] = iblock * length2;
    rand_block_end2[iblock] = rand_block_start2[iblock] + jk_dd2;
  }
}

void correlation::calc_ideal_pair(uint64_t npair)
{
  auto nxi = dd_pair.size();
  rr_pair.assign(nxi, 0.0);

  // double V_box = lbox * lbox * lbox;
  double V_box = 1.0;

#pragma omp parallel for
  for(int ir = 0; ir < nr; ir++) {

    double r_low, r_high;
    if(log_scale) {
      r_low = rmin * pow(ratio, ir);
      r_high = rmin * pow(ratio, ir + 1);
    } else {
      r_low = rmin + lin_dr * ir;
      r_high = rmin + lin_dr * (ir + 1);
    }

    auto shell_volume = (4.0 / 3.0) * M_PI;
    shell_volume *= (r_high * r_high * r_high - r_low * r_low * r_low);

    auto norm = npair * shell_volume / V_box;
    rr_pair[ir] = norm;
  }
}

void correlation::calc_ideal_smu_pair(uint64_t npair)
{
  auto nxi = dd_pair.size();
  rr_pair.assign(nxi, 0.0);

  double V_box = 1.0;
  double mu_range = mumax - mumin;

#pragma omp parallel for
  for(int ir = 0; ir < nr; ir++) {

    double r_low, r_high;
    if(log_scale) {
      r_low = rmin * pow(ratio, ir);
      r_high = rmin * pow(ratio, ir + 1);
    } else {
      r_low = rmin + lin_dr * ir;
      r_high = rmin + lin_dr * (ir + 1);
    }

    auto shell_volume = (4.0 / 3.0) * M_PI;
    shell_volume *= (r_high * r_high * r_high - r_low * r_low * r_low);

    auto norm = npair * shell_volume / V_box;
    norm *= (0.5 * mu_range);

    for(int imu = 0; imu < nmu; imu++) {
      rr_pair[ir * nmu + imu] = norm / static_cast<double>(nmu);
    }
  }
}

void correlation::calc_ideal_spsp_pair(uint64_t npair)
{
  auto nxi = dd_pair.size();
  rr_pair.assign(nxi, 0.0);

  double V_box = 1.0;

#pragma omp parallel for collapse(2)
  for(int is = 0; is < nsperp; is++) {
    for(int ip = 0; ip < nspara; ip++) {

      double s_perp_low = sperp_min + dsperp * is;
      double s_perp_high = sperp_min + dsperp * (is + 1);

      double s_para_low = spara_min + dspara * ip;
      double s_para_high = spara_min + dspara * (ip + 1);

      double cyl_volume = M_PI;
      cyl_volume *= (s_perp_high * s_perp_high - s_perp_low * s_perp_low);
      cyl_volume *= (s_para_high - s_para_low);

      double norm = npair * cyl_volume / V_box;
      rr_pair[is * nspara + ip] = norm;
    }
  }
}

void correlation::calc_xi_from_pair(uint64_t ngrp)
{
  auto nxi = dd_pair.size();
  auto nrand = ngrp * nrand_factor;
  auto f = (double)nrand / (double)ngrp;
  auto f2 = f * f;

  if(estimator == "ideal") {
    for(int i = 0; i < nxi; i++) {
      if(rr_pair[i] != 0.0) {
        xi[i] = dd_pair[i] / rr_pair[i] - 1.0;
      }
    }

  } else if(estimator == "RR") {
    for(int i = 0; i < nxi; i++) {
      if(rr_pair[i] != 0.0) {
        xi[i] = f2 * dd_pair[i] / rr_pair[i] - 1.0;
      }
    }

  } else if(estimator == "LS") {
    for(int i = 0; i < nxi; i++) {
      if(rr_pair[i] != 0.0) {
        xi[i] = (f2 * dd_pair[i] - 2.0 * f * dr_pair[i] + rr_pair[i]) / rr_pair[i];
      }
    }
  }
}

void correlation::calc_xi_from_pair(uint64_t ngrp1, uint64_t ngrp2)
{
  assert(estimator != "ideal");

  auto nxi = dd_pair.size();
  auto nrand1 = ngrp1 * nrand_factor;
  auto nrand2 = ngrp2 * nrand_factor;
  auto f1 = (double)nrand1 / (double)ngrp1;
  auto f2 = (double)nrand2 / (double)ngrp2;
  auto f12 = f1 * f2;

  if(estimator == "RR") {
    for(int i = 0; i < nxi; i++) {
      if(rr_pair[i] != 0.0) {
        xi[i] = f12 * dd_pair[i] / rr_pair[i] - 1.0;
      }
    }

  } else if(estimator == "LS") {
    for(int i = 0; i < nxi; i++) {
      if(rr_pair[i] != 0.0) {
        xi[i] = (f12 * dd_pair[i] - (f1 * dr_pair[i] + f2 * dr2_pair[i]) + rr_pair[i]) / rr_pair[i];
      }
    }
  }
}

template <typename T>
void correlation::calc_xi(T &grp)
{
    if(jk_block > 1) calc_xi_jk_impl(grp);
    else calc_xi_impl(grp);
}

template <typename T>
void correlation::calc_xi(T &grp1, T &grp2)
{
  if(grp1.data() == grp2.data()) {
    calc_xi(grp1);
    return;
  }

    if(jk_block > 1) calc_xi_jk_impl(grp1, grp2);
    else calc_xi_impl(grp1, grp2);
}

template <typename T>
void correlation::calc_xi_smu(T &grp)
{
  calc_xi_smu_impl(grp);
}

template <typename T>
void correlation::calc_xi_spsp(T &grp)
{
  calc_xi_spsp_impl(grp);
}

template <typename T>
void correlation::calc_xi_ifft(T &mesh, T &weight)
{
  if(jk_block > 1) calc_xi_jk_ifft_impl(mesh, weight);
  else calc_xi_ifft_impl(mesh, weight);
}

template <typename T>
void correlation::calc_xi_ifft(T &mesh1, T &mesh2, T &weight)
{
  if(mesh1.data() == mesh2.data()) {
    calc_xi_ifft(mesh1, weight);
    return;
  }

  if(jk_block > 1) calc_xi_jk_ifft_impl(mesh1, mesh2, weight);
  else calc_xi_ifft_impl(mesh1, mesh2, weight);
}

template <typename G, typename C>
std::vector<double> correlation::calc_pair(const double w, G &grp, C &cell_list, int ncx, int ncy, int ncz,
                                           const std::string label)
{
  const int nc3 = ncx * ncy * ncz;
  int nthread = omp_get_num_threads();
  int ithread = omp_get_thread_num();
  uint64_t progress = 0;
  uint64_t progress_thread = nc3 / nthread;
  uint64_t progress_div = 1 + progress_thread / 200;

  std::vector<double> thr_pair(nr, 0.0);

#pragma omp for collapse(3) schedule(dynamic)
  for(int ix = 0; ix < ncx; ix++) {
    for(int iy = 0; iy < ncy; iy++) {
      for(int iz = 0; iz < ncz; iz++) {

        const int cell_id = iz + ncz * (iy + ncy * ix);
        const auto &clist = cell_list[cell_id];

#pragma omp unroll
        for(int jx = -ndiv_1d; jx <= ndiv_1d; jx++) {
          for(int jy = -ndiv_1d; jy <= ndiv_1d; jy++) {
            for(int jz = -ndiv_1d; jz <= ndiv_1d; jz++) {

              const int nix = ((ix + jx) + ncx) % ncx;
              const int niy = ((iy + jy) + ncy) % ncy;
              const int niz = ((iz + jz) + ncz) % ncz;

              const int ncell_id = niz + ncz * (niy + ncy * nix);
              const auto &nlist = cell_list[ncell_id];

              for(int ii : clist) {
                for(int jj : nlist) {

                  if(cell_id == ncell_id && ii == jj) continue;

                  if(symmetry) {
                    if(cell_id > ncell_id) continue;
                    if(cell_id == ncell_id && ii >= jj) continue;
                  }

                  double dx = grp[jj].xpos - grp[ii].xpos;
                  double dy = grp[jj].ypos - grp[ii].ypos;
                  double dz = grp[jj].zpos - grp[ii].zpos;
                  const int ir = get_ir_form_dr(dx, dy, dz);

                  if(ir >= 0 && ir < nr) {
                    thr_pair[ir] += w;
                  }
                }
              }
            }
          }
        } // dx, dy, dz

        if(ithread == 0) {
          progress++;
          if(progress % progress_div == 0) {
            std::cerr << "\r\033[2K " << label << " : " << (double)100.0 * progress / (double)progress_thread << " [%]";
          }
        }
      }
    }
  } // ix, iy, iz

  if(ithread == 0) std::cerr << std::endl;
  return thr_pair;
}

template <typename G, typename C>
std::vector<double> correlation::calc_pair(const double w, G &grp1, G &grp2, C &cell_list1, C &cell_list2, int ncx,
                                           int ncy, int ncz, const std::string label)
{
  /*
   allways symmetry is false
  */
  const int nc3 = ncx * ncy * ncz;
  int nthread = omp_get_num_threads();
  int ithread = omp_get_thread_num();
  uint64_t progress = 0;
  uint64_t progress_thread = nc3 / nthread;
  uint64_t progress_div = 1 + progress_thread / 200;

  std::vector<double> thr_pair(nr, 0.0);

#pragma omp for collapse(3) schedule(dynamic)
  for(int ix = 0; ix < ncx; ix++) {
    for(int iy = 0; iy < ncy; iy++) {
      for(int iz = 0; iz < ncz; iz++) {

        const int cell_id = iz + ncz * (iy + ncy * ix);
        const auto &clist = cell_list1[cell_id];

#pragma omp unroll
        for(int jx = -ndiv_1d; jx <= ndiv_1d; jx++) {
          for(int jy = -ndiv_1d; jy <= ndiv_1d; jy++) {
            for(int jz = -ndiv_1d; jz <= ndiv_1d; jz++) {

              const int nix = ((ix + jx) + ncx) % ncx;
              const int niy = ((iy + jy) + ncy) % ncy;
              const int niz = ((iz + jz) + ncz) % ncz;

              const int ncell_id = niz + ncz * (niy + ncy * nix);
              const auto &nlist = cell_list2[ncell_id];

              for(int ii : clist) {
                for(int jj : nlist) {
                  double dx = grp2[jj].xpos - grp1[ii].xpos;
                  double dy = grp2[jj].ypos - grp1[ii].ypos;
                  double dz = grp2[jj].zpos - grp1[ii].zpos;
                  const int ir = get_ir_form_dr(dx, dy, dz);

                  if(ir >= 0 && ir < nr) {
                    thr_pair[ir] += w;
                  }
                }
              }
            }
          }
        } // dx, dy, dz

        if(ithread == 0) {
          progress++;
          if(progress % progress_div == 0) {
            std::cerr << "\r\033[2K " << label << " : " << (double)100.0 * progress / (double)progress_thread << " [%]";
          }
        }
      }
    }
  } // ix, iy, iz

  if(ithread == 0) std::cerr << std::endl;
  return thr_pair;
}

template <typename G, typename C>
std::vector<double> correlation::calc_pair_smu(const double w, G &grp, C &cell_list, int ncx, int ncy, int ncz,
                                               const std::string label)
{
  const int nc3 = ncx * ncy * ncz;
  int nthread = omp_get_num_threads();
  int ithread = omp_get_thread_num();
  uint64_t progress = 0;
  uint64_t progress_thread = nc3 / nthread;
  uint64_t progress_div = 1 + progress_thread / 200;

  std::vector<double> thr_pair_smu(nr * nmu, 0.0); // [ir,imu]

#pragma omp for collapse(3) schedule(dynamic)
  for(int ix = 0; ix < ncx; ix++) {
    for(int iy = 0; iy < ncy; iy++) {
      for(int iz = 0; iz < ncz; iz++) {

        const int cell_id = iz + ncz * (iy + ncy * ix);
        const auto &clist = cell_list[cell_id];

#pragma omp unroll
        for(int jx = -ndiv_1d; jx <= ndiv_1d; jx++) {
          for(int jy = -ndiv_1d; jy <= ndiv_1d; jy++) {
            for(int jz = -ndiv_1d; jz <= ndiv_1d; jz++) {

              const int nix = ((ix + jx) + ncx) % ncx;
              const int niy = ((iy + jy) + ncy) % ncy;
              const int niz = ((iz + jz) + ncz) % ncz;

              const int ncell_id = niz + ncz * (niy + ncy * nix);
              const auto &nlist = cell_list[ncell_id];

              for(int ii : clist) {
                for(int jj : nlist) {

                  if(cell_id == ncell_id && ii == jj) continue;

                  if(symmetry) {
                    if(cell_id > ncell_id) continue;
                    if(cell_id == ncell_id && ii >= jj) continue;
                  }

                  double dx = grp[jj].xpos - grp[ii].xpos;
                  double dy = grp[jj].ypos - grp[ii].ypos;
                  double dz = grp[jj].zpos - grp[ii].zpos;

                  const int ir = get_ir_form_dr(dx, dy, dz);
                  const int imu = get_imu_form_dr(dx, dy, dz);

                  if(ir >= 0 && ir < nr) {
                    if(imu >= 0 && imu < nmu) {
                      auto idx = imu + nmu * ir;
                      thr_pair_smu[idx] += w;
                    } // imu
                  } // ir
                }
              }
            }
          }
        } // dx, dy, dz

        if(ithread == 0) {
          progress++;
          if(progress % progress_div == 0) {
            std::cerr << "\r\033[2K " << label << " (s,mu) : " << (double)100.0 * progress / (double)progress_thread
                      << " [%]";
          }
        }
      }
    }
  } // ix, iy, iz

  if(ithread == 0) std::cerr << std::endl;
  return thr_pair_smu;
}

template <typename G, typename C>
std::vector<double> correlation::calc_pair_smu(const double w, G &grp1, G &grp2, C &cell_list1, C &cell_list2, int ncx,
                                               int ncy, int ncz, const std::string label)
{
  /*
   allways symmetry is false
  */
  const int nc3 = ncx * ncy * ncz;
  int nthread = omp_get_num_threads();
  int ithread = omp_get_thread_num();
  uint64_t progress = 0;
  uint64_t progress_thread = nc3 / nthread;
  uint64_t progress_div = 1 + progress_thread / 200;

  std::vector<double> thr_pair_smu(nr * nmu, 0.0); // [ir,imu]

#pragma omp for collapse(3) schedule(dynamic)
  for(int ix = 0; ix < ncx; ix++) {
    for(int iy = 0; iy < ncy; iy++) {
      for(int iz = 0; iz < ncz; iz++) {

        const int cell_id = iz + ncz * (iy + ncy * ix);
        const auto &clist = cell_list1[cell_id];

#pragma omp unroll
        for(int jx = -ndiv_1d; jx <= ndiv_1d; jx++) {
          for(int jy = -ndiv_1d; jy <= ndiv_1d; jy++) {
            for(int jz = -ndiv_1d; jz <= ndiv_1d; jz++) {

              const int nix = ((ix + jx) + ncx) % ncx;
              const int niy = ((iy + jy) + ncy) % ncy;
              const int niz = ((iz + jz) + ncz) % ncz;

              const int ncell_id = niz + ncz * (niy + ncy * nix);
              const auto &nlist = cell_list2[ncell_id];

              for(int ii : clist) {
                for(int jj : nlist) {
                  double dx = grp2[jj].xpos - grp1[ii].xpos;
                  double dy = grp2[jj].ypos - grp1[ii].ypos;
                  double dz = grp2[jj].zpos - grp1[ii].zpos;

                  const int ir = get_ir_form_dr(dx, dy, dz);
                  const int imu = get_imu_form_dr(dx, dy, dz);

                  if(ir >= 0 && ir < nr) {
                    if(imu >= 0 && imu < nmu) {
                      auto idx = imu + nmu * ir;
                      thr_pair_smu[idx] += w;
                    } // imu
                  } // ir
                }
              }
            }
          }
        } // dx, dy, dz

        if(ithread == 0) {
          progress++;
          if(progress % progress_div == 0) {
            std::cerr << "\r\033[2K " << label << " (s,mu) : " << (double)100.0 * progress / (double)progress_thread
                      << " [%]";
          }
        }
      }
    }
  } // ix, iy, iz

  if(ithread == 0) std::cerr << std::endl;
  return thr_pair_smu;
}

template <typename G, typename C>
std::vector<double> correlation::calc_pair_spsp(const double w, G &grp, C &cell_list, int ncx, int ncy, int ncz,
                                                const std::string label)
{
  const int nc3 = ncx * ncy * ncz;
  int nthread = omp_get_num_threads();
  int ithread = omp_get_thread_num();
  uint64_t progress = 0;
  uint64_t progress_thread = nc3 / nthread;
  uint64_t progress_div = 1 + progress_thread / 200;

  std::vector<double> thr_pair_spsp(nsperp * nspara, 0.0); // [iperp,ipara]

#pragma omp for collapse(3) schedule(dynamic)
  for(int ix = 0; ix < ncx; ix++) {
    for(int iy = 0; iy < ncy; iy++) {
      for(int iz = 0; iz < ncz; iz++) {

        const int cell_id = iz + ncz * (iy + ncy * ix);
        const auto &clist = cell_list[cell_id];

#pragma omp unroll
        for(int jx = -ndiv_1d; jx <= ndiv_1d; jx++) {
          for(int jy = -ndiv_1d; jy <= ndiv_1d; jy++) {
            for(int jz = -ndiv_1d; jz <= ndiv_1d; jz++) {

              const int nix = ((ix + jx) + ncx) % ncx;
              const int niy = ((iy + jy) + ncy) % ncy;
              const int niz = ((iz + jz) + ncz) % ncz;

              const int ncell_id = niz + ncz * (niy + ncy * nix);
              const auto &nlist = cell_list[ncell_id];

              for(int ii : clist) {
                for(int jj : nlist) {

                  if(cell_id == ncell_id && ii == jj) continue;

                  if(symmetry) {
                    if(cell_id > ncell_id) continue;
                    if(cell_id == ncell_id && ii >= jj) continue;
                  }

                  double dx = grp[jj].xpos - grp[ii].xpos;
                  double dy = grp[jj].ypos - grp[ii].ypos;
                  double dz = grp[jj].zpos - grp[ii].zpos;

                  dx -= std::nearbyint(dx);
                  dy -= std::nearbyint(dy);
                  dz -= std::nearbyint(dz);

                  const double spara = dz;                           //  r * mu;
                  const double sperp = std::sqrt(dx * dx + dy * dy); //  sqrt(r^2 - spara^2);
                  const int iperp = std::floor((sperp - sperp_min) / dsperp);
                  const int ipara = std::floor((spara - spara_min) / dspara);

                  if(iperp >= 0 && iperp < nsperp) {
                    if(ipara >= 0 && ipara < nspara) {
                      auto idx = ipara + nspara * iperp;
                      thr_pair_spsp[idx] += w;
                    } // iperp
                  } // ipara
                }
              }
            }
          }
        } // dx, dy, dz

        if(ithread == 0) {
          progress++;
          if(progress % progress_div == 0) {
            std::cerr << "\r\033[2K " << label
                      << " (sperp,spara) : " << (double)100.0 * progress / (double)progress_thread << " [%]";
          }
        }
      }
    }
  } // ix, iy, iz

  if(ithread == 0) std::cerr << std::endl;
  return thr_pair_spsp;
}

template <typename G, typename C>
std::vector<double> correlation::calc_pair_spsp(const double w, G &grp1, G &grp2, C &cell_list1, C &cell_list2, int ncx,
                                                int ncy, int ncz, const std::string label)
{
  const int nc3 = ncx * ncy * ncz;
    int nthread = omp_get_num_threads();
    int ithread = omp_get_thread_num();
  uint64_t progress = 0;
    uint64_t progress_thread = nc3 / nthread;
  uint64_t progress_div = 1 + progress_thread / 200;

  std::vector<double> thr_pair_spsp(nsperp * nspara, 0.0); // [iperp,ipara]

#pragma omp for collapse(3) schedule(dynamic)
  for(int ix = 0; ix < ncx; ix++) {
    for(int iy = 0; iy < ncy; iy++) {
      for(int iz = 0; iz < ncz; iz++) {

        const int cell_id = iz + ncz * (iy + ncy * ix);
        const auto &clist = cell_list1[cell_id];

#pragma omp unroll
        for(int jx = -ndiv_1d; jx <= ndiv_1d; jx++) {
          for(int jy = -ndiv_1d; jy <= ndiv_1d; jy++) {
            for(int jz = -ndiv_1d; jz <= ndiv_1d; jz++) {

              const int nix = ((ix + jx) + ncx) % ncx;
              const int niy = ((iy + jy) + ncy) % ncy;
              const int niz = ((iz + jz) + ncz) % ncz;

              const int ncell_id = niz + ncz * (niy + ncy * nix);
              const auto &nlist = cell_list2[ncell_id];

              for(int ii : clist) {
                for(int jj : nlist) {

                  double dx = grp2[jj].xpos - grp1[ii].xpos;
                  double dy = grp2[jj].ypos - grp1[ii].ypos;
                  double dz = grp2[jj].zpos - grp1[ii].zpos;

                  dx -= std::nearbyint(dx);
                  dy -= std::nearbyint(dy);
                  dz -= std::nearbyint(dz);

                  const double spara = dz;                           //  r * mu;
                  const double sperp = std::sqrt(dx * dx + dy * dy); //  sqrt(r^2 - spara^2);
                  const int iperp = std::floor((sperp - sperp_min) / dsperp);
                  const int ipara = std::floor((spara - spara_min) / dspara);

                  if(iperp >= 0 && iperp < nsperp) {
                    if(ipara >= 0 && ipara < nspara) {
                      auto idx = ipara + nspara * iperp;
                      thr_pair_spsp[idx] += w;
                    } // iperp
                  } // ipara
  }
      }
            }
          }
        } // dx, dy, dz

        if(ithread == 0) {
          progress++;
          if(progress % progress_div == 0) {
            std::cerr << "\r\033[2K " << label
                      << " (sperp,spara) : " << (double)100.0 * progress / (double)progress_thread << " [%]";
    }
  }
      }
    }
  } // ix, iy, iz

  if(ithread == 0) std::cerr << std::endl;
  return thr_pair_spsp;
}

template <typename T>
void correlation::calc_xi_impl(T &grp)
{
  bool use_random = (estimator != "ideal");
  uint64_t ngrp = grp.size();
  uint64_t nrand = 0;

  dd_pair.assign(nr, 0.0);
  rr_pair.assign(nr, 0.0);
  if(estimator == "LS") dr_pair.assign(nr, 0.0);

  xi.assign(nr, 0.0);

  const int ncx = ndiv_1d * std::max(static_cast<int>(1.0 / rmax), 1);
  const int ncy = ndiv_1d * std::max(static_cast<int>(1.0 / rmax), 1);
  const int ncz = ndiv_1d * std::max(static_cast<int>(1.0 / rmax), 1);
  const int nc3 = ncx * ncy * ncz;

  std::vector<std::vector<int>> cell_list(nc3), cell_list_rand;
  std::vector<group> rand;

  for(uint64_t i = 0; i < ngrp; i++) {
    const int ix = get_cell_index(grp[i].xpos, ncx);
    const int iy = get_cell_index(grp[i].ypos, ncy);
    const int iz = get_cell_index(grp[i].zpos, ncz);
    const int cell_id = iz + ncz * (iy + ncy * ix);
    cell_list[cell_id].push_back(i);
  }

  if(use_random) {
    nrand = ngrp * nrand_factor;
    cell_list_rand.resize(nc3);
    groupcatalog g;
    rand = g.set_random_group(nrand);

    for(uint64_t i = 0; i < nrand; i++) {
      const int ix = get_cell_index(rand[i].xpos, ncx);
      const int iy = get_cell_index(rand[i].ypos, ncy);
      const int iz = get_cell_index(rand[i].zpos, ncz);
      const int cell_id = iz + ncz * (iy + ncy * ix);
      cell_list_rand[cell_id].push_back(i);
    }
  }

#pragma omp parallel
  {
    int nthread = omp_get_num_threads();
    int ithread = omp_get_thread_num();
    if(ithread == 0) std::cerr << "# nc^3 = " << nc3 << " divided into " << nthread << " threads." << std::endl;

    auto thr_dd_pair = calc_pair(1.0, grp, cell_list, ncx, ncy, ncz, "DD");
    decltype(thr_dd_pair) thr_rr_pair, thr_dr_pair;

    if(use_random) {
      thr_rr_pair = calc_pair(1.0, rand, cell_list_rand, ncx, ncy, ncz, "RR");
      if(estimator == "LS") thr_dr_pair = calc_pair(0.5, grp, rand, cell_list, cell_list_rand, ncx, ncy, ncz, "DR");
    }

#pragma omp critical
    {
      for(size_t i = 0; i < nr; i++) dd_pair[i] += thr_dd_pair[i];
      if(use_random) {
        for(size_t i = 0; i < nr; i++) rr_pair[i] += thr_rr_pair[i];
        if(estimator == "LS")
          for(size_t i = 0; i < nr; i++) dr_pair[i] += thr_dr_pair[i];
      }
    } // omp critical
  } // omp parallel

  if(estimator == "ideal") {
    auto npair = static_cast<double>(ngrp) * (ngrp - 1);
    if(symmetry) npair *= 0.5;
    calc_ideal_pair(npair); // calc ideal rr_pair
  }

  calc_xi_from_pair(ngrp); // calc xi
    }

template <typename T>
void correlation::calc_xi_smu_impl(T &grp)
{
  bool use_random = (estimator != "ideal");

  uint64_t nn = nr * nmu;
  uint64_t ngrp = grp.size();
  uint64_t nrand = 0;

  std::vector<std::vector<int>> cell_list, cell_list_rand;
  std::vector<group> rand;

  dd_pair.assign(nn, 0.0);
  rr_pair.assign(nn, 0.0);
  if(estimator == "LS") dr_pair.assign(nn, 0.0);
  xi.assign(nn, 0.0);

  const int ncx = ndiv_1d * std::max(static_cast<int>(1.0 / rmax), 1);
  const int ncy = ndiv_1d * std::max(static_cast<int>(1.0 / rmax), 1);
  const int ncz = ndiv_1d * std::max(static_cast<int>(1.0 / rmax), 1);
  const int nc3 = ncx * ncy * ncz;

  cell_list.resize(nc3);

  for(uint64_t i = 0; i < ngrp; i++) {
    const int ix = get_cell_index(grp[i].xpos, ncx);
    const int iy = get_cell_index(grp[i].ypos, ncy);
    const int iz = get_cell_index(grp[i].zpos, ncz);
    const int cell_id = iz + ncz * (iy + ncy * ix);
    cell_list[cell_id].push_back(i);
  }

  if(use_random) {
    nrand = ngrp * nrand_factor;
    cell_list_rand.resize(nc3);
    groupcatalog g;
    rand = g.set_random_group(nrand);

  for(uint64_t i = 0; i < nrand; i++) {
    const int ix = get_cell_index(rand[i].xpos, ncx);
    const int iy = get_cell_index(rand[i].ypos, ncy);
    const int iz = get_cell_index(rand[i].zpos, ncz);
    const int cell_id = iz + ncz * (iy + ncy * ix);
    cell_list_rand[cell_id].push_back(i);
  }
  }

#pragma omp parallel
  {
    int nthread = omp_get_num_threads();
    int ithread = omp_get_thread_num();
    if(ithread == 0) std::cerr << "# nc^3 = " << nc3 << " divided into " << nthread << " threads." << std::endl;

    auto thr_dd_pair_smu = calc_pair_smu(1.0, grp, cell_list, ncx, ncy, ncz, "DD");
    decltype(thr_dd_pair_smu) thr_rr_pair_smu, thr_dr_pair_smu;

    if(use_random) {
      thr_rr_pair_smu = calc_pair_smu(1.0, rand, cell_list_rand, ncx, ncy, ncz, "RR");
      if(estimator == "LS")
        thr_dr_pair_smu = calc_pair_smu(0.5, grp, rand, cell_list, cell_list_rand, ncx, ncy, ncz, "DR");
    }

#pragma omp critical
    {
      for(size_t i = 0; i < nn; i++) dd_pair[i] += thr_dd_pair_smu[i];
      if(use_random) {
        for(size_t i = 0; i < nn; i++) rr_pair[i] += thr_rr_pair_smu[i];
        if(estimator == "LS")
          for(size_t i = 0; i < nn; i++) dr_pair[i] += thr_dr_pair_smu[i];
      }
    } // omp critical
  } // omp parallel

  if(estimator == "ideal") {
    auto npair = ngrp * (ngrp - 1);
    if(symmetry) npair *= 0.5;
    calc_ideal_smu_pair(npair); // calc ideal rr_pair
    }

  calc_xi_from_pair(ngrp); // calc xi
}

template <typename T>
void correlation::calc_xi_spsp_impl(T &grp)
{
  bool use_random = (estimator != "ideal");

  uint64_t nn = nsperp * nspara;
  uint64_t ngrp = grp.size();
  uint64_t nrand = 0;

  std::vector<std::vector<int>> cell_list, cell_list_rand;
  std::vector<group> rand;

  dd_pair.assign(nn, 0.0);
  rr_pair.assign(nn, 0.0);
  if(estimator == "LS") dr_pair.assign(nn, 0.0);
  xi.assign(nn, 0.0);

  /* cell list setup */
  const int ncx = ndiv_1d * std::max(static_cast<int>(1.0 / rmax), 1);
  const int ncy = ndiv_1d * std::max(static_cast<int>(1.0 / rmax), 1);
  const int ncz = ndiv_1d * std::max(static_cast<int>(1.0 / rmax), 1);
  const int nc3 = ncx * ncy * ncz;

  cell_list.resize(nc3);

  for(uint64_t i = 0; i < ngrp; i++) {
    const int ix = get_cell_index(grp[i].xpos, ncx);
    const int iy = get_cell_index(grp[i].ypos, ncy);
    const int iz = get_cell_index(grp[i].zpos, ncz);
    const int cell_id = iz + ncz * (iy + ncy * ix);
    cell_list[cell_id].push_back(i);
  }

  if(use_random) {
    nrand = ngrp * nrand_factor;
    cell_list_rand.resize(nc3);
    groupcatalog g;
    rand = g.set_random_group(nrand);

  for(uint64_t i = 0; i < nrand; i++) {
    const int ix = get_cell_index(rand[i].xpos, ncx);
    const int iy = get_cell_index(rand[i].ypos, ncy);
    const int iz = get_cell_index(rand[i].zpos, ncz);
    const int cell_id = iz + ncz * (iy + ncy * ix);
    cell_list_rand[cell_id].push_back(i);
  }
  }

#pragma omp parallel
  {
    int nthread = omp_get_num_threads();
    int ithread = omp_get_thread_num();
    if(ithread == 0)
      std::cerr << "# nc^3, ngrp_thread = " << nc3 << ", " << (nc3 / nthread) << " in " << nthread << " threads."
                << std::endl;

    auto thr_dd_pair_spsp = calc_pair_spsp(1.0, grp, cell_list, ncx, ncy, ncz, "DD");
    decltype(thr_dd_pair_spsp) thr_rr_pair_spsp, thr_dr_pair_spsp;

    if(use_random) {
      thr_rr_pair_spsp = calc_pair_spsp(1.0, rand, cell_list_rand, ncx, ncy, ncz, "RR");
      if(estimator == "LS")
        thr_dr_pair_spsp = calc_pair_spsp(0.5, grp, rand, cell_list, cell_list_rand, ncx, ncy, ncz, "DR");
    }

#pragma omp critical
    {
      for(size_t i = 0; i < nn; i++) dd_pair[i] += thr_dd_pair_spsp[i];
      if(use_random) {
        for(size_t i = 0; i < nn; i++) rr_pair[i] += thr_rr_pair_spsp[i];
        if(estimator == "LS")
          for(size_t i = 0; i < nn; i++) dr_pair[i] += thr_dr_pair_spsp[i];
      }
    } // omp critical
  } // omp parallel

  if(estimator == "ideal") {
    auto npair = static_cast<double>(ngrp) * (ngrp - 1);
    if(symmetry) npair *= 0.5;
    calc_ideal_spsp_pair(npair); // calc ideal rr_pair
    }

  calc_xi_from_pair(ngrp); // calc xi
  }

template <typename T>
void correlation::calc_xi_impl(T &grp1, T &grp2)
{
  uint64_t ngrp1 = grp1.size();
  uint64_t ngrp2 = grp2.size();

  dd_pair.assign(nr, 0.0);
  xi.assign(nr, 0.0);

  /* Here only the global box size */
  const int ncx = ndiv_1d * std::max(static_cast<int>(1.0 / rmax), 1);
  const int ncy = ndiv_1d * std::max(static_cast<int>(1.0 / rmax), 1);
  const int ncz = ndiv_1d * std::max(static_cast<int>(1.0 / rmax), 1);
  const int nc3 = ncx * ncy * ncz;

  std::vector<std::vector<int>> cell_list1(nc3);
  std::vector<std::vector<int>> cell_list2(nc3);

  for(uint64_t i = 0; i < ngrp1; i++) {
    const int ix = get_cell_index(grp1[i].xpos, ncx);
    const int iy = get_cell_index(grp1[i].ypos, ncy);
    const int iz = get_cell_index(grp1[i].zpos, ncz);
    const int cell_id = iz + ncz * (iy + ncy * ix);
    cell_list1[cell_id].push_back(i);
  }

  for(uint64_t i = 0; i < ngrp2; i++) {
    const int ix = get_cell_index(grp2[i].xpos, ncx);
    const int iy = get_cell_index(grp2[i].ypos, ncy);
    const int iz = get_cell_index(grp2[i].zpos, ncz);
    const int cell_id = iz + ncz * (iy + ncy * ix);
    cell_list2[cell_id].push_back(i);
  }

#pragma omp parallel
  {

    int nthread = omp_get_num_threads();
    int ithread = omp_get_thread_num();
    uint64_t progress = 0;
    uint64_t progress_thread = nc3 / nthread;
    uint64_t progress_div = 1 + progress_thread / 200;

    if(ithread == 0)
      std::cerr << "# nc^3, ngrp_thread = " << nc3 << ", " << progress_thread << " in " << nthread << " threads."
                << std::endl;

    auto thr_dd_pair = calc_pair(1.0, grp1, grp2, cell_list1, cell_list2, ncx, ncy, ncz, "D1D2");

#pragma omp critical
    {
      for(int ir = 0; ir < nr; ir++) {
        dd_pair[ir] += thr_dd_pair[ir];
      }
    } // omp critical
  } // omp parallel

  double V_box = 1.0;
  double npair = (double)ngrp1 * (double)ngrp2;
  double dr = (log_scale) ? (log(rmax / rmin) / nr) : ((rmax - rmin) / nr);

  for(int ir = 0; ir < nr; ir++) {
    double r_low = (log_scale) ? (rmin * exp(ir * dr)) : (rmin + ir * dr);
    double r_high = (log_scale) ? (rmin * exp((ir + 1) * dr)) : (rmin + (ir + 1) * dr);
    double shell_volume = (4.0 / 3.0) * M_PI * (r_high * r_high * r_high - r_low * r_low * r_low);
    double norm = npair * shell_volume / V_box;
    xi[ir] = dd_pair[ir] / norm - 1.0;
  }
}

template <typename T>
void correlation::calc_xi_LS_impl(T &grp)
{
  /* Landy SD, Szalay AS 1993, Apj */
  /* Advice by chatgpt */
  uint64_t ngrp = grp.size();
  uint64_t nrand = ngrp * nrand_factor;

  dd_pair.assign(nr, 0.0);
  dr_pair.assign(nr, 0.0);
  rr_pair.assign(nr, 0.0);
  xi.assign(nr, 0.0);

  groupcatalog g;
  auto rand = g.set_random_group(nrand);

  /* Here only the global box size */
  const int ncx = ndiv_1d * std::max(static_cast<int>(1.0 / rmax), 1);
  const int ncy = ndiv_1d * std::max(static_cast<int>(1.0 / rmax), 1);
  const int ncz = ndiv_1d * std::max(static_cast<int>(1.0 / rmax), 1);
  const int nc3 = ncx * ncy * ncz;

  std::vector<std::vector<int>> cell_list(nc3);
  std::vector<std::vector<int>> cell_list_rand(nc3);

  for(uint64_t i = 0; i < ngrp; i++) {
    const int ix = get_cell_index(grp[i].xpos, ncx);
    const int iy = get_cell_index(grp[i].ypos, ncy);
    const int iz = get_cell_index(grp[i].zpos, ncz);
    const int cell_id = iz + ncz * (iy + ncy * ix);
    cell_list[cell_id].push_back(i);
  }

  for(uint64_t i = 0; i < nrand; i++) {
    const int ix = get_cell_index(rand[i].xpos, ncx);
    const int iy = get_cell_index(rand[i].ypos, ncy);
    const int iz = get_cell_index(rand[i].zpos, ncz);
    const int cell_id = iz + ncz * (iy + ncy * ix);
    cell_list_rand[cell_id].push_back(i);
  }

#pragma omp parallel
  {

    int nthread = omp_get_num_threads();
    int ithread = omp_get_thread_num();
    uint64_t progress = 0;
    uint64_t progress_thread = nc3 / nthread;
    uint64_t progress_div = 1 + progress_thread / 200;

    if(ithread == 0)
      std::cerr << "# nc^3, ngrp_thread = " << nc3 << ", " << progress_thread << " in " << nthread << " threads."
                << std::endl;

    auto thr_rr_pair = calc_pair(1.0, rand, cell_list_rand, ncx, ncy, ncz, "RR");
    auto thr_dd_pair = calc_pair(1.0, grp, cell_list, ncx, ncy, ncz, "DD");
    auto thr_dr_pair = calc_pair(0.5, grp, rand, cell_list, cell_list_rand, ncx, ncy, ncz, "DR");

#pragma omp critical
    {
      for(int ir = 0; ir < nr; ir++) {
        rr_pair[ir] += thr_rr_pair[ir];
        dd_pair[ir] += thr_dd_pair[ir];
        dr_pair[ir] += thr_dr_pair[ir];
      }
    } // omp critical
  } // omp parallel

  double f = (double)nrand / (double)ngrp;
  double f2 = f * f;
  for(int ir = 0; ir < nr; ir++) {
    if(rr_pair[ir] != 0.0) {
      xi[ir] = (dd_pair[ir] * f2 - 2.0 * dr_pair[ir] * f + rr_pair[ir]) / rr_pair[ir];
    }
  }
}

template <typename T>
void correlation::calc_xi_LS_impl(T &grp1, T &grp2)
{
  /* Landy SD, Szalay AS 1993, Apj */
  /* Advice by chatgpt */
  uint64_t ngrp1 = grp1.size();
  uint64_t ngrp2 = grp2.size();

  uint64_t nrand1 = ngrp1 * nrand_factor;
  uint64_t nrand2 = ngrp2 * nrand_factor;

  dd_pair.assign(nr, 0.0);
  dr_pair.assign(nr, 0.0);
  dr2_pair.assign(nr, 0.0);
  rr_pair.assign(nr, 0.0);
  xi.assign(nr, 0.0);

  groupcatalog g;
  auto rand1 = g.set_random_group(nrand1, 1);
  auto rand2 = g.set_random_group(nrand2, 2);

  /* Here only the global box size */
  const int ncx = ndiv_1d * std::max(static_cast<int>(1.0 / rmax), 1);
  const int ncy = ndiv_1d * std::max(static_cast<int>(1.0 / rmax), 1);
  const int ncz = ndiv_1d * std::max(static_cast<int>(1.0 / rmax), 1);
  const int nc3 = ncx * ncy * ncz;

  std::vector<std::vector<int>> cell_list1(nc3);
  std::vector<std::vector<int>> cell_list2(nc3);
  std::vector<std::vector<int>> cell_list_rand1(nc3);
  std::vector<std::vector<int>> cell_list_rand2(nc3);

  for(uint64_t i = 0; i < ngrp1; i++) {
    const int ix = get_cell_index(grp1[i].xpos, ncx);
    const int iy = get_cell_index(grp1[i].ypos, ncy);
    const int iz = get_cell_index(grp1[i].zpos, ncz);
    const int cell_id = iz + ncz * (iy + ncy * ix);
    cell_list1[cell_id].push_back(i);
  }

  for(uint64_t i = 0; i < ngrp2; i++) {
    const int ix = get_cell_index(grp2[i].xpos, ncx);
    const int iy = get_cell_index(grp2[i].ypos, ncy);
    const int iz = get_cell_index(grp2[i].zpos, ncz);
    const int cell_id = iz + ncz * (iy + ncy * ix);
    cell_list2[cell_id].push_back(i);
  }

  for(uint64_t i = 0; i < nrand1; i++) {
    const int ix = get_cell_index(rand1[i].xpos, ncx);
    const int iy = get_cell_index(rand1[i].ypos, ncy);
    const int iz = get_cell_index(rand1[i].zpos, ncz);
    const int cell_id = iz + ncz * (iy + ncy * ix);
    cell_list_rand1[cell_id].push_back(i);
  }

  for(uint64_t i = 0; i < nrand2; i++) {
    const int ix = get_cell_index(rand2[i].xpos, ncx);
    const int iy = get_cell_index(rand2[i].ypos, ncy);
    const int iz = get_cell_index(rand2[i].zpos, ncz);
    const int cell_id = iz + ncz * (iy + ncy * ix);
    cell_list_rand2[cell_id].push_back(i);
  }

#pragma omp parallel
  {

    int nthread = omp_get_num_threads();
    int ithread = omp_get_thread_num();
    uint64_t progress = 0;
    uint64_t progress_thread = nc3 / nthread;
    uint64_t progress_div = 1 + progress_thread / 200;

    if(ithread == 0)
      std::cerr << "# nc^3, ngrp_thread = " << nc3 << ", " << progress_thread << " in " << nthread << " threads."
                << std::endl;

    auto thr_rr_pair = calc_pair(1.0, rand1, rand2, cell_list_rand1, cell_list_rand2, ncx, ncy, ncz, "R1R2");
    auto thr_dr_pair = calc_pair(1.0, grp1, rand2, cell_list1, cell_list_rand2, ncx, ncy, ncz, "D1R2");
    auto thr_dr2_pair = calc_pair(1.0, grp2, rand1, cell_list2, cell_list_rand1, ncx, ncy, ncz, "D2R1");
    auto thr_dd_pair = calc_pair(1.0, grp1, grp2, cell_list1, cell_list2, ncx, ncy, ncz, "D1D2");

#pragma omp critical
    {
      for(int ir = 0; ir < nr; ir++) {
        rr_pair[ir] += thr_rr_pair[ir];
        dd_pair[ir] += thr_dd_pair[ir];
        dr_pair[ir] += thr_dr_pair[ir];
        dr2_pair[ir] += thr_dr2_pair[ir];
      }
    } // omp critical
  } // omp parallel

  double f1 = (double)nrand1 / (double)ngrp1;
  double f2 = (double)nrand2 / (double)ngrp2;
  double f12 = f1 * f2;

  for(int ir = 0; ir < nr; ir++) {
    if(rr_pair[ir] != 0.0) {
      xi[ir] = (dd_pair[ir] * f12 - dr_pair[ir] * f1 - dr2_pair[ir] * f2 + rr_pair[ir]) / rr_pair[ir];
    }
  }
}

template <typename T>
void correlation::calc_xi_jk_impl(T &grp)
{
  uint64_t ngrp = grp.size();
  jk_dd = ngrp / jk_block;
  nblock = (int)ceil(ngrp / jk_dd);

  xi_ave.assign(nr, 0.0);
  xi_sd.assign(nr, 0.0);
  xi_se.assign(nr, 0.0);

  xi_jk.resize(nblock, std::vector<double>(nr));
  dd_pair_jk.resize(nblock, std::vector<double>(nr));

#pragma omp parallel for collapse(2)
  for(int iblock = 0; iblock < nblock; iblock++) {
    for(int ir = 0; ir < nr; ir++) {
      dd_pair_jk[iblock][ir] = 0.0;
      xi_jk[iblock][ir] = 0.0;
    }
  }

  if(jk_type == 0) {
    std::cerr << "blocked jackknife by spaced sampling" << std::endl;
    sort_jk_block(grp);
    set_block_edge_id(grp);
  } else {
    std::cerr << "blocked jackknife by shuffled sampling" << std::endl;
    shuffle_data(grp);
    set_block_edge_id_shuffle(ngrp);
  }

  resample_jk(grp);
  calc_jk_xi_average();
  calc_jk_xi_error();
}

template <typename T>
void correlation::calc_xi_jk_impl(T &grp1, T &grp2)
{
  uint64_t ngrp1 = grp1.size();
  uint64_t ngrp2 = grp2.size();

  /* based on ngrp1 */
  jk_dd = ngrp1 / jk_block;
  nblock = (int)ceil(ngrp1 / jk_dd);

  xi_ave.assign(nr, 0.0);
  xi_sd.assign(nr, 0.0);
  xi_se.assign(nr, 0.0);

  xi_jk.resize(nblock, std::vector<double>(nr));
  dd_pair_jk.resize(nblock, std::vector<double>(nr));

#pragma omp parallel for collapse(2)
  for(int iblock = 0; iblock < nblock; iblock++) {
    for(int ir = 0; ir < nr; ir++) {
      dd_pair_jk[iblock][ir] = 0.0;
      xi_jk[iblock][ir] = 0.0;
    }
  }

  if(jk_type == 0) {
    std::cerr << "blocked jackknife by spaced sampling" << std::endl;
    sort_jk_block(grp1);
    sort_jk_block(grp2);
    set_block_edge_id(grp1, grp2);
  } else {
    std::cerr << "blocked jackknife by shuffled sampling" << std::endl;
    shuffle_data(grp1);
    shuffle_data(grp2);
    set_block_edge_id_shuffle(ngrp1, ngrp2);
  }

  resample_jk(grp1, grp2);
  calc_jk_xi_average();
  calc_jk_xi_error();
}

template <typename T>
void correlation::calc_xi_jk_LS_impl(T &grp)
{
  uint64_t ngrp = grp.size();
  uint64_t nrand = ngrp * nrand_factor;

  jk_dd = ngrp / jk_block; // delete-d
  nblock = (int)ceil(ngrp / jk_dd);

  xi_ave.assign(nr, 0.0);
  xi_sd.assign(nr, 0.0);
  xi_se.assign(nr, 0.0);

  xi_jk.resize(nblock, std::vector<double>(nr));
  dd_pair_jk.resize(nblock, std::vector<double>(nr));
  dr_pair_jk.resize(nblock, std::vector<double>(nr));
  rr_pair_jk.resize(nblock, std::vector<double>(nr));

#pragma omp parallel for collapse(2)
  for(int iblock = 0; iblock < nblock; iblock++) {
    for(int ir = 0; ir < nr; ir++) {
      dd_pair_jk[iblock][ir] = 0.0;
      rr_pair_jk[iblock][ir] = 0.0;
      dr_pair_jk[iblock][ir] = 0.0;
      xi_jk[iblock][ir] = 0.0;
    }
  }

  groupcatalog g;
  auto rand = g.set_random_group(nrand);

  if(jk_type == 0) {
    std::cerr << "blocked jackknife by spaced sampling" << std::endl;
    sort_jk_block(grp);
    set_block_edge_id(grp);
    sort_jk_rand_block(rand);
    set_rand_block_edge_id(rand);
  } else {
    std::cerr << "blocked jackknife by shuffled sampling" << std::endl;
    shuffle_data(grp);
    set_block_edge_id_shuffle(ngrp);
    set_rand_block_edge_id_shuffle(nrand);
  }

  resample_jk_LS(grp, rand);
  calc_jk_xi_average();
  calc_jk_xi_error();
}

template <typename T>
void correlation::calc_xi_jk_LS_impl(T &grp1, T &grp2)
{
  uint64_t ngrp1 = grp1.size();
  uint64_t ngrp2 = grp2.size();
  uint64_t nrand1 = ngrp1 * nrand_factor;
  uint64_t nrand2 = ngrp2 * nrand_factor;

  /* based on ngrp1 */
  jk_dd = ngrp1 / jk_block;
  nblock = (int)ceil(ngrp1 / jk_dd);

  xi_ave.assign(nr, 0.0);
  xi_sd.assign(nr, 0.0);
  xi_se.assign(nr, 0.0);

  xi_jk.resize(nblock, std::vector<double>(nr));
  dd_pair_jk.resize(nblock, std::vector<double>(nr));
  dr_pair_jk.resize(nblock, std::vector<double>(nr));
  dr2_pair_jk.resize(nblock, std::vector<double>(nr));
  rr_pair_jk.resize(nblock, std::vector<double>(nr));

#pragma omp parallel for collapse(2)
  for(int iblock = 0; iblock < nblock; iblock++) {
    for(int ir = 0; ir < nr; ir++) {
      dd_pair_jk[iblock][ir] = 0.0;
      rr_pair_jk[iblock][ir] = 0.0;
      dr_pair_jk[iblock][ir] = 0.0;
      dr2_pair_jk[iblock][ir] = 0.0;
      xi_jk[iblock][ir] = 0.0;
    }
  }

  groupcatalog g;
  auto rand1 = g.set_random_group(nrand1, 3);
  auto rand2 = g.set_random_group(nrand2, 4);

  if(jk_type == 0) {
    std::cerr << "blocked jackknife by spaced sampling" << std::endl;
    sort_jk_block(grp1);
    sort_jk_block(grp2);
    sort_jk_rand_block(rand1);
    sort_jk_rand_block(rand2);
    set_block_edge_id(grp1, grp2);
    set_rand_block_edge_id(rand1, rand2);
  } else {
    std::cerr << "blocked jackknife by shuffled sampling" << std::endl;
    shuffle_data(grp1);
    shuffle_data(grp2);
    set_block_edge_id_shuffle(ngrp1, ngrp2);
    set_rand_block_edge_id_shuffle(nrand1, nrand2);
  }

  resample_jk_LS(grp1, grp2, rand1, rand2);
  calc_jk_xi_average();
  calc_jk_xi_error();
}

void correlation::calc_jk_xi_average()
{
#pragma omp parallel for
  for(int ir = 0; ir < nr; ir++) xi_ave[ir] = 0.0;

  for(int iblock = 0; iblock < nblock; iblock++) {
    for(int ir = 0; ir < nr; ir++) xi_ave[ir] += xi_jk[iblock][ir];
  }

#pragma omp parallel for
  for(int ir = 0; ir < nr; ir++) xi_ave[ir] /= (double)nblock;

  std::cerr << "# done " << __func__ << std::endl;
}

void correlation::calc_jk_xi_error()
{
  std::vector<double> variance(nr);

#pragma omp parallel for
  for(int ir = 0; ir < nr; ir++) {
    variance[ir] = 0.0;
    xi_sd[ir] = 0.0;
    xi_se[ir] = 0.0;
  }

  for(int iblock = 0; iblock < nblock; iblock++) {
    for(int ir = 0; ir < nr; ir++) {
      variance[ir] += (xi_jk[iblock][ir] - xi_ave[ir]) * (xi_jk[iblock][ir] - xi_ave[ir]);
    }
  }

#pragma omp parallel for
  for(int ir = 0; ir < nr; ir++) {
    variance[ir] *= (double)(nblock - 1.0) / (double)(nblock);
    xi_sd[ir] = sqrt(variance[ir]);
    xi_se[ir] = xi_sd[ir] / sqrt((double)nblock);
  }

  std::cerr << "# done " << __func__ << std::endl;
}

template <typename T>
void correlation::resample_jk(T &grp)
{
  uint64_t ngrp = grp.size();

  /* Here only the global box size */
  const int ncx = ndiv_1d * std::max(static_cast<int>(1.0 / rmax), 1);
  const int ncy = ndiv_1d * std::max(static_cast<int>(1.0 / rmax), 1);
  const int ncz = ndiv_1d * std::max(static_cast<int>(1.0 / rmax), 1);
  const int nc3 = ncx * ncy * ncz;

  for(int iblock = 0; iblock < nblock; iblock++) {
    int64_t delete_block_start = block_start[iblock];
    int64_t delete_block_end = block_end[iblock];

    std::cerr << "# iblock " << iblock << " " << delete_block_start << " " << delete_block_end << " length "
              << delete_block_end - delete_block_start << std::endl;

    std::vector<std::vector<int>> cell_list(nc3);

    for(int64_t i = 0; i < ngrp; i++) {
      if((delete_block_start <= i) && (i < delete_block_end)) continue;
      const int ix = get_cell_index(grp[i].xpos, ncx);
      const int iy = get_cell_index(grp[i].ypos, ncy);
      const int iz = get_cell_index(grp[i].zpos, ncz);
      const int cell_id = iz + ncz * (iy + ncy * ix);
      cell_list[cell_id].push_back(i);
    }

#pragma omp parallel
    {
      int nthread = omp_get_num_threads();
      int ithread = omp_get_thread_num();
      uint64_t progress = 0;
      uint64_t progress_thread = nc3 / nthread;
      uint64_t progress_div = 1 + progress_thread / 200;

      if(ithread == 0)
        std::cerr << "# nc^3, ngrp_thread = " << nc3 << ", " << progress_thread << " in " << nthread << " threads."
                  << std::endl;

      auto thr_dd_pair = calc_pair(1.0, grp, cell_list, ncx, ncy, ncz, "DD");

#pragma omp critical
      {
        for(int ir = 0; ir < nr; ir++) {
          dd_pair_jk[iblock][ir] += thr_dd_pair[ir];
        }
      }
    } // end parallel
  } // end nblock loop

  double V_box = 1.0;

  for(int iblock = 0; iblock < nblock; iblock++) {
    int64_t length = block_end[iblock] - block_start[iblock];
    double npair = (double)(ngrp - length) * ((ngrp - length) - 1) / 2.0;
    double dr = (log_scale) ? (log(rmax / rmin) / nr) : ((rmax - rmin) / nr);

    for(int ir = 0; ir < nr; ir++) {
      double r_low = (log_scale) ? (rmin * exp(ir * dr)) : (rmin + ir * dr);
      double r_high = (log_scale) ? (rmin * exp((ir + 1) * dr)) : (rmin + (ir + 1) * dr);
      double shell_volume = (4.0 / 3.0) * M_PI * (r_high * r_high * r_high - r_low * r_low * r_low);
      double norm = npair * shell_volume / V_box;
      xi_jk[iblock][ir] = dd_pair_jk[iblock][ir] / norm - 1.0;
    }
  }
}

template <typename T>
void correlation::resample_jk(T &grp1, T &grp2)
{
  uint64_t ngrp1 = grp1.size();
  uint64_t ngrp2 = grp2.size();

  /* Here only the global box size */
  const int ncx = ndiv_1d * std::max(static_cast<int>(1.0 / rmax), 1);
  const int ncy = ndiv_1d * std::max(static_cast<int>(1.0 / rmax), 1);
  const int ncz = ndiv_1d * std::max(static_cast<int>(1.0 / rmax), 1);
  const int nc3 = ncx * ncy * ncz;

  for(int iblock = 0; iblock < nblock; iblock++) {
    int64_t delete_block_start1 = block_start[iblock];
    int64_t delete_block_end1 = block_end[iblock];

    int64_t delete_block_start2 = block_start2[iblock];
    int64_t delete_block_end2 = block_end2[iblock];

    std::cerr << "# iblock " << iblock << " " << delete_block_start1 << " " << delete_block_end1 << " length "
              << delete_block_end1 - delete_block_start1 << std::endl;

    std::vector<std::vector<int>> cell_list1(nc3);
    std::vector<std::vector<int>> cell_list2(nc3);

    for(int64_t i = 0; i < ngrp1; i++) {
      if((delete_block_start1 <= i) && (i < delete_block_end1)) continue;
      const int ix = get_cell_index(grp1[i].xpos, ncx);
      const int iy = get_cell_index(grp1[i].ypos, ncy);
      const int iz = get_cell_index(grp1[i].zpos, ncz);
      const int cell_id = iz + ncz * (iy + ncy * ix);
      cell_list1[cell_id].push_back(i);
    }

    for(int64_t i = 0; i < ngrp2; i++) {
      if((delete_block_start2 <= i) && (i < delete_block_end2)) continue;
      const int ix = get_cell_index(grp2[i].xpos, ncx);
      const int iy = get_cell_index(grp2[i].ypos, ncy);
      const int iz = get_cell_index(grp2[i].zpos, ncz);
      const int cell_id = iz + ncz * (iy + ncy * ix);
      cell_list2[cell_id].push_back(i);
    }

#pragma omp parallel
    {
      int nthread = omp_get_num_threads();
      int ithread = omp_get_thread_num();
      uint64_t progress = 0;
      uint64_t progress_thread = nc3 / nthread;
      uint64_t progress_div = 1 + progress_thread / 200;

      if(ithread == 0)
        std::cerr << "# nc^3, ngrp_thread = " << nc3 << ", " << progress_thread << " in " << nthread << " threads."
                  << std::endl;

      auto thr_dd_pair = calc_pair(1.0, grp1, grp2, cell_list1, cell_list2, ncx, ncy, ncz, "D1D2");

#pragma omp critical
      {
        for(int ir = 0; ir < nr; ir++) {
          dd_pair_jk[iblock][ir] += thr_dd_pair[ir];
        }
      }
    } // end parallel
  } // end nblock loop

  double V_box = 1.0;

  for(int iblock = 0; iblock < nblock; iblock++) {
    int64_t length1 = block_end[iblock] - block_start[iblock];
    int64_t length2 = block_end2[iblock] - block_start2[iblock];

    double npair = (double)(ngrp1 - length1) * (double)(ngrp2 - length2);
    double dr = (log_scale) ? (log(rmax / rmin) / nr) : ((rmax - rmin) / nr);

    for(int ir = 0; ir < nr; ir++) {
      double r_low = (log_scale) ? (rmin * exp(ir * dr)) : (rmin + ir * dr);
      double r_high = (log_scale) ? (rmin * exp((ir + 1) * dr)) : (rmin + (ir + 1) * dr);
      double shell_volume = (4.0 / 3.0) * M_PI * (r_high * r_high * r_high - r_low * r_low * r_low);
      double norm = npair * shell_volume / V_box;
      xi_jk[iblock][ir] = dd_pair_jk[iblock][ir] / norm - 1.0;
    }
  }
}

template <typename T>
void correlation::resample_jk_LS(T &grp, T &rand)
{
  uint64_t ngrp = grp.size();
  uint64_t nrand = rand.size();

  /* Here only the global box size */
  const int ncx = ndiv_1d * std::max(static_cast<int>(1.0 / rmax), 1);
  const int ncy = ndiv_1d * std::max(static_cast<int>(1.0 / rmax), 1);
  const int ncz = ndiv_1d * std::max(static_cast<int>(1.0 / rmax), 1);
  const int nc3 = ncx * ncy * ncz;

  for(int iblock = 0; iblock < nblock; iblock++) {
    int64_t delete_block_start = block_start[iblock];
    int64_t delete_block_end = block_end[iblock];

    int64_t delete_rand_block_start = rand_block_start[iblock];
    int64_t delete_rand_block_end = rand_block_end[iblock];

    std::cerr << "# iblock " << iblock << " " << delete_block_start << " " << delete_block_end << " length "
              << delete_block_end - delete_block_start << std::endl;

    std::vector<std::vector<int>> cell_list(nc3);
    std::vector<std::vector<int>> cell_list_rand(nc3);

    for(int64_t i = 0; i < ngrp; i++) {
      if((delete_block_start <= i) && (i < delete_block_end)) continue;
      const int ix = get_cell_index(grp[i].xpos, ncx);
      const int iy = get_cell_index(grp[i].ypos, ncy);
      const int iz = get_cell_index(grp[i].zpos, ncz);
      const int cell_id = iz + ncz * (iy + ncy * ix);
      cell_list[cell_id].push_back(i);
    }

    for(int i = 0; i < nrand; i++) {
      if((delete_rand_block_start <= i) && (i < delete_rand_block_end)) continue;
      const int ix = get_cell_index(rand[i].xpos, ncx);
      const int iy = get_cell_index(rand[i].ypos, ncy);
      const int iz = get_cell_index(rand[i].zpos, ncz);
      const int cell_id = iz + ncz * (iy + ncy * ix);
      cell_list_rand[cell_id].push_back(i);
    }

#pragma omp parallel
    {
      int nthread = omp_get_num_threads();
      int ithread = omp_get_thread_num();
      uint64_t progress = 0;
      uint64_t progress_thread = nc3 / nthread;
      uint64_t progress_div = 1 + progress_thread / 200;

      if(ithread == 0)
        std::cerr << "# nc^3, ngrp_thread = " << nc3 << ", " << progress_thread << " in " << nthread << " threads."
                  << std::endl;

      auto thr_rr_pair = calc_pair(1.0, rand, cell_list_rand, ncx, ncy, ncz, "RR");
      auto thr_dd_pair = calc_pair(1.0, grp, cell_list, ncx, ncy, ncz, "DD");
      auto thr_dr_pair = calc_pair(0.5, grp, rand, cell_list, cell_list_rand, ncx, ncy, ncz, "DR");

#pragma omp critical
      {
        for(int ir = 0; ir < nr; ir++) {
          dd_pair_jk[iblock][ir] += thr_dd_pair[ir];
          dr_pair_jk[iblock][ir] += thr_dr_pair[ir];
          rr_pair_jk[iblock][ir] += thr_rr_pair[ir];
        }
      }
    } // end parallel
  } // end nblock loop

  for(int iblock = 0; iblock < nblock; iblock++) {
    int64_t length = block_end[iblock] - block_start[iblock];
    int64_t rand_length = rand_block_end[iblock] - rand_block_start[iblock];

    double f = (double)(nrand - rand_length) / (double)(ngrp - length);
    double f2 = f * f;

    for(int ir = 0; ir < nr; ir++) {
      if(rr_pair_jk[iblock][ir] != 0.0) {
        xi_jk[iblock][ir] = (dd_pair_jk[iblock][ir] * f2 - 2.0 * dr_pair_jk[iblock][ir] * f + rr_pair_jk[iblock][ir]) /
                            rr_pair_jk[iblock][ir];
      }
    }
  }
}

template <typename T>
void correlation::resample_jk_LS(T &grp1, T &grp2, T &rand1, T &rand2)
{
  uint64_t ngrp1 = grp1.size();
  uint64_t ngrp2 = grp2.size();
  uint64_t nrand1 = rand1.size();
  uint64_t nrand2 = rand2.size();

  /* Here only the global box size */
  const int ncx = ndiv_1d * std::max(static_cast<int>(1.0 / rmax), 1);
  const int ncy = ndiv_1d * std::max(static_cast<int>(1.0 / rmax), 1);
  const int ncz = ndiv_1d * std::max(static_cast<int>(1.0 / rmax), 1);
  const int nc3 = ncx * ncy * ncz;

  for(int iblock = 0; iblock < nblock; iblock++) {
    int64_t delete_block_start1 = block_start[iblock];
    int64_t delete_block_end1 = block_end[iblock];
    int64_t delete_block_start2 = block_start2[iblock];
    int64_t delete_block_end2 = block_end2[iblock];

    int64_t delete_rand_block_start1 = rand_block_start[iblock];
    int64_t delete_rand_block_end1 = rand_block_end[iblock];
    int64_t delete_rand_block_start2 = rand_block_start2[iblock];
    int64_t delete_rand_block_end2 = rand_block_end2[iblock];

    std::cerr << "# iblock " << iblock << " " << delete_block_start1 << " " << delete_block_end1 << " length "
              << delete_block_end1 - delete_block_start1 << std::endl;

    std::vector<std::vector<int>> cell_list1(nc3);
    std::vector<std::vector<int>> cell_list2(nc3);
    std::vector<std::vector<int>> cell_list_rand1(nc3);
    std::vector<std::vector<int>> cell_list_rand2(nc3);

    for(int64_t i = 0; i < ngrp1; i++) {
      if((delete_block_start1 <= i) && (i < delete_block_end1)) continue;
      const int ix = get_cell_index(grp1[i].xpos, ncx);
      const int iy = get_cell_index(grp1[i].ypos, ncy);
      const int iz = get_cell_index(grp1[i].zpos, ncz);
      const int cell_id = iz + ncz * (iy + ncy * ix);
      cell_list1[cell_id].push_back(i);
    }

    for(int64_t i = 0; i < ngrp2; i++) {
      if((delete_block_start2 <= i) && (i < delete_block_end2)) continue;
      const int ix = get_cell_index(grp2[i].xpos, ncx);
      const int iy = get_cell_index(grp2[i].ypos, ncy);
      const int iz = get_cell_index(grp2[i].zpos, ncz);
      const int cell_id = iz + ncz * (iy + ncy * ix);
      cell_list2[cell_id].push_back(i);
    }

    for(int i = 0; i < nrand1; i++) {
      if((delete_rand_block_start1 <= i) && (i < delete_rand_block_end1)) continue;
      const int ix = get_cell_index(rand1[i].xpos, ncx);
      const int iy = get_cell_index(rand1[i].ypos, ncy);
      const int iz = get_cell_index(rand1[i].zpos, ncz);
      const int cell_id = iz + ncz * (iy + ncy * ix);
      cell_list_rand1[cell_id].push_back(i);
    }

    for(int i = 0; i < nrand2; i++) {
      if((delete_rand_block_start2 <= i) && (i < delete_rand_block_end2)) continue;
      const int ix = get_cell_index(rand2[i].xpos, ncx);
      const int iy = get_cell_index(rand2[i].ypos, ncy);
      const int iz = get_cell_index(rand2[i].zpos, ncz);
      const int cell_id = iz + ncz * (iy + ncy * ix);
      cell_list_rand2[cell_id].push_back(i);
    }

#pragma omp parallel
    {
      int nthread = omp_get_num_threads();
      int ithread = omp_get_thread_num();
      uint64_t progress = 0;
      uint64_t progress_thread = nc3 / nthread;
      uint64_t progress_div = 1 + progress_thread / 200;

      if(ithread == 0)
        std::cerr << "# nc^3, ngrp_thread = " << nc3 << ", " << progress_thread << " in " << nthread << " threads."
                  << std::endl;

      auto thr_rr_pair = calc_pair(1.0, rand1, rand2, cell_list_rand1, cell_list_rand2, ncx, ncy, ncz, "R1R2");
      auto thr_dr_pair = calc_pair(1.0, grp1, rand2, cell_list1, cell_list_rand2, ncx, ncy, ncz, "D1R2");
      auto thr_dr2_pair = calc_pair(1.0, grp2, rand1, cell_list2, cell_list_rand1, ncx, ncy, ncz, "D2R1");
      auto thr_dd_pair = calc_pair(1.0, grp1, grp2, cell_list1, cell_list2, ncx, ncy, ncz, "D1D2");

#pragma omp critical
      {
        for(int ir = 0; ir < nr; ir++) {
          dd_pair_jk[iblock][ir] += thr_dd_pair[ir];
          dr_pair_jk[iblock][ir] += thr_dr_pair[ir];
          dr2_pair_jk[iblock][ir] += thr_dr2_pair[ir];
          rr_pair_jk[iblock][ir] += thr_rr_pair[ir];
        }
      }
    } // end parallel
  } // end nblock loop

  for(int iblock = 0; iblock < nblock; iblock++) {
    int64_t length1 = block_end[iblock] - block_start[iblock];
    int64_t length2 = block_end2[iblock] - block_start2[iblock];

    int64_t rand_length1 = rand_block_end[iblock] - rand_block_start[iblock];
    int64_t rand_length2 = rand_block_end2[iblock] - rand_block_start2[iblock];

    double f1 = (double)(nrand1 - rand_length1) / (double)(ngrp1 - length1);
    double f2 = (double)(nrand2 - rand_length2) / (double)(ngrp2 - length2);
    double f12 = f1 * f2;

    for(int ir = 0; ir < nr; ir++) {
      if(rr_pair_jk[iblock][ir] != 0.0) {
        xi_jk[iblock][ir] = (dd_pair_jk[iblock][ir] * f12 - dr_pair_jk[iblock][ir] * f1 - dr2_pair_jk[iblock][ir] * f2 +
                             rr_pair_jk[iblock][ir]) /
                            rr_pair_jk[iblock][ir];
      }
    }
  }
}

template <typename T>
void correlation::calc_xi_ifft_impl(T &mesh, T &weight)
{
  static_assert(!std::is_same<T, float>::value, "Only float is allowed");

  check_p();

  static bool fft_init = false;
  if(fft_init == false) {
    fftwf_init_threads();
    fftwf_plan_with_nthreads(omp_get_max_threads());
    fft_init = true;
  }

  const double pk_norm = (double)(nmesh * nmesh) * (double)(nmesh * nmesh) * (double)(nmesh * nmesh);
  const auto norm = 1.0 / pk_norm;

  xi.assign(nr, 0.0);
  weight.assign(nr, 0);

  /* fft */
  fftwf_plan plan_forward =
      fftwf_plan_dft_r2c_3d(nmesh, nmesh, nmesh, mesh.data(), (fftwf_complex *)mesh.data(), FFTW_ESTIMATE);
  fftwf_execute(plan_forward);

  fftwf_complex *mesh_hat = (fftwf_complex *)mesh.data();

#pragma omp parallel for collapse(3)
  for(uint64_t ix = 0; ix < nmesh; ix++) {
    for(uint64_t iy = 0; iy < nmesh; iy++) {
      for(uint64_t iz = 0; iz < nmesh / 2 + 1; iz++) {
        int64_t im = iz + (nmesh / 2 + 1) * (iy + nmesh * ix);
        auto win2 = calc_window(ix, iy, iz);
        auto power = SQR(c_re(mesh_hat[im])) + SQR(c_im(mesh_hat[im]));
        c_re(mesh_hat[im]) = (power / win2) - shotnoise;
        c_im(mesh_hat[im]) = 0.0;
      }
    }
  } // ix,iy,iz loop

  /* ifft */
  fftwf_plan plan_backward = fftwf_plan_dft_c2r_3d(nmesh, nmesh, nmesh, mesh_hat, mesh.data(), FFTW_ESTIMATE);
  fftwf_execute(plan_backward);

#pragma omp parallel for
  for(int64_t i = 0; i < mesh.size(); i++) {
    mesh[i] *= norm;
  }

#pragma omp parallel for collapse(3) reduction(vec_double_plus : xi) reduction(vec_float_plus : weight)
  for(uint64_t ix = 0; ix < nmesh; ix++) {
    for(uint64_t iy = 0; iy < nmesh; iy++) {
      for(uint64_t iz = 0; iz < nmesh; iz++) {

        int64_t im = iz + (nmesh + 2) * (iy + nmesh * ix);
        const float dx = (ix < nmesh / 2) ? (float)(ix) : (float)(nmesh - ix);
        const float dy = (iy < nmesh / 2) ? (float)(iy) : (float)(nmesh - iy);
        const float dz = (iz < nmesh / 2) ? (float)(iz) : (float)(nmesh - iz);

        float r = sqrt(dx * dx + dy * dy + dz * dz) / (float)(nmesh);
        const int ir = get_r_index(r);

        if(ir >= 0 && ir < nr) {
          xi[ir] += mesh[im];
          weight[ir]++;
        }
      }
    }
  } // ix, iy, iz loop

#pragma omp parallel for
  for(int ir = 0; ir < nr; ir++) {
    if(weight[ir] > 0) {
      xi[ir] /= weight[ir];
    }
  }

  fftwf_destroy_plan(plan_forward);
  fftwf_destroy_plan(plan_backward);
}

template <typename T>
void correlation::calc_xi_ifft_impl(T &mesh1, T &mesh2, T &weight)
{
  static_assert(!std::is_same<T, float>::value, "Only float is allowed");

  assert(mesh1.size() == mesh2.size());

  check_p();

  static bool fft_init = false;
  if(fft_init == false) {
    fftwf_init_threads();
    fftwf_plan_with_nthreads(omp_get_max_threads());
    fft_init = true;
  }

  xi.assign(nr, 0.0);
  weight.assign(nr, 0);

  /* fft */
  fftwf_plan plan_forward =
      fftwf_plan_dft_r2c_3d(nmesh, nmesh, nmesh, mesh1.data(), (fftwf_complex *)mesh1.data(), FFTW_ESTIMATE);
  fftwf_execute_dft_r2c(plan_forward, mesh1.data(), (fftwf_complex *)(mesh1.data()));
  fftwf_execute_dft_r2c(plan_forward, mesh2.data(), (fftwf_complex *)(mesh2.data()));

  fftwf_complex *mesh_hat1 = (fftwf_complex *)mesh1.data();
  fftwf_complex *mesh_hat2 = (fftwf_complex *)mesh2.data();

#pragma omp parallel for collapse(3)
  for(uint64_t ix = 0; ix < nmesh; ix++) {
    for(uint64_t iy = 0; iy < nmesh; iy++) {
      for(uint64_t iz = 0; iz < nmesh / 2 + 1; iz++) {
        int64_t im = iz + (nmesh / 2 + 1) * (iy + nmesh * ix);
        auto win2 = calc_window(ix, iy, iz);
        auto power = (c_re(mesh_hat1[im]) * c_re(mesh_hat2[im])) + (c_im(mesh_hat1[im]) * c_im(mesh_hat2[im]));
        c_re(mesh_hat1[im]) = power / win2;
        c_im(mesh_hat1[im]) = 0.0;
      }
    }
  } // ix,iy,iz loop

  /* ifft */
  fftwf_plan plan_backward =
      fftwf_plan_dft_c2r_3d(nmesh, nmesh, nmesh, (fftwf_complex *)mesh1.data(), mesh1.data(), FFTW_ESTIMATE);
  fftwf_execute_dft_c2r(plan_backward, (fftwf_complex *)(mesh1.data()), mesh1.data());

  const double pk_norm = (double)(nmesh * nmesh) * (double)(nmesh * nmesh) * (double)(nmesh * nmesh);
  const auto norm = 1.0 / pk_norm;
#pragma omp parallel for
  for(int64_t i = 0; i < mesh1.size(); i++) {
    mesh1[i] *= norm;
  }

#pragma omp parallel for collapse(3) reduction(vec_double_plus : xi) reduction(vec_float_plus : weight)
  for(uint64_t ix = 0; ix < nmesh; ix++) {
    for(uint64_t iy = 0; iy < nmesh; iy++) {
      for(uint64_t iz = 0; iz < nmesh; iz++) {

        int64_t im = iz + (nmesh + 2) * (iy + nmesh * ix);
        const float dx = (ix < nmesh / 2) ? (float)(ix) : (float)(nmesh - ix);
        const float dy = (iy < nmesh / 2) ? (float)(iy) : (float)(nmesh - iy);
        const float dz = (iz < nmesh / 2) ? (float)(iz) : (float)(nmesh - iz);

        float r = sqrt(dx * dx + dy * dy + dz * dz) / (float)(nmesh);
        const int ir = get_r_index(r);

        if(ir >= 0 && ir < nr) {
          xi[ir] += mesh1[im];
          weight[ir]++;
        }
      }
    }
  } // ix, iy, iz loop

#pragma omp parallel for
  for(int ir = 0; ir < nr; ir++) {
    if(weight[ir] > 0) {
      xi[ir] /= weight[ir];
    }
  }

  fftwf_destroy_plan(plan_forward);
  fftwf_destroy_plan(plan_backward);
}

template <typename T>
void correlation::calc_xi_jk_ifft_impl(T &mesh_orig, T &weight)
{
  static_assert(!std::is_same<T, float>::value, "Only float is allowed");

  check_p();

  static bool fft_init = false;
  if(fft_init == false) {
    fftwf_init_threads();
    fftwf_plan_with_nthreads(omp_get_max_threads());
    fft_init = true;
  }

  nblock = jk_level * jk_level * jk_level;

  // for calc_jk_xi_error
  uint64_t ngrp = mesh_orig.size();
  jk_dd = mesh_orig.size() / nblock;

  std::vector<T> weight_jk; // size[block][nr]

  xi_ave.assign(nr, 0.0);
  xi_sd.assign(nr, 0.0);
  xi_se.assign(nr, 0.0);
  weight.assign(nr, 0.0);

  xi_jk.resize(nblock, std::vector<double>(nr));
  weight_jk.resize(nblock, T(nr));

#pragma omp parallel for collapse(2)
  for(int iblock = 0; iblock < nblock; iblock++) {
    for(int ir = 0; ir < nr; ir++) {
      xi_jk[iblock][ir] = 0.0;
      weight_jk[iblock][ir] = 0.0;
    }
  }

  T mesh(mesh_orig.size());

  fftwf_complex *mesh_hat = (fftwf_complex *)mesh.data();

  /* fft */
  fftwf_plan plan_forward =
      fftwf_plan_dft_r2c_3d(nmesh, nmesh, nmesh, mesh.data(), (fftwf_complex *)mesh.data(), FFTW_ESTIMATE);
  fftwf_plan plan_backward =
      fftwf_plan_dft_c2r_3d(nmesh, nmesh, nmesh, (fftwf_complex *)mesh.data(), mesh.data(), FFTW_ESTIMATE);

  for(int iblock = 0; iblock < jk_level * jk_level * jk_level; iblock++) {

    std::cerr << "# iblock " << iblock << " / " << nblock << std::endl;

    int ibx = iblock / (jk_level * jk_level);
    int iby = (iblock - ibx * (jk_level * jk_level)) / jk_level;
    int ibz = iblock - ibx * (jk_level * jk_level) - iby * jk_level;

    int bnx = nmesh / jk_level;
    int bny = nmesh / jk_level;
    int bnz = nmesh / jk_level;
    int ix_min = ibx * bnx;
    int ix_max = (ibx + 1) * bnx;
    int iy_min = iby * bny;
    int iy_max = (iby + 1) * bny;
    int iz_min = ibz * bnz;
    int iz_max = (ibz + 1) * bnz;

    double fblock = (double)nblock / (double)(nblock - 1);
    double amp_correction = sqrt(fblock);
    uint64_t count = 0.0;

#pragma omp parallel for collapse(3) reduction(+ : count)
    for(int64_t ix = 0; ix < nmesh; ix++) {
      for(int64_t iy = 0; iy < nmesh; iy++) {
        for(int64_t iz = 0; iz < nmesh; iz++) {
          int64_t im = iz + (nmesh + 2) * (iy + nmesh * ix);

          const bool in_block =
              (ix >= ix_min && ix < ix_max) && (iy >= iy_min && iy < iy_max) && (iz >= iz_min && iz < iz_max);

          if(in_block) {
            mesh[im] = 0.0;
          } else {
            mesh[im] = mesh_orig[im] * amp_correction;
            count++;
          }
        }
      }
    }

    double mean_with_zero = 0.0;
#pragma omp parallel for collapse(3) reduction(+ : mean_with_zero)
    for(int64_t ix = 0; ix < nmesh; ix++) {
      for(int64_t iy = 0; iy < nmesh; iy++) {
        for(int64_t iz = 0; iz < nmesh; iz++) {
          int64_t im = iz + (nmesh + 2) * (iy + nmesh * ix);
          mean_with_zero += mesh[im];
        }
      }
    }

    mean_with_zero /= count;

#pragma omp parallel for collapse(3)
    for(int64_t ix = 0; ix < nmesh; ix++) {
      for(int64_t iy = 0; iy < nmesh; iy++) {
        for(int64_t iz = 0; iz < nmesh; iz++) {
          int64_t im = iz + (nmesh + 2) * (iy + nmesh * ix);

          const bool in_block =
              (ix >= ix_min && ix < ix_max) && (iy >= iy_min && iy < iy_max) && (iz >= iz_min && iz < iz_max);

          if(!in_block) {
            mesh[im] -= mean_with_zero;
          }
        }
      }
    }

    /* fft */
    fftwf_execute(plan_forward);

#pragma omp parallel for collapse(3)
    for(uint64_t ix = 0; ix < nmesh; ix++) {
      for(uint64_t iy = 0; iy < nmesh; iy++) {
        for(uint64_t iz = 0; iz < nmesh / 2 + 1; iz++) {
          int64_t im = iz + (nmesh / 2 + 1) * (iy + nmesh * ix);
          auto win2 = calc_window(ix, iy, iz);
          auto power = SQR(c_re(mesh_hat[im])) + SQR(c_im(mesh_hat[im]));
          c_re(mesh_hat[im]) = (power / win2) - shotnoise;
          c_im(mesh_hat[im]) = 0.0;
        }
      }
    } // ix,iy,iz loop
    /* ifft */
    fftwf_execute(plan_backward);

    const double pk_norm = (double)(nmesh * nmesh) * (double)(nmesh * nmesh) * (double)(nmesh * nmesh);
    const auto norm = 1.0 / pk_norm;
#pragma omp parallel for
    for(int64_t i = 0; i < mesh.size(); i++) {
      mesh[i] *= norm;
    }

#pragma omp parallel for collapse(3) reduction(vec_double_plus : xi) reduction(vec_float_plus : weight)
    for(uint64_t ix = 0; ix < nmesh; ix++) {
      for(uint64_t iy = 0; iy < nmesh; iy++) {
        for(uint64_t iz = 0; iz < nmesh; iz++) {

          int64_t im = iz + (nmesh + 2) * (iy + nmesh * ix);
          const float dx = (ix < nmesh / 2) ? (float)(ix) : (float)(nmesh - ix);
          const float dy = (iy < nmesh / 2) ? (float)(iy) : (float)(nmesh - iy);
          const float dz = (iz < nmesh / 2) ? (float)(iz) : (float)(nmesh - iz);

          float r = sqrt(dx * dx + dy * dy + dz * dz) / (float)(nmesh);
          const int ir = get_r_index(r);

          if(ir >= 0 && ir < nr) {
            xi_jk[iblock][ir] += mesh[im];
            weight_jk[iblock][ir]++;
          }
        }
      }
    } // ix, iy, iz loop

#pragma omp parallel for
    for(int ir = 0; ir < nr; ir++) {
      if(weight_jk[iblock][ir] > 0) {
        xi_jk[iblock][ir] /= weight_jk[iblock][ir];
      }
      weight[ir] += weight_jk[iblock][ir] / (double)nblock;
    }
  } // iblock loop

  fftwf_destroy_plan(plan_forward);
  fftwf_destroy_plan(plan_backward);

  mesh.clear();
  mesh.shrink_to_fit();

  calc_jk_xi_average();
  calc_jk_xi_error();
}

template <typename T>
void correlation::calc_xi_jk_ifft_impl(T &mesh_orig1, T &mesh_orig2, T &weight)
{
  static_assert(!std::is_same<T, float>::value, "Only float is allowed");

  assert(mesh_orig1.size() == mesh_orig2.size());

  check_p();

  static bool fft_init = false;
  if(fft_init == false) {
    fftwf_init_threads();
    fftwf_plan_with_nthreads(omp_get_max_threads());
    fft_init = true;
  }

  nblock = jk_level * jk_level * jk_level;

  // for calc_jk_xi_error
  int64_t ngrp = mesh_orig1.size();
  jk_dd = mesh_orig1.size() / nblock;

  std::vector<T> weight_jk; // size[block][nr]

  xi_ave.assign(nr, 0.0);
  xi_sd.assign(nr, 0.0);
  xi_se.assign(nr, 0.0);
  weight.assign(nr, 0.0);

  xi_jk.resize(nblock, std::vector<double>(nr));
  weight_jk.resize(nblock, T(nr));

#pragma omp parallel for collapse(2)
  for(int iblock = 0; iblock < nblock; iblock++) {
    for(int ir = 0; ir < nr; ir++) {
      xi_jk[iblock][ir] = 0.0;
      weight_jk[iblock][ir] = 0.0;
    }
  }

  T mesh1(mesh_orig1.size());
  T mesh2(mesh_orig2.size());

  fftwf_complex *mesh_hat1 = (fftwf_complex *)mesh1.data();
  fftwf_complex *mesh_hat2 = (fftwf_complex *)mesh2.data();

  /* fft */
  fftwf_plan plan_forward =
      fftwf_plan_dft_r2c_3d(nmesh, nmesh, nmesh, mesh1.data(), (fftwf_complex *)mesh1.data(), FFTW_ESTIMATE);
  fftwf_plan plan_backward =
      fftwf_plan_dft_c2r_3d(nmesh, nmesh, nmesh, (fftwf_complex *)mesh1.data(), mesh1.data(), FFTW_ESTIMATE);

  for(int iblock = 0; iblock < jk_level * jk_level * jk_level; iblock++) {

    std::cerr << "# iblock " << iblock << " / " << nblock << std::endl;

    int ibx = iblock / (jk_level * jk_level);
    int iby = (iblock - ibx * (jk_level * jk_level)) / jk_level;
    int ibz = iblock - ibx * (jk_level * jk_level) - iby * jk_level;

    int bnx = nmesh / jk_level;
    int bny = nmesh / jk_level;
    int bnz = nmesh / jk_level;
    int ix_min = ibx * bnx;
    int ix_max = (ibx + 1) * bnx;
    int iy_min = iby * bny;
    int iy_max = (iby + 1) * bny;
    int iz_min = ibz * bnz;
    int iz_max = (ibz + 1) * bnz;

    double fblock = (double)nblock / (double)(nblock - 1);
    double amp_correction = sqrt(fblock);
    uint64_t count = 0.0;

#pragma omp parallel for collapse(3) reduction(+ : count)
    for(int64_t ix = 0; ix < nmesh; ix++) {
      for(int64_t iy = 0; iy < nmesh; iy++) {
        for(int64_t iz = 0; iz < nmesh; iz++) {
          int64_t im = iz + (nmesh + 2) * (iy + nmesh * ix);

          const bool in_block =
              (ix >= ix_min && ix < ix_max) && (iy >= iy_min && iy < iy_max) && (iz >= iz_min && iz < iz_max);

          if(in_block) {
            mesh1[im] = 0.0;
            mesh2[im] = 0.0;
          } else {
            mesh1[im] = mesh_orig1[im] * amp_correction;
            mesh2[im] = mesh_orig2[im] * amp_correction;
            count++;
          }
        }
      }
    }

    double mean_with_zero1 = 0.0;
    double mean_with_zero2 = 0.0;
#pragma omp parallel for collapse(3) reduction(+ : mean_with_zero1, mean_with_zero2)
    for(int64_t ix = 0; ix < nmesh; ix++) {
      for(int64_t iy = 0; iy < nmesh; iy++) {
        for(int64_t iz = 0; iz < nmesh; iz++) {
          int64_t im = iz + (nmesh + 2) * (iy + nmesh * ix);
          mean_with_zero1 += mesh1[im];
          mean_with_zero2 += mesh2[im];
        }
      }
    }

    mean_with_zero1 /= count;
    mean_with_zero2 /= count;

#pragma omp parallel for collapse(3)
    for(int64_t ix = 0; ix < nmesh; ix++) {
      for(int64_t iy = 0; iy < nmesh; iy++) {
        for(int64_t iz = 0; iz < nmesh; iz++) {
          int64_t im = iz + (nmesh + 2) * (iy + nmesh * ix);

          const bool in_block =
              (ix >= ix_min && ix < ix_max) && (iy >= iy_min && iy < iy_max) && (iz >= iz_min && iz < iz_max);

          if(!in_block) {
            mesh1[im] -= mean_with_zero1;
            mesh2[im] -= mean_with_zero2;
          }
        }
      }
    }

    /* fft */
    fftwf_execute_dft_r2c(plan_forward, mesh1.data(), (fftwf_complex *)(mesh1.data()));
    fftwf_execute_dft_r2c(plan_forward, mesh2.data(), (fftwf_complex *)(mesh2.data()));

#pragma omp parallel for collapse(3)
    for(uint64_t ix = 0; ix < nmesh; ix++) {
      for(uint64_t iy = 0; iy < nmesh; iy++) {
        for(uint64_t iz = 0; iz < nmesh / 2 + 1; iz++) {
          int64_t im = iz + (nmesh / 2 + 1) * (iy + nmesh * ix);
          auto win2 = calc_window(ix, iy, iz);
          auto power = (c_re(mesh_hat1[im]) * c_re(mesh_hat2[im])) + (c_im(mesh_hat1[im]) * c_im(mesh_hat2[im]));
          c_re(mesh_hat1[im]) = power / win2;
          c_im(mesh_hat1[im]) = 0.0;
        }
      }
    } // ix,iy,iz loop

    /* ifft */
    fftwf_execute_dft_c2r(plan_backward, (fftwf_complex *)(mesh1.data()), mesh1.data());

    const double pk_norm = (double)(nmesh * nmesh) * (double)(nmesh * nmesh) * (double)(nmesh * nmesh);
    const auto norm = 1.0 / pk_norm;
#pragma omp parallel for
    for(int64_t i = 0; i < mesh1.size(); i++) {
      mesh1[i] *= norm;
    }

#pragma omp parallel for collapse(3) reduction(vec_double_plus : xi) reduction(vec_float_plus : weight)
    for(uint64_t ix = 0; ix < nmesh; ix++) {
      for(uint64_t iy = 0; iy < nmesh; iy++) {
        for(uint64_t iz = 0; iz < nmesh; iz++) {

          int64_t im = iz + (nmesh + 2) * (iy + nmesh * ix);
          const float dx = (ix < nmesh / 2) ? (float)(ix) : (float)(nmesh - ix);
          const float dy = (iy < nmesh / 2) ? (float)(iy) : (float)(nmesh - iy);
          const float dz = (iz < nmesh / 2) ? (float)(iz) : (float)(nmesh - iz);

          float r = sqrt(dx * dx + dy * dy + dz * dz) / (float)(nmesh);
          const int ir = get_r_index(r);

          if(ir >= 0 && ir < nr) {
            xi_jk[iblock][ir] += mesh1[im];
            weight_jk[iblock][ir]++;
          }
        }
      }
    } // ix, iy, iz loop

#pragma omp parallel for
    for(int ir = 0; ir < nr; ir++) {
      if(weight_jk[iblock][ir] > 0) {
        xi_jk[iblock][ir] /= weight_jk[iblock][ir];
      }
      weight[ir] += weight_jk[iblock][ir] / (double)nblock;
    }
  } // iblock loop

  fftwf_destroy_plan(plan_forward);
  fftwf_destroy_plan(plan_backward);

  mesh1.clear();
  mesh1.shrink_to_fit();
  mesh2.clear();
  mesh2.shrink_to_fit();

  calc_jk_xi_average();
  calc_jk_xi_error();
}

void correlation::output_xi(std::string filename)
{
  std::ofstream fout(filename);
  if(jk_block <= 1) {
    if(use_LS) {
      fout << "# r[Mpc/h] xi DD DR RR" << std::endl;
      for(int ir = 0; ir < nr; ir++) {
        double rad = rcen[ir] * lbox;
        fout << std::scientific << std::setprecision(10) << rad << " " << xi[ir] << " " << dd_pair[ir] << " "
             << dr_pair[ir] << " " << rr_pair[ir] << "\n";
      }
    } else {
      fout << "# r[Mpc/h] xi DD RR" << std::endl;
        for(int ir = 0; ir < nr; ir++) {
          double rad = rcen[ir] * lbox;
        fout << std::scientific << std::setprecision(10) << rad << " " << xi[ir] << " " << dd_pair[ir] << " "
             << rr_pair[ir] << "\n";
      }
    }

  } else {
    fout << "# r[Mpc/h] xi_ave SD SE block1 block2 block3 ..." << std::endl;
    for(int ir = 0; ir < nr; ir++) {
      double rad = rcen[ir] * lbox;
      fout << std::scientific << std::setprecision(10) << rad << " " << xi_ave[ir] << " " << xi_sd[ir] << " "
           << xi_se[ir];
      for(int i = 0; i < nblock; i++) fout << std::scientific << std::setprecision(10) << " " << xi_jk[i][ir];
      fout << "\n";
    }
  }

  fout.flush();
  fout.close();
  std::cout << "output to " << filename << std::endl;
}

template <typename T>
void correlation::output_xi(std::string filename, T &weight)
{
  std::ofstream fout(filename);
  if(jk_block <= 1) {
    fout << "# r[Mpc/h] xi weight" << std::endl;
    for(int ir = 0; ir < nr; ir++) {
      double rad = rcen[ir] * lbox;
      fout << std::scientific << std::setprecision(10) << rad << " " << xi[ir] << " " << weight[ir] << "\n";
    }

  } else {
    fout << "# r[Mpc/h] xi_ave SD SE block1 block2 block3 ... weight" << std::endl;
    for(int ir = 0; ir < nr; ir++) {
      double rad = rcen[ir] * lbox;
      fout << std::scientific << std::setprecision(10) << rad << " " << xi_ave[ir] << " " << xi_sd[ir] << " "
           << xi_se[ir];
      for(int i = 0; i < nblock; i++) fout << std::scientific << std::setprecision(10) << " " << xi_jk[i][ir];
      fout << " " << weight[ir] << "\n";
    }
  }
  fout.flush();
  fout.close();
  std::cout << "output to " << filename << std::endl;
}

void correlation::output_xi_smu(std::string filename)
{
  std::ofstream fout(filename);
  if(jk_block <= 1) {
    if(use_LS) {
      fout << "# r[Mpc/h] mu xi DD DR RR" << std::endl;
      for(int ir = 0; ir < nr; ir++) {
        double rad = rcen[ir] * lbox;
        for(int imu = 0; imu < nmu; imu++) {
          auto idx = imu + nmu * ir;
          fout << std::scientific << std::setprecision(10) << rad << " " << mucen[imu] << " " << xi[idx] << " "
               << dd_pair[idx] << " " << dr_pair[idx] << " " << rr_pair[idx] << "\n";
        } // imu
      } // ir
    } else {
      if(dd_pair.size() > 0) {
        fout << "# r[Mpc/h] mu xi DD" << std::endl;
        for(int ir = 0; ir < nr; ir++) {
          double rad = rcen[ir] * lbox;
          for(int imu = 0; imu < nmu; imu++) {
            auto idx = imu + nmu * ir;
            fout << std::scientific << std::setprecision(10) << rad << " " << mucen[imu] << " " << xi[idx] << " "
                 << dd_pair[idx] << "\n";
          } // imu
        } // ir
      } else {
        fout << "# r[Mpc/h] mu xi" << std::endl;
        for(int ir = 0; ir < nr; ir++) {
          double rad = rcen[ir] * lbox;
          for(int imu = 0; imu < nmu; imu++) {
            auto idx = imu + nmu * ir;
            fout << std::scientific << std::setprecision(10) << rad << " " << mucen[imu] << " " << xi[idx] << "\n";
          } // imu
        } // ir
      }
    }
  } else {
    fout << "# r[Mpc/h] mu xi_ave SD SE block1 block2 block3 ..." << std::endl;
    for(int ir = 0; ir < nr; ir++) {
      double rad = rcen[ir] * lbox;
      for(int imu = 0; imu < nmu; imu++) {
        auto idx = imu + nmu * ir;
        fout << std::scientific << std::setprecision(10) << rad << " " << mucen[imu] << " " << xi_ave[idx] << " "
             << xi_sd[idx] << " " << xi_se[idx];
        for(int i = 0; i < nblock; i++) fout << std::scientific << std::setprecision(10) << " " << xi_jk[i][idx];
        fout << "\n";
      } // imu
    } // ir
  }

  fout.flush();
  fout.close();
  std::cout << "output to " << filename << std::endl;
}

void correlation::output_xi_spsp(std::string filename)
{
  std::ofstream fout(filename);
  if(jk_block <= 1) {
    if(use_LS) {
      fout << "# s_perp[Mpc/h] s_para[Mpc/h] xi DD DR RR" << std::endl;
      for(int iperp = 0; iperp < nsperp; iperp++) {
        for(int ipara = 0; ipara < nspara; ipara++) {
          auto spara = sparacen[ipara] * lbox;
          auto sperp = sperpcen[iperp] * lbox;
          auto idx = ipara + nspara * iperp;
          fout << std::scientific << std::setprecision(10) << sperp << " " << spara << " " << xi[idx] << " "
               << dd_pair[idx] << " " << dr_pair[idx] << " " << rr_pair[idx] << "\n";
        }
      }
    } else {
      if(dd_pair.size() > 0) {
        fout << "# s_perp[Mpc/h] s_para[Mpc/h] xi DD" << std::endl;
        for(int iperp = 0; iperp < nsperp; iperp++) {
          for(int ipara = 0; ipara < nspara; ipara++) {
            auto spara = sparacen[ipara] * lbox;
            auto sperp = sperpcen[iperp] * lbox;
            auto idx = ipara + nspara * iperp;
            fout << std::scientific << std::setprecision(10) << sperp << " " << spara << " " << xi[idx] << " "
                 << dd_pair[idx] << "\n";
          }
        }
      } else {
        fout << "# s_perp[Mpc/h] s_para[Mpc/h] xi" << std::endl;
        for(int iperp = 0; iperp < nsperp; iperp++) {
          for(int ipara = 0; ipara < nspara; ipara++) {
            auto spara = sparacen[ipara] * lbox;
            auto sperp = sperpcen[iperp] * lbox;
            auto idx = ipara + nspara * iperp;
            fout << std::scientific << std::setprecision(10) << sperp << " " << spara << " " << xi[idx] << "\n";
          }
        }
      }
    }
  } else {
    fout << "# s_perp[Mpc/h] s_para[Mpc/h] xi_ave SD SE block1 block2 ..." << std::endl;
    for(int iperp = 0; iperp < nsperp; iperp++) {
      for(int ipara = 0; ipara < nspara; ipara++) {
        auto spara = sparacen[ipara] * lbox;
        auto sperp = sperpcen[iperp] * lbox;
        auto idx = ipara + nspara * iperp;
        fout << std::scientific << std::setprecision(10) << sperp << " " << spara << " " << xi_ave[idx] << " "
             << xi_sd[idx] << " " << xi_se[idx];
        for(int i = 0; i < nblock; i++) fout << std::scientific << std::setprecision(10) << " " << xi_jk[i][idx];
        fout << "\n";
      }
    }
  }

  fout.flush();
  fout.close();
  std::cout << "output to " << filename << std::endl;
}
