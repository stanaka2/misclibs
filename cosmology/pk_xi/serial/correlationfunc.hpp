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
  enum class Mode { R, SMU, SPSP };
  enum class Estimator { IDEAL, RR, LS, IFFT };

  int nr = 100;
  double rmin, rmax;
  std::vector<float> rcen; // bin center

  double rmin2, rmax2;    // r, r^2 base
  double lin_dr, lin_dr2; // r, r^2 base

  /* radius of searching cell : (ndiv_1d+2)^3 */
  // int ndiv_1d = 1;
  int ndiv_1d = 4;

  bool symmetry = true; // true: count each pair once (i<j only)
                        // false: count both directions

  int los = 2; // default : z-axis
  int nmu;
  double mumin, mumax, dmu;
  std::vector<float> mucen; // bin center
  int nsperp, nspara;
  double sperp_min, sperp_max, dsperp;
  double spara_min, spara_max, dspara;
  std::vector<float> sperpcen, sparacen; // bin center
  bool half_angle = false;

  Mode mode = Mode::R; // R:xi(r), SMU:xi(s,mu), SPSP:xi(s_perp,s_para)

  // for position xi
  // support Landy SD, Szalay AS 1993, Apj
  Estimator est = Estimator::IDEAL;
  int64_t nrand_factor = 1;
  std::vector<double> dd_pair, dr_pair, dr2_pair, rr_pair; // size[nr]
  std::vector<double> xi, xi_ave, xi_se;                   // size[nr]

  void set_cor_estimator(const std::string &s)
  {
    std::string upper = s;
    std::transform(upper.begin(), upper.end(), upper.begin(), [](unsigned char c) { return std::toupper(c); });
    if(upper == "IDEAL") est = Estimator::IDEAL;
    else if(upper == "RR") est = Estimator::RR;
    else if(upper == "LS") est = Estimator::LS;
    else if(upper == "IFFT") est = Estimator::IFFT;
  }

  void set_cor_mode(const std::string &s)
  {
    std::string upper = s;
    std::transform(upper.begin(), upper.end(), upper.begin(), [](unsigned char c) { return std::toupper(c); });
    if(upper == "R") mode = Mode::R;
    else if(upper == "S") mode = Mode::R;
    else if(upper == "SMU") mode = Mode::SMU;
    else if(upper == "SPSP") mode = Mode::SPSP;
  }

  void set_rbin(double, double, int, double, bool = true);
  void set_rbin2D(double, double, int, double, double, int, double);

  void check_rbin();

  template <typename T>
  void calc_xi(T &);
  template <typename T>
  void calc_xi(T &, T &);
  template <typename T>
  void calc_xi_ifft(T &, T &);
  template <typename T>
  void calc_xi_ifft(T &, T &, T &);

  void output_xi(std::string);
  template <typename T>
  void output_xi(std::string, T &);
  void output_xi2D(std::string);

private:
  void set_smubin(double, double, int, double, double, int, double);
  void set_spspbin(double, double, int, double, double, int, double);

  template <typename T>
  int get_r_index(T);
  template <typename T>
  int get_r2_index(T);
  template <typename T>
  int get_ir_from_dr(T, T, T);
  template <typename T>
  int get_imu_from_dr(T, T, T);
  template <typename T>
  int get_cell_index(T, int);

  template <typename T>
  T get_ismu_from_dr(T, T, T, int &, int &);
  template <typename T>
  T get_ispsp_from_dr(T, T, T, int &, int &);

  template <typename T>
  void shuffle_data(T &, const int = 10);

  void calc_ideal_pair_r(uint64_t);
  void calc_ideal_smu_pair(uint64_t);
  void calc_ideal_spsp_pair(uint64_t);
  void calc_ideal_pair(uint64_t);
  void calc_xi_from_pair(uint64_t);
  void calc_xi_from_pair(uint64_t, uint64_t);

  template <typename G, typename C>
  G sort_group_by_cell(const G &, const int, const int, const int, C &);

  template <typename G, typename C>
  std::vector<double> calc_pair(const double, G &, C &, int, int, int, const std::string &);
  template <typename G, typename C>
  std::vector<double> calc_pair(const double, G &, G &, C &, C &, int, int, int, const std::string &);
  template <typename G, typename C>
  std::vector<double> calc_pair_smu(const double, G &, C &, int, int, int, const std::string &);
  template <typename G, typename C>
  std::vector<double> calc_pair_smu(const double, G &, G &, C &, C &, int, int, int, const std::string &);
  template <typename G, typename C>
  std::vector<double> calc_pair_spsp(const double, G &, C &, int, int, int, const std::string &);
  template <typename G, typename C>
  std::vector<double> calc_pair_spsp(const double, G &, G &, C &, C &, int, int, int, const std::string &);

  template <typename G, typename C>
  std::vector<double> calc_cor_pair(const double, G &, C &, int, int, int, const std::string &);
  template <typename G, typename C>
  std::vector<double> calc_cor_pair(const double, G &, G &, C &, C &, int, int, int, const std::string &);

  template <typename T>
  void calc_xi_impl(T &);
  template <typename T>
  void calc_xi_impl(T &, T &);

  template <typename T>
  void calc_xi_ifft_impl(T &, T &);
  template <typename T>
  void calc_xi_ifft_impl(T &, T &, T &);

  template <typename T>
  T xismu_to_xir(const T &, const T &) const;
  template <typename T>
  T xispsp_to_wpr(const T &, const T &) const;

  void output_xi1D_smu(std::string);
  void output_xi2D_smu(std::string);
  void output_xi1D_spsp(std::string);
  void output_xi2D_spsp(std::string);
};

void correlation::set_rbin(double _rmin, double _rmax, int _nr, double _lbox, bool _log_scale)
{
  nr = _nr;
  lbox = _lbox;
  rmin = _rmin / lbox;
  rmax = _rmax / lbox;
  log_scale = _log_scale;

  if(log_scale) {
    if(rmin < 1e-8) rmin = 1e-8;
  }

  rmin2 = rmin * rmin;
  rmax2 = rmax * rmax;

  rcen.assign(nr, 0.0);

  if(log_scale) {
    logratio = std::log(rmax / rmin) / (double)nr;
    logratio2 = 2 * logratio;
    for(int ir = 0; ir < nr; ir++) rcen[ir] = rmin * std::exp((ir + 0.5) * logratio);
  } else {
    lin_dr = (rmax - rmin) / (double)(nr);
    for(int ir = 0; ir < nr; ir++) rcen[ir] = rmin + lin_dr * (ir + 0.5);
  }
}

void correlation::check_rbin()
{
  for(int ir = 0; ir < nr; ir++) std::cerr << ir << " " << rcen[ir] << " " << get_r_index(rcen[ir]) << "\n";
}

void correlation::set_rbin2D(double _rmin, double _rmax, int _nr, double _mumin, double _mumax, int _nmu, double _lbox)
{
  if(mode == Mode::R) {
    std::cerr << "Mode=R is not supported." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  std::cerr << "force log_bin=false" << std::endl;
  if(mode == Mode::SMU) set_smubin(_rmin, _rmax, _nr, _mumin, _mumax, _nmu, _lbox);
  if(mode == Mode::SPSP) set_spspbin(_rmin, _rmax, _nr, _mumin, _mumax, _nmu, _lbox);
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

  if(mumin < -1e-10) half_angle = false;
  else half_angle = true;
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

  nr = _nsperp;
  rmin = _sperp_min / lbox;
  rmax = _sperp_max / lbox;
  rmin2 = rmin * rmin;
  rmax2 = rmax * rmax;

  rcen.assign(nr, 0.0);
  lin_dr = (rmax - rmin) / (double)(nr);
  for(int ir = 0; ir < nr; ir++) rcen[ir] = rmin + lin_dr * (ir + 0.5);

  nsperp = nr;
  sperp_min = rmin;
  sperp_max = rmax;
  dsperp = lin_dr;
  sperpcen = rcen;

  nspara = _nspara;
  spara_min = _spara_min / lbox;
  spara_max = _spara_max / lbox;
  dspara = (spara_max - spara_min) / (double)(nspara);
  sparacen.assign(nspara, 0.0);
  for(int ir = 0; ir < nspara; ir++) sparacen[ir] = spara_min + dspara * (ir + 0.5);

  if(spara_min < -1e-10) half_angle = false;
  else half_angle = true;
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
int correlation::get_ir_from_dr(T dx, T dy, T dz)
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

  const T dr2 = dx * dx + dy * dy + dz * dz;
  int ir = get_r2_index(dr2);
  return ir;
}

template <typename T>
int correlation::get_imu_from_dr(T dx, T dy, T dz)
{
  dx -= std::nearbyint(dx);
  dy -= std::nearbyint(dy);
  dz -= std::nearbyint(dz);

  const T dr2 = dx * dx + dy * dy + dz * dz;
  const T dlos[3] = {dx, dy, dz};
  double mu = dlos[los] / std::sqrt(dr2);
  if(half_angle) mu = std::abs(mu);
  int imu = (int)(std::floor((mu - mumin) / dmu));
  return (imu >= 0 && imu < nmu) ? imu : -1;
}

template <typename T>
T correlation::get_ismu_from_dr(T dx, T dy, T dz, int &ir, int &imu)
{
  dx -= std::nearbyint(dx);
  dy -= std::nearbyint(dy);
  dz -= std::nearbyint(dz);

  const T dr2 = dx * dx + dy * dy + dz * dz;
  ir = get_r2_index(dr2);

  const T dlos[3] = {dx, dy, dz};
  T mu = dlos[los] / std::sqrt(dr2);
  if(half_angle) mu = std::abs(mu);

  imu = (int)(std::floor((mu - mumin) / dmu));
  return mu;
}

template <typename T>
T correlation::get_ispsp_from_dr(T dx, T dy, T dz, int &iperp, int &ipara)
{
  dx -= std::nearbyint(dx);
  dy -= std::nearbyint(dy);
  dz -= std::nearbyint(dz);

  const T dr2 = dx * dx + dy * dy + dz * dz;

  const T dlos[3] = {dx, dy, dz};
  const T dr = std::sqrt(dr2);
  T spara = dlos[los];                            //  r * mu;
  const T sperp = std::sqrt(dr2 - spara * spara); //  sqrt(r^2 - spara^2);

  if(half_angle) spara = std::abs(spara);

  iperp = (int)(std::floor((sperp - sperp_min) / dsperp));
  ipara = (int)(std::floor((spara - spara_min) / dspara));
  T mu = spara / dr;
  return mu;
}

template <typename T>
int correlation::get_cell_index(T pos, int nc)
{
  int idx = static_cast<int>(pos * nc);
  if(idx < 0) idx += nc;
  if(idx >= nc) idx -= nc;
  return idx;
}

void correlation::calc_ideal_pair_r(uint64_t npair)
{
  auto nxi = dd_pair.size();
  // double V_box = lbox * lbox * lbox;
  double V_box = 1.0;

#pragma omp parallel for
  for(int ir = 0; ir < nr; ir++) {

    double r_low, r_high;
    if(log_scale) {
      r_low = rmin * std::exp(ir * logratio);
      r_high = rmin * std::exp((ir + 1) * logratio);
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
  double V_box = 1.0;
  double mu_range = mumax - mumin;
  const double mu_factor = half_angle ? (mu_range) : (0.5 * mu_range);

#pragma omp parallel for
  for(int ir = 0; ir < nr; ir++) {

    double r_low, r_high;
    if(log_scale) {
      r_low = rmin * std::exp(ir * logratio);
      r_high = rmin * std::exp((ir + 1) * logratio);
    } else {
      r_low = rmin + lin_dr * ir;
      r_high = rmin + lin_dr * (ir + 1);
    }

    auto shell_volume = (4.0 / 3.0) * M_PI;
    shell_volume *= (r_high * r_high * r_high - r_low * r_low * r_low);

    auto norm = npair * shell_volume / V_box;
    norm *= mu_factor;

    for(int imu = 0; imu < nmu; imu++) {
      rr_pair[ir * nmu + imu] = norm / static_cast<double>(nmu);
    }
  }
}

void correlation::calc_ideal_spsp_pair(uint64_t npair)
{
  auto nxi = dd_pair.size();
  double V_box = 1.0;
  const double fold = half_angle ? 2.0 : 1.0;

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
      rr_pair[is * nspara + ip] = norm * fold;
    }
  }
}

void correlation::calc_ideal_pair(uint64_t npair)
{
  if(mode == Mode::R) calc_ideal_pair_r(npair);
  else if(mode == Mode::SMU) calc_ideal_smu_pair(npair);
  else if(mode == Mode::SPSP) calc_ideal_spsp_pair(npair);
}

void correlation::calc_xi_from_pair(uint64_t ngrp)
{
  auto nxi = dd_pair.size();
  auto nrand = ngrp * nrand_factor;
  auto f = (double)nrand / (double)ngrp;
  auto f2 = f * f;

  if(est == Estimator::IDEAL) {
    // set ideal count to rr_pair
    uint64_t npair = ngrp * (ngrp - 1);
    if(symmetry) npair /= 2;
    calc_ideal_pair(npair);

    for(int i = 0; i < nxi; i++) {
      if(rr_pair[i] != 0.0) {
        xi[i] = dd_pair[i] / rr_pair[i] - 1.0;
      }
    }

  } else if(est == Estimator::RR) {
    for(int i = 0; i < nxi; i++) {
      if(rr_pair[i] != 0.0) {
        xi[i] = f2 * dd_pair[i] / rr_pair[i] - 1.0;
      }
    }

  } else if(est == Estimator::LS) {
    for(int i = 0; i < nxi; i++) {
      if(rr_pair[i] != 0.0) {
        xi[i] = (f2 * dd_pair[i] - 2.0 * f * dr_pair[i] + rr_pair[i]) / rr_pair[i];
      }
    }
  }
}

void correlation::calc_xi_from_pair(uint64_t ngrp1, uint64_t ngrp2)
{
  auto nxi = dd_pair.size();
  auto nrand1 = ngrp1 * nrand_factor;
  auto nrand2 = ngrp2 * nrand_factor;
  auto f1 = (double)nrand1 / (double)ngrp1;
  auto f2 = (double)nrand2 / (double)ngrp2;
  auto f12 = f1 * f2;

  if(est == Estimator::IDEAL) {
    // set ideal count to rr_pair
    uint64_t npair = ngrp1 * ngrp2;
    calc_ideal_pair(npair);

    for(int i = 0; i < nxi; i++) {
      if(rr_pair[i] != 0.0) {
        xi[i] = dd_pair[i] / rr_pair[i] - 1.0;
      }
    }

  } else if(est == Estimator::RR) {
    for(int i = 0; i < nxi; i++) {
      if(rr_pair[i] != 0.0) {
        xi[i] = f12 * dd_pair[i] / rr_pair[i] - 1.0;
      }
    }

  } else if(est == Estimator::LS) {
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
  calc_xi_impl(grp);
}

template <typename T>
void correlation::calc_xi(T &grp1, T &grp2)
{
  if(grp1.data() == grp2.data()) {
    calc_xi(grp1);
    return;
  }

  symmetry = false;
  calc_xi_impl(grp1, grp2);
}

template <typename T>
void correlation::calc_xi_ifft(T &mesh, T &weight)
{
  calc_xi_ifft_impl(mesh, weight);
}

template <typename T>
void correlation::calc_xi_ifft(T &mesh1, T &mesh2, T &weight)
{
  if(mesh1.data() == mesh2.data()) {
    calc_xi_ifft(mesh1, weight);
    return;
  }

  symmetry = false;
  calc_xi_ifft_impl(mesh1, mesh2, weight);
}

template <typename G, typename C>
std::vector<double> correlation::calc_pair(const double w, G &grp, C &cell_idx, int ncx, int ncy, int ncz,
                                           const std::string &label)
{
  const int nc3 = ncx * ncy * ncz;
  int nthread = omp_get_num_threads();
  int ithread = omp_get_thread_num();
  uint64_t progress = 0;
  uint64_t progress_thread = nc3 / nthread;
  uint64_t progress_div = 1 + progress_thread / 100;

  std::vector<double> thr_pair(nr, 0.0);

#pragma omp for collapse(3) schedule(auto)
  for(int ix = 0; ix < ncx; ix++) {
    for(int iy = 0; iy < ncy; iy++) {
      for(int iz = 0; iz < ncz; iz++) {

        const int cell_id = iz + ncz * (iy + ncy * ix);
        const int a_begin = cell_idx[cell_id];
        const int a_end = cell_idx[cell_id + 1];

        for(int jx = -ndiv_1d; jx <= ndiv_1d; jx++) {
          for(int jy = -ndiv_1d; jy <= ndiv_1d; jy++) {
            for(int jz = -ndiv_1d; jz <= ndiv_1d; jz++) {

              const int nix = ((ix + jx) + ncx) % ncx;
              const int niy = ((iy + jy) + ncy) % ncy;
              const int niz = ((iz + jz) + ncz) % ncz;

              const int ncell_id = niz + ncz * (niy + ncy * nix);
              const int b_begin = cell_idx[ncell_id];
              const int b_end = cell_idx[ncell_id + 1];

              if(symmetry) {
                if(ncell_id < cell_id) continue;
                if(ncell_id == cell_id) {
                  // same cell: only i<j
                  for(int ii = a_begin; ii < a_end; ii++) {
                    // jj is ii+1 to a_end
                    for(int jj = ii + 1; jj < a_end; jj++) {
                      double dx = grp[jj].xpos - grp[ii].xpos;
                      double dy = grp[jj].ypos - grp[ii].ypos;
                      double dz = grp[jj].zpos - grp[ii].zpos;

                      const int ir = get_ir_from_dr(dx, dy, dz);
                      if(ir >= 0 && ir < nr) thr_pair[ir] += w;
                    }
                  }
                  continue;
                }
              }

              for(int ii = a_begin; ii < a_end; ii++) {
                for(int jj = b_begin; jj < b_end; jj++) {
                  double dx = grp[jj].xpos - grp[ii].xpos;
                  double dy = grp[jj].ypos - grp[ii].ypos;
                  double dz = grp[jj].zpos - grp[ii].zpos;
                  const int ir = get_ir_from_dr(dx, dy, dz);
                  if((unsigned)ir < (unsigned)nr) thr_pair[ir] += w;
                }
              }
            }
          }
        } // jx,jy,jz

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
std::vector<double> correlation::calc_pair(const double w, G &grp1, G &grp2, C &cell_idx1, C &cell_idx2, int ncx,
                                           int ncy, int ncz, const std::string &label)
{
  // symmetry は常に false（相互相関）
  const int nc3 = ncx * ncy * ncz;
  const int nthread = omp_get_num_threads();
  const int ithread = omp_get_thread_num();
  uint64_t progress = 0;
  const uint64_t progress_thread = nc3 / nthread;
  const uint64_t progress_div = 1 + progress_thread / 100;

  std::vector<double> thr_pair(nr, 0.0);

#pragma omp for collapse(3) schedule(auto)
  for(int ix = 0; ix < ncx; ix++) {
    for(int iy = 0; iy < ncy; iy++) {
      for(int iz = 0; iz < ncz; iz++) {

        const int cell_id = iz + ncz * (iy + ncy * ix);
        const int a_begin = cell_idx1[cell_id];
        const int a_end = cell_idx1[cell_id + 1];

        for(int jx = -ndiv_1d; jx <= ndiv_1d; jx++) {
          for(int jy = -ndiv_1d; jy <= ndiv_1d; jy++) {
            for(int jz = -ndiv_1d; jz <= ndiv_1d; jz++) {

              const int nix = ((ix + jx) + ncx) % ncx;
              const int niy = ((iy + jy) + ncy) % ncy;
              const int niz = ((iz + jz) + ncz) % ncz;

              const int ncell_id = niz + ncz * (niy + ncy * nix);
              const int b_begin = cell_idx2[ncell_id];
              const int b_end = cell_idx2[ncell_id + 1];

              for(int ii = a_begin; ii < a_end; ii++) {
                for(int jj = b_begin; jj < b_end; jj++) {
                  double dx = grp2[jj].xpos - grp1[ii].xpos;
                  double dy = grp2[jj].ypos - grp1[ii].ypos;
                  double dz = grp2[jj].zpos - grp1[ii].zpos;

                  const int ir = get_ir_from_dr(dx, dy, dz);
                  if(ir >= 0 && ir < nr) thr_pair[ir] += w;
                }
              }
            }
          }
        }

        if(ithread == 0) {
          progress++;
          if(progress % progress_div == 0) {
            std::cerr << "\r\033[2K " << label << " : " << (double)100.0 * progress / (double)progress_thread << " [%]";
          }
        }
      }
    }
  }

  if(ithread == 0) std::cerr << std::endl;
  return thr_pair;
}

template <typename G, typename C>
std::vector<double> correlation::calc_pair_smu(const double w, G &grp, C &cell_idx, int ncx, int ncy, int ncz,
                                               const std::string &label)
{
  const int nc3 = ncx * ncy * ncz;
  const int nthread = omp_get_num_threads();
  const int ithread = omp_get_thread_num();
  uint64_t progress = 0;
  const uint64_t progress_thread = nc3 / nthread;
  const uint64_t progress_div = 1 + progress_thread / 100;

  std::vector<double> thr_pair(nr * nmu, 0.0);

#pragma omp for collapse(3) schedule(auto)
  for(int ix = 0; ix < ncx; ix++) {
    for(int iy = 0; iy < ncy; iy++) {
      for(int iz = 0; iz < ncz; iz++) {

        const int cell_id = iz + ncz * (iy + ncy * ix);
        const int a_begin = cell_idx[cell_id];
        const int a_end = cell_idx[cell_id + 1];

        for(int jx = -ndiv_1d; jx <= ndiv_1d; jx++) {
          for(int jy = -ndiv_1d; jy <= ndiv_1d; jy++) {
            for(int jz = -ndiv_1d; jz <= ndiv_1d; jz++) {

              const int nix = ((ix + jx) + ncx) % ncx;
              const int niy = ((iy + jy) + ncy) % ncy;
              const int niz = ((iz + jz) + ncz) % ncz;

              const int ncell_id = niz + ncz * (niy + ncy * nix);
              const int b_begin = cell_idx[ncell_id];
              const int b_end = cell_idx[ncell_id + 1];

              if(symmetry) {
                if(ncell_id < cell_id) continue;
                if(ncell_id == cell_id) {
                  for(int ii = a_begin; ii < a_end; ii++) {
                    for(int jj = ii + 1; jj < a_end; jj++) {
                      double dx = grp[jj].xpos - grp[ii].xpos;
                      double dy = grp[jj].ypos - grp[ii].ypos;
                      double dz = grp[jj].zpos - grp[ii].zpos;
                      int ir, imu;
                      auto mu = get_ismu_from_dr(dx, dy, dz, ir, imu);

                      if(ir >= 0 && ir < nr) {
                        if(imu >= 0 && imu < nmu) {
                          thr_pair[imu + nmu * ir] += w;
                        }
                      } // ir, imu
                    }
                  }
                  continue;
                }
              } // symmetry

              for(int ii = a_begin; ii < a_end; ii++) {
                for(int jj = b_begin; jj < b_end; jj++) {
                  double dx = grp[jj].xpos - grp[ii].xpos;
                  double dy = grp[jj].ypos - grp[ii].ypos;
                  double dz = grp[jj].zpos - grp[ii].zpos;
                  int ir, imu;
                  auto mu = get_ismu_from_dr(dx, dy, dz, ir, imu);

                  if(ir >= 0 && ir < nr) {
                    if(imu >= 0 && imu < nmu) {
                      thr_pair[imu + nmu * ir] += w;
                    }
                  } // ir, imu
                }
              }
            }
          }
        } // jx,jy,jz

        if(ithread == 0) {
          progress++;
          if(progress % progress_div == 0) {
            std::cerr << "\r\033[2K " << label << " (s,mu) : " << (double)100.0 * progress / (double)progress_thread
                      << " [%]";
          }
        }
      }
    }
  } // ix,iy,iz

  if(ithread == 0) std::cerr << std::endl;
  return thr_pair;
}

template <typename G, typename C>
std::vector<double> correlation::calc_pair_smu(const double w, G &grp1, G &grp2, C &cell_idx1, C &cell_idx2, int ncx,
                                               int ncy, int ncz, const std::string &label)
{
  const int nc3 = ncx * ncy * ncz;
  const int nthread = omp_get_num_threads();
  const int ithread = omp_get_thread_num();
  uint64_t progress = 0;
  const uint64_t progress_thread = nc3 / nthread;
  const uint64_t progress_div = 1 + progress_thread / 100;

  std::vector<double> thr_pair(nr * nmu, 0.0);

#pragma omp for collapse(3) schedule(auto)
  for(int ix = 0; ix < ncx; ix++) {
    for(int iy = 0; iy < ncy; iy++) {
      for(int iz = 0; iz < ncz; iz++) {

        const int cell_id = iz + ncz * (iy + ncy * ix);
        const int a_begin = cell_idx1[cell_id];
        const int a_end = cell_idx1[cell_id + 1];

        for(int jx = -ndiv_1d; jx <= ndiv_1d; jx++) {
          for(int jy = -ndiv_1d; jy <= ndiv_1d; jy++) {
            for(int jz = -ndiv_1d; jz <= ndiv_1d; jz++) {

              const int nix = ((ix + jx) + ncx) % ncx;
              const int niy = ((iy + jy) + ncy) % ncy;
              const int niz = ((iz + jz) + ncz) % ncz;

              const int ncell_id = niz + ncz * (niy + ncy * nix);
              const int b_begin = cell_idx2[ncell_id];
              const int b_end = cell_idx2[ncell_id + 1];

              for(int ii = a_begin; ii < a_end; ii++) {
                for(int jj = b_begin; jj < b_end; jj++) {
                  double dx = grp2[jj].xpos - grp1[ii].xpos;
                  double dy = grp2[jj].ypos - grp1[ii].ypos;
                  double dz = grp2[jj].zpos - grp1[ii].zpos;
                  int ir, imu;
                  auto mu = get_ismu_from_dr(dx, dy, dz, ir, imu);

                  if(ir >= 0 && ir < nr) {
                    if(imu >= 0 && imu < nmu) {
                      thr_pair[imu + nmu * ir] += w;
                    }
                  } // ir, imu
                }
              }
            }
          }
        } // jx,jy,jz

        if(ithread == 0) {
          progress++;
          if(progress % progress_div == 0) {
            std::cerr << "\r\033[2K " << label << " (s,mu) : " << (double)100.0 * progress / (double)progress_thread
                      << " [%]";
          }
        }
      }
    }
  } // ix,iy,iz

  if(ithread == 0) std::cerr << std::endl;
  return thr_pair;
}
template <typename G, typename C>
std::vector<double> correlation::calc_pair_spsp(const double w, G &grp, C &cell_idx, int ncx, int ncy, int ncz,
                                                const std::string &label)
{
  const int nc3 = ncx * ncy * ncz;
  const int nthread = omp_get_num_threads();
  const int ithread = omp_get_thread_num();
  uint64_t progress = 0;
  const uint64_t progress_thread = nc3 / nthread;
  const uint64_t progress_div = 1 + progress_thread / 100;

  std::vector<double> thr_pair(nsperp * nspara, 0.0);

#pragma omp for collapse(3) schedule(auto)
  for(int ix = 0; ix < ncx; ix++) {
    for(int iy = 0; iy < ncy; iy++) {
      for(int iz = 0; iz < ncz; iz++) {

        const int cell_id = iz + ncz * (iy + ncy * ix);
        const int a_begin = cell_idx[cell_id];
        const int a_end = cell_idx[cell_id + 1];

        for(int jx = -ndiv_1d; jx <= ndiv_1d; jx++) {
          for(int jy = -ndiv_1d; jy <= ndiv_1d; jy++) {
            for(int jz = -ndiv_1d; jz <= ndiv_1d; jz++) {

              const int nix = ((ix + jx) + ncx) % ncx;
              const int niy = ((iy + jy) + ncy) % ncy;
              const int niz = ((iz + jz) + ncz) % ncz;

              const int ncell_id = niz + ncz * (niy + ncy * nix);
              const int b_begin = cell_idx[ncell_id];
              const int b_end = cell_idx[ncell_id + 1];

              if(symmetry) {
                if(ncell_id < cell_id) continue;
                if(ncell_id == cell_id) {
                  for(int ii = a_begin; ii < a_end; ii++) {
                    for(int jj = ii + 1; jj < a_end; jj++) {
                      double dx = grp[jj].xpos - grp[ii].xpos;
                      double dy = grp[jj].ypos - grp[ii].ypos;
                      double dz = grp[jj].zpos - grp[ii].zpos;

                      int iperp, ipara;
                      auto mu = get_ispsp_from_dr(dx, dy, dz, iperp, ipara);
                      if(iperp >= 0 && iperp < nsperp) {
                        if(ipara >= 0 && ipara < nspara) {
                          thr_pair[ipara + nspara * iperp] += w;
                        }
                      } // iperp, ipara
                    }
                  } // ii,jj
                  continue;
                }
              }

              for(int ii = a_begin; ii < a_end; ii++) {
                for(int jj = b_begin; jj < b_end; jj++) {
                  double dx = grp[jj].xpos - grp[ii].xpos;
                  double dy = grp[jj].ypos - grp[ii].ypos;
                  double dz = grp[jj].zpos - grp[ii].zpos;

                  int iperp, ipara;
                  auto mu = get_ispsp_from_dr(dx, dy, dz, iperp, ipara);
                  if(iperp >= 0 && iperp < nsperp) {
                    if(ipara >= 0 && ipara < nspara) {
                      thr_pair[ipara + nspara * iperp] += w;
                    }
                  } // iperp, ipara
                }
              } // ii,jj
            }
          }
        } // jx,jy,jz

        if(ithread == 0) {
          progress++;
          if(progress % progress_div == 0) {
            std::cerr << "\r\033[2K " << label
                      << " (sperp,spara) : " << (double)100.0 * progress / (double)progress_thread << " [%]";
          }
        }
      }
    }
  } // ix,iy,iz

  if(ithread == 0) std::cerr << std::endl;
  return thr_pair;
}

template <typename G, typename C>
std::vector<double> correlation::calc_pair_spsp(const double w, G &grp1, G &grp2, C &cell_idx1, C &cell_idx2, int ncx,
                                                int ncy, int ncz, const std::string &label)
{
  const int nc3 = ncx * ncy * ncz;
  const int nthread = omp_get_num_threads();
  const int ithread = omp_get_thread_num();
  uint64_t progress = 0;
  const uint64_t progress_thread = nc3 / nthread;
  const uint64_t progress_div = 1 + progress_thread / 100;

  std::vector<double> thr_pair(nsperp * nspara, 0.0);

#pragma omp for collapse(3) schedule(auto)
  for(int ix = 0; ix < ncx; ix++) {
    for(int iy = 0; iy < ncy; iy++) {
      for(int iz = 0; iz < ncz; iz++) {

        const int cell_id = iz + ncz * (iy + ncy * ix);
        const int a_begin = cell_idx1[cell_id];
        const int a_end = cell_idx1[cell_id + 1];

        for(int jx = -ndiv_1d; jx <= ndiv_1d; jx++) {
          for(int jy = -ndiv_1d; jy <= ndiv_1d; jy++) {
            for(int jz = -ndiv_1d; jz <= ndiv_1d; jz++) {

              const int nix = ((ix + jx) + ncx) % ncx;
              const int niy = ((iy + jy) + ncy) % ncy;
              const int niz = ((iz + jz) + ncz) % ncz;

              const int ncell_id = niz + ncz * (niy + ncy * nix);
              const int b_begin = cell_idx2[ncell_id];
              const int b_end = cell_idx2[ncell_id + 1];

              for(int ii = a_begin; ii < a_end; ii++) {
                for(int jj = b_begin; jj < b_end; jj++) {
                  double dx = grp2[jj].xpos - grp1[ii].xpos;
                  double dy = grp2[jj].ypos - grp1[ii].ypos;
                  double dz = grp2[jj].zpos - grp1[ii].zpos;

                  int iperp, ipara;
                  auto mu = get_ispsp_from_dr(dx, dy, dz, iperp, ipara);
                  if(iperp >= 0 && iperp < nsperp) {
                    if(ipara >= 0 && ipara < nspara) {
                      thr_pair[ipara + nspara * iperp] += w;
                    }
                  } // iperp, ipara
                }
              }
            }
          }
        }

        if(ithread == 0) {
          progress++;
          if(progress % progress_div == 0) {
            std::cerr << "\r\033[2K " << label
                      << " (sperp,spara) : " << (double)100.0 * progress / (double)progress_thread << " [%]";
          }
        }
      }
    }
  }

  if(ithread == 0) std::cerr << std::endl;
  return thr_pair;
}

template <typename G, typename C>
std::vector<double> correlation::calc_cor_pair(const double w, G &grp, C &cell_idx, int ncx, int ncy, int ncz,
                                               const std::string &label)
{
  if(mode == Mode::R) return calc_pair(w, grp, cell_idx, ncx, ncy, ncz, label);
  if(mode == Mode::SMU) return calc_pair_smu(w, grp, cell_idx, ncx, ncy, ncz, label);
  if(mode == Mode::SPSP) return calc_pair_spsp(w, grp, cell_idx, ncx, ncy, ncz, label);
  assert(!"invalid mode");
  return {};
}

template <typename G, typename C>
std::vector<double> correlation::calc_cor_pair(const double w, G &grp1, G &grp2, C &cell_idx1, C &cell_idx2, int ncx,
                                               int ncy, int ncz, const std::string &label)
{
  if(mode == Mode::R) return calc_pair(w, grp1, grp2, cell_idx1, cell_idx2, ncx, ncy, ncz, label);
  if(mode == Mode::SMU) return calc_pair_smu(w, grp1, grp2, cell_idx1, cell_idx2, ncx, ncy, ncz, label);
  if(mode == Mode::SPSP) return calc_pair_spsp(w, grp1, grp2, cell_idx1, cell_idx2, ncx, ncy, ncz, label);
  assert(!"invalid mode");
  return {};
}

template <typename G, typename C>
G correlation::sort_group_by_cell(const G &grp, const int ncx, const int ncy, const int ncz, C &cell_idx)
{
  const int nc3 = ncx * ncy * ncz;
  const uint64_t ngrp = grp.size();

  std::vector<int> count(nc3, 0);

#pragma omp parallel for
  for(uint64_t i = 0; i < ngrp; i++) {
    const int ix = get_cell_index(grp[i].xpos, ncx);
    const int iy = get_cell_index(grp[i].ypos, ncy);
    const int iz = get_cell_index(grp[i].zpos, ncz);
    const int cid = iz + ncz * (iy + ncy * ix);
#pragma omp atomic
    count[cid]++;
  }

  cell_idx.assign(nc3 + 1, 0);
  for(int cid = 0; cid < nc3; cid++) cell_idx[cid + 1] = cell_idx[cid] + count[cid];

  G sorted_grp(ngrp);
  std::vector<int> cursor = cell_idx; // Copy of cell_idx to avoid destroying

  for(uint64_t i = 0; i < ngrp; i++) {
    const int ix = get_cell_index(grp[i].xpos, ncx);
    const int iy = get_cell_index(grp[i].ypos, ncy);
    const int iz = get_cell_index(grp[i].zpos, ncz);
    const int cid = iz + ncz * (iy + ncy * ix);
    sorted_grp[cursor[cid]++] = grp[i];
  }

  return sorted_grp;
}

template <typename T>
void correlation::calc_xi_impl(T &grp)
{
  bool use_random = (est != Estimator::IDEAL);

  uint64_t ngrp = grp.size();
  uint64_t nrand = 0;

  uint64_t nn = nr;
  if(mode == Mode::SMU) nn = nr * nmu;
  if(mode == Mode::SPSP) nn = nsperp * nspara;

  dd_pair.assign(nn, 0.0);
  rr_pair.assign(nn, 0.0);
  if(est == Estimator::LS) dr_pair.assign(nn, 0.0);
  xi.assign(nn, 0.0);

  const int ncx = ndiv_1d * std::max(static_cast<int>(1.0 / rmax), 1);
  const int ncy = ndiv_1d * std::max(static_cast<int>(1.0 / rmax), 1);
  const int ncz = ndiv_1d * std::max(static_cast<int>(1.0 / rmax), 1);
  const int nc3 = ncx * ncy * ncz;

  std::vector<int> cell_idx;
  grp = sort_group_by_cell(grp, ncx, ncy, ncz, cell_idx);

  std::vector<int> cell_idx_rand;
  std::vector<group> randoms;

  if(use_random) {
    nrand = ngrp * nrand_factor;
    cell_idx_rand.resize(nc3);
    groupcatalog g;
    randoms = g.set_random_group(nrand);
    randoms = sort_group_by_cell(randoms, ncx, ncy, ncz, cell_idx_rand);
  }

#pragma omp parallel
  {
    int nthread = omp_get_num_threads();
    int ithread = omp_get_thread_num();
    if(ithread == 0) std::cerr << "# nc^3 = " << nc3 << " divided into " << nthread << " threads." << std::endl;

    auto thr_dd_pair = calc_cor_pair(1.0, grp, cell_idx, ncx, ncy, ncz, "DD");
    decltype(thr_dd_pair) thr_rr_pair, thr_dr_pair;

    if(use_random) {
      thr_rr_pair = calc_cor_pair(1.0, randoms, cell_idx_rand, ncx, ncy, ncz, "RR");
      if(est == Estimator::LS)
        thr_dr_pair = calc_cor_pair(0.5, grp, randoms, cell_idx, cell_idx_rand, ncx, ncy, ncz, "DR");
    }

#pragma omp critical
    {
      for(size_t i = 0; i < nn; i++) dd_pair[i] += thr_dd_pair[i];
      if(use_random) {
        for(size_t i = 0; i < nn; i++) rr_pair[i] += thr_rr_pair[i];
        if(est == Estimator::LS)
          for(size_t i = 0; i < nn; i++) dr_pair[i] += thr_dr_pair[i];
      }
    } // omp critical
  } // omp parallel

  calc_xi_from_pair(ngrp); // calc xi
}

template <typename T>
void correlation::calc_xi_impl(T &grp1, T &grp2)
{
  bool use_random = (est != Estimator::IDEAL);

  uint64_t ngrp1 = grp1.size();
  uint64_t ngrp2 = grp2.size();
  uint64_t nrand1 = 0;
  uint64_t nrand2 = 0;

  uint64_t nn = nr;
  if(mode == Mode::SMU) nn = nr * nmu;
  if(mode == Mode::SPSP) nn = nsperp * nspara;

  dd_pair.assign(nn, 0.0);
  rr_pair.assign(nn, 0.0);
  if(est == Estimator::LS) {
    dr_pair.assign(nn, 0.0);
    dr2_pair.assign(nn, 0.0);
  }
  xi.assign(nn, 0.0);

  /* Here only the global box size */
  const int ncx = ndiv_1d * std::max(static_cast<int>(1.0 / rmax), 1);
  const int ncy = ndiv_1d * std::max(static_cast<int>(1.0 / rmax), 1);
  const int ncz = ndiv_1d * std::max(static_cast<int>(1.0 / rmax), 1);
  const int nc3 = ncx * ncy * ncz;

  std::vector<int> cell_idx1, cell_idx2;
  grp1 = sort_group_by_cell(grp1, ncx, ncy, ncz, cell_idx1);
  grp2 = sort_group_by_cell(grp2, ncx, ncy, ncz, cell_idx2);

  std::vector<int> cell_idx_rand1, cell_idx_rand2;
  std::vector<group> randoms1, randoms2;

  if(use_random) {
    groupcatalog g;
    nrand1 = ngrp1 * nrand_factor;
    randoms1 = g.set_random_group(nrand1, 1);
    randoms1 = sort_group_by_cell(randoms1, ncx, ncy, ncz, cell_idx_rand1);

    nrand2 = ngrp2 * nrand_factor;
    randoms2 = g.set_random_group(nrand2, 2);
    randoms2 = sort_group_by_cell(randoms2, ncx, ncy, ncz, cell_idx_rand2);
  }

#pragma omp parallel
  {

    int nthread = omp_get_num_threads();
    int ithread = omp_get_thread_num();
    uint64_t progress = 0;
    uint64_t progress_thread = nc3 / nthread;
    uint64_t progress_div = 1 + progress_thread / 100;

    if(ithread == 0)
      std::cerr << "# nc^3, ngrp_thread = " << nc3 << ", " << progress_thread << " in " << nthread << " threads."
                << std::endl;

    auto thr_dd_pair = calc_cor_pair(1.0, grp1, grp2, cell_idx1, cell_idx2, ncx, ncy, ncz, "D1D2");
    decltype(thr_dd_pair) thr_rr_pair, thr_dr_pair, thr_dr2_pair;

    if(use_random) {
      thr_rr_pair = calc_cor_pair(1.0, randoms1, randoms2, cell_idx_rand1, cell_idx_rand2, ncx, ncy, ncz, "R1R2");
      if(est == Estimator::LS) {
        thr_dr_pair = calc_cor_pair(1.0, grp1, randoms2, cell_idx1, cell_idx_rand2, ncx, ncy, ncz, "D1R2");
        thr_dr2_pair = calc_cor_pair(1.0, grp2, randoms1, cell_idx2, cell_idx_rand1, ncx, ncy, ncz, "D2R1");
      }
    }

#pragma omp critical
    {
      for(size_t i = 0; i < nn; i++) dd_pair[i] += thr_dd_pair[i];
      if(use_random) {
        for(size_t i = 0; i < nn; i++) rr_pair[i] += thr_rr_pair[i];
        if(est == Estimator::LS) {
          for(size_t i = 0; i < nn; i++) dr_pair[i] += thr_dr_pair[i];
          for(size_t i = 0; i < nn; i++) dr2_pair[i] += thr_dr2_pair[i];
        }
      }
    } // omp critical
  } // omp parallel

  calc_xi_from_pair(ngrp1, ngrp2); // calc xi
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
T correlation::xismu_to_xir(const T &xi2d, const T &rr2d) const
{
  T xi1D(nr, 0.0);

#pragma omp parallel for
  for(int ir = 0; ir < nr; ++ir) {
    double num = 0.0, den = 0.0;
    for(int imu = 0; imu < nmu; ++imu) {
      const int idx = imu + nmu * ir;
      const double w = rr2d[idx];
      num += w * xi2d[idx];
      den += w;
    }
    xi1D[ir] = (den > 0.0) ? (num / den) : 0.0;
  }
  return xi1D;
}

template <typename T>
T correlation::xispsp_to_wpr(const T &xi2d, const T &rr2d) const
{
  double dpi_int = (spara_max - spara_min) * lbox;
  dpi_int = half_angle ? 2.0 * dpi_int : dpi_int;

  T wp(nsperp, 0.0);

#pragma omp parallel for
  for(int ip = 0; ip < nsperp; ++ip) {
    double num = 0.0;
    double den = 0.0;
    for(int iz = 0; iz < nspara; ++iz) {
      const int idx = iz + nspara * ip;

      const double w = rr2d[idx];
      num += w * xi2d[idx];
      den += w;
    }
    wp[ip] = (den > 0.0) ? dpi_int * (num / den) : 0.0;
  }

  return wp;
}

void correlation::output_xi(std::string filename)
{
  std::ofstream fout(filename);

  if(dr_pair.size() > 0 && dr2_pair.size() > 0) {
    fout << "# r[Mpc/h] xi D1D2 D1R2 D2R1 R1R2" << std::endl;
    for(int ir = 0; ir < nr; ir++) {
      double rad = rcen[ir] * lbox;
      fout << std::scientific << std::setprecision(10) << rad << " " << xi[ir] << " " << dd_pair[ir] << " "
           << dr_pair[ir] << " " << dr2_pair[ir] << " " << rr_pair[ir] << "\n";
    }
  } else if(dr_pair.size() > 0) {
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

  fout.flush();
  fout.close();
  std::cout << "output to " << filename << std::endl;
}

template <typename T>
void correlation::output_xi(std::string filename, T &weight)
{
  std::ofstream fout(filename);

  fout << "# r[Mpc/h] xi weight" << std::endl;
  for(int ir = 0; ir < nr; ir++) {
    double rad = rcen[ir] * lbox;
    fout << std::scientific << std::setprecision(10) << rad << " " << xi[ir] << " " << weight[ir] << "\n";
  }

  fout.flush();
  fout.close();
  std::cout << "output to " << filename << std::endl;
}

void correlation::output_xi2D_smu(std::string filename)
{
  std::ofstream fout(filename);

  if(est == Estimator::LS) {
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
    fout << "# r[Mpc/h] mu xi DD RR" << std::endl;
    for(int ir = 0; ir < nr; ir++) {
      double rad = rcen[ir] * lbox;
      for(int imu = 0; imu < nmu; imu++) {
        auto idx = imu + nmu * ir;
        fout << std::scientific << std::setprecision(10) << rad << " " << mucen[imu] << " " << xi[idx] << " "
             << dd_pair[idx] << " " << rr_pair[idx] << "\n";
      } // imu
    } // ir
  }

  fout.flush();
  fout.close();
  std::cout << "output to " << filename << std::endl;
}

void correlation::output_xi2D_spsp(std::string filename)
{
  std::ofstream fout(filename);

  if(est == Estimator::LS) {
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
      fout << "# s_perp[Mpc/h] s_para[Mpc/h] xi DD RR" << std::endl;
      for(int iperp = 0; iperp < nsperp; iperp++) {
        for(int ipara = 0; ipara < nspara; ipara++) {
          auto spara = sparacen[ipara] * lbox;
          auto sperp = sperpcen[iperp] * lbox;
          auto idx = ipara + nspara * iperp;
          fout << std::scientific << std::setprecision(10) << sperp << " " << spara << " " << xi[idx] << " "
               << dd_pair[idx] << " " << rr_pair[idx] << "\n";
        }
      }
    }
  }

  fout.flush();
  fout.close();
  std::cout << "output to " << filename << std::endl;
}

void correlation::output_xi1D_smu(std::string filename)
{
  std::ofstream fout(filename);

  auto xir = xismu_to_xir(xi, rr_pair);
  fout << "# r[Mpc/h] xi(r)" << std::endl;
  for(int ir = 0; ir < nr; ++ir) {
    const double r = rcen[ir] * lbox;
    fout << std::scientific << std::setprecision(10) << r << " " << xir[ir] << "\n";
  }

  fout.flush();
  fout.close();
  std::cout << "output to " << filename << std::endl;
}

void correlation::output_xi1D_spsp(std::string filename)
{
  std::ofstream fout(filename);

  auto wp = xispsp_to_wpr(xi, rr_pair);
  fout << "# r_perp[Mpc/h] w_p(r_perp)" << std::endl;
  for(int ip = 0; ip < nsperp; ++ip) {
    const double rp_phys = sperpcen[ip] * lbox;
    fout << std::scientific << std::setprecision(10) << rp_phys << " " << wp[ip] << "\n";
  }

  fout.flush();
  fout.close();
  std::cout << "output to " << filename << std::endl;
}

void correlation::output_xi2D(std::string filename)
{
  if(mode == Mode::SMU) {
    output_xi2D_smu(add_label_string(filename, "2d"));
    output_xi1D_smu(add_label_string(filename, "1d"));
  } else if(mode == Mode::SPSP) {
    output_xi2D_spsp(add_label_string(filename, "2d"));
    output_xi1D_spsp(add_label_string(filename, "1d"));
  }
}
