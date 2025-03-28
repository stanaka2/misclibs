#pragma once

#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cassert>

#include <fftw3.h>
#include "powerspec.hpp"
#include "util.hpp"

#define c_re(c) ((c)[0])
#define c_im(c) ((c)[1])
#define SQR(a) ((a) * (a))
#define CUBE(a) ((a) * (a) * (a))

struct group {
  float xpos, ypos, zpos;
  float xvel, yvel, zvel;
  float rad, mass;
  float pot;
  int nmem;
};

class correlation : public powerspec
{
public:
  int nr = 100;
  double rmin = 1.5, rmax = 150; // [Mpc/h]
  std::vector<float> rbin, rcen; // bin edge and center

  double rmin2, rmax2;    // r, r^2 base
  double lin_dr, lin_dr2; // r, r^2 base

  /* radius of searching cell : (ndiv_1d+2)^3 */
  const int ndiv_1d = 1;
  // const int ndiv_1d = 2;

  const int jk_block = 8;
  // const int jk_block = 32;

  int nrand, ngrp;
  int jk_dd, nblock;
  double mmin, mmax;

  std::vector<group> grp;
  std::vector<group> rand;

  // for position xi
  // support Landy SD, Szalay AS 1993, Apj
  std::vector<double> dd_pair, dr_pair, rr_pair;                       // size[nr]
  std::vector<std::vector<double>> dd_pair_jk, dr_pair_jk, rr_pair_jk; // size[block][nr]
  std::vector<double> xi, xi_ave, xi_sd, xi_se;                        // size[nr]
  std::vector<std::vector<double>> xi_jk;                              // size[block][nr]

  void set_rbin(double, double, int, double, bool = true);
  void check_rbin();
  template <typename T>
  int get_r_index(T);
  template <typename T>
  int get_r2_index(T);
  template <typename T>
  int get_cell_index(T, int);

  template <typename T>
  void set_halo_pm_group(T &, T &);
  template <typename T>
  void set_halo_pvm_group(T &, T &, T &);
  template <typename T>
  void set_halo_ppm_group(T &, T &, T &);
  void set_random_group();
  void shuffle_halo_data(const int = 10);

  void calc_xi(const int = 1);
  void calc_xi_LS(const int = 0);

  void calc_xi_direct();
  void calc_xi_cell_list();
  void calc_xi_LS_direct();
  void calc_xi_LS_cell_list();

  void calc_xi_jk_LS();
  void resample_jackknife_LS();
  void calc_jk_xi_average();
  void calc_jk_xi_error();

  void output_xi(std::string);
  void output_xi_LS(std::string);
  void output_xi_jk(std::string);
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

  rbin.resize(nr + 1, 0.0);
  rcen.resize(nr, 0.0);

  if(log_scale) {
    if(rmin < 1e-10) rmin = 1e-10;
    ratio = pow(rmax / rmin, 1.0 / (double)(nr));
    logratio = log(ratio);
    logratio2 = 2 * logratio;
    for(int ir = 0; ir < nr + 1; ir++) rbin[ir] = rmin * pow(ratio, ir);
    for(int ir = 0; ir < nr; ir++) rcen[ir] = rmin * pow(ratio, ir + 0.5);

  } else {
    rmin = 0.0;
    lin_dr = (rmax - rmin) / (double)(nr);
    for(int ir = 0; ir < nr + 1; ir++) rbin[ir] = rmin + lin_dr * ir;
    for(int ir = 0; ir < nr; ir++) rcen[ir] = rmin + lin_dr * (ir + 0.5);
  }
}

void correlation::check_rbin()
{
  for(int ir = 0; ir < nr; ir++) std::cerr << ir << " " << rcen[ir] << " " << get_r_index(rcen[ir]) << "\n";
}

template <typename T>
int correlation::get_r_index(T r)
{
  if(r < rmin || r >= rmax) return -1;
  if(log_scale) return floor(log(r / rmin) / logratio);
  else return floor((r - rmin) / lin_dr);
  return -1;
}

template <typename T>
int correlation::get_r2_index(T r2)
{
  if(r2 < rmin2 || r2 >= rmax2) return -1;
  if(log_scale) return floor(log(r2 / rmin2) / logratio2);
  else return floor((std::sqrt(r2) - rmin) / lin_dr);
  return -1;
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
void correlation::set_halo_pm_group(T &pos, T &mvir)
{
  uint64_t nhalo = mvir.size();
  grp.resize(nhalo);
  std::cerr << "# input halo " << nhalo << " ~ " << (int)(pow((double)nhalo, 1.0 / 3.0)) << "^3" << std::endl;

  uint64_t ig = 0;
  for(uint64_t i = 0; i < nhalo; i++) {
    double m = mvir[i];
    if(mmin <= m && m <= mmax) {
      grp[ig].xpos = pos[3 * i + 0] / lbox; // [0,1]
      grp[ig].ypos = pos[3 * i + 1] / lbox; // [0,1]
      grp[ig].zpos = pos[3 * i + 2] / lbox; // [0,1]
      grp[ig].mass = m;
      ig++;
    }
  }
  grp.resize(ig);
  std::cerr << "# selection halo " << ig << " ~ " << (int)(pow((double)ig, 1.0 / 3.0)) << "^3" << std::endl;
}

template <typename T>
void correlation::set_halo_pvm_group(T &pos, T &vel, T &mvir)
{
  uint64_t nhalo = mvir.size();
  grp.resize(nhalo);
  std::cerr << "# input halo " << nhalo << " ~ " << (int)(pow((double)nhalo, 1.0 / 3.0)) << "^3" << std::endl;

  uint64_t ig = 0;
  for(uint64_t i = 0; i < nhalo; i++) {
    double m = mvir[i];
    if(mmin <= m && m <= mmax) {
      grp[ig].xpos = pos[3 * i + 0] / lbox; // [0,1]
      grp[ig].ypos = pos[3 * i + 1] / lbox; // [0,1]
      grp[ig].zpos = pos[3 * i + 2] / lbox; // [0,1]
      grp[ig].xvel = vel[3 * i + 0];        // km/s
      grp[ig].yvel = vel[3 * i + 1];        // km/s
      grp[ig].zvel = vel[3 * i + 2];        // km/s
      grp[ig].mass = m;
      ig++;
    }
  }
  grp.resize(ig);
  std::cerr << "# selection halo " << ig << " ~ " << (int)(pow((double)ig, 1.0 / 3.0)) << "^3" << std::endl;
}

template <typename T>
void correlation::set_halo_ppm_group(T &pos, T &pot, T &mvir)
{
  uint64_t nhalo = mvir.size();
  grp.resize(nhalo);
  std::cerr << "# input halo " << nhalo << " ~ " << (int)(pow((double)nhalo, 1.0 / 3.0)) << "^3" << std::endl;

  uint64_t ig = 0;
  for(uint64_t i = 0; i < nhalo; i++) {
    double m = mvir[i];
    if(mmin <= m && m <= mmax) {
      grp[ig].xpos = pos[3 * i + 0] / lbox; // [0,1]
      grp[ig].ypos = pos[3 * i + 1] / lbox; // [0,1]
      grp[ig].zpos = pos[3 * i + 2] / lbox; // [0,1]
      grp[ig].pot = pot[i];
      grp[ig].mass = m;
      ig++;
    }
  }
  grp.resize(ig);
  std::cerr << "# selection halo " << ig << " ~ " << (int)(pow((double)ig, 1.0 / 3.0)) << "^3" << std::endl;
}

void correlation::set_random_group()
{
  const int seed = 2;
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<> dist(0.0, 1.0);

  rand.resize(nrand);

  for(uint64_t i = 0; i < nrand; i++) {
    rand[i].xpos = dist(rng);
    rand[i].ypos = dist(rng);
    rand[i].zpos = dist(rng);
  }

  std::cerr << "# set random halo pos " << nrand << " ~ " << (int)(pow((double)nrand, 1.0 / 3.0)) << "^3" << std::endl;
}

void correlation::shuffle_halo_data(const int seed)
{
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<uint64_t> dist(0, ngrp - 1);

  for(uint64_t i = 0; i < ngrp; i++) {
    uint64_t irand = dist(rng);
    std::swap(grp[i], grp[irand]);
  }
}

void correlation::calc_xi(const int calc_type)
{
  if(calc_type == 0) calc_xi_direct();
  else if(calc_type == 1) calc_xi_cell_list();
}

void correlation::calc_xi_LS(const int calc_type)
{
  if(calc_type == 0) calc_xi_LS_direct();
  else if(calc_type == 1) calc_xi_LS_cell_list();
}

void correlation::calc_xi_direct()
{
  ngrp = grp.size();
  nrand = ngrp;

  dd_pair.resize(nr);
  xi.resize(nr);

  for(int i = 0; i < nr; i++) {
    dd_pair[i] = 0.0;
    xi[i] = 0.0;
  }

#pragma omp parallel
  {
    std::vector<double> thr_dd_pair(nr, 0.0);

    int nthread = omp_get_num_threads();
    int ithread = omp_get_thread_num();
    uint64_t progress = 0;
    uint64_t progress_thread = ngrp / nthread;
    uint64_t progress_div = 1 + progress_thread / 200;

    if(ithread == 0)
      std::cerr << "# ngrp, ngrp_thread = " << ngrp << ", " << progress_thread << " in " << nthread << " threads."
                << std::endl;

    /* calc DD */
// #pragma omp for schedule(auto) nowait
#pragma omp for schedule(dynamic) nowait
    for(uint64_t i = 0; i < ngrp; i++) {
      const double xi = grp[i].xpos;
      const double yi = grp[i].ypos;
      const double zi = grp[i].zpos;

      for(uint64_t j = i + 1; j < ngrp; j++) {
        double dx = grp[j].xpos - xi;
        double dy = grp[j].ypos - yi;
        double dz = grp[j].zpos - zi;

        dx = (dx > 0.5 ? dx - 1.e0 : dx);
        dy = (dy > 0.5 ? dy - 1.e0 : dy);
        dz = (dz > 0.5 ? dz - 1.e0 : dz);
        dx = (dx < -0.5 ? dx + 1.e0 : dx);
        dy = (dy < -0.5 ? dy + 1.e0 : dy);
        dz = (dz < -0.5 ? dz + 1.e0 : dz);

        const double dr2 = dx * dx + dy * dy + dz * dz;
        const int ir = get_r2_index(dr2);
        if(ir < nr && ir >= 0) {
          thr_dd_pair[ir] += 1.0;
        }
      }

      if(ithread == 0) {
        progress++;
        if(progress % progress_div == 0) {
          std::cerr << "\r\033[2K DD : " << (double)100.0 * progress / (double)progress_thread << " [%]";
        }
      }
    }

    if(ithread == 0) std::cerr << std::endl;

#pragma omp critical
    {
      for(int ir = 0; ir < nr; ir++) {
        dd_pair[ir] += thr_dd_pair[ir];
      }
    } // omp critical
  } // omp parallel

  // double V_box = lbox * lbox * lbox;
  double V_box = 1.0;
  double N_pairs = (double)ngrp * (ngrp - 1) / 2.0;
  double dr = (log_scale) ? (log(rmax / rmin) / nr) : ((rmax - rmin) / nr);

  for(int ir = 0; ir < nr; ir++) {
    double r_low = (log_scale) ? (rmin * exp(ir * dr)) : (rmin + ir * dr);
    double r_high = (log_scale) ? (rmin * exp((ir + 1) * dr)) : (rmin + (ir + 1) * dr);
    double shell_volume = (4.0 / 3.0) * M_PI * (r_high * r_high * r_high - r_low * r_low * r_low);
    double norm = N_pairs * shell_volume / V_box;
    xi[ir] = dd_pair[ir] / norm - 1.0;
  }
}

void correlation::calc_xi_cell_list()
{
  /* Advice by chatgpt */
  ngrp = grp.size();
  nrand = ngrp;

  dd_pair.resize(nr);
  xi.resize(nr);

  for(int i = 0; i < nr; i++) {
    dd_pair[i] = 0.0;
    xi[i] = 0.0;
  }

  /* Here only the global box size */
  const int ncx = ndiv_1d * std::ceil(1.0 / rmax);
  const int ncy = ndiv_1d * std::ceil(1.0 / rmax);
  const int ncz = ndiv_1d * std::ceil(1.0 / rmax);
  const int nc3 = ncx * ncy * ncz;

  std::vector<std::vector<int>> cell_list(nc3);

  for(int i = 0; i < ngrp; i++) {
    const int ix = get_cell_index(grp[i].xpos, ncx);
    const int iy = get_cell_index(grp[i].ypos, ncy);
    const int iz = get_cell_index(grp[i].zpos, ncz);
    const int cell_id = iz + ncz * (iy + ncy * ix);
    cell_list[cell_id].push_back(i);
  }

#pragma omp parallel
  {
    std::vector<double> thr_dd_pair(nr, 0.0);

    int nthread = omp_get_num_threads();
    int ithread = omp_get_thread_num();
    uint64_t progress = 0;
    uint64_t progress_thread = nc3 / nthread;
    uint64_t progress_div = 1 + progress_thread / 200;

    if(ithread == 0)
      std::cerr << "# nc^3, ngrp_thread = " << nc3 << ", " << progress_thread << " in " << nthread << " threads."
                << std::endl;

#pragma omp for collapse(3) schedule(dynamic)
    for(int ix = 0; ix < ncx; ix++) {
      for(int iy = 0; iy < ncy; iy++) {
        for(int iz = 0; iz < ncz; iz++) {

          const int cell_id = iz + ncz * (iy + ncy * ix);
          const auto &clist = cell_list[cell_id];

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
                    if(cell_id > ncell_id) continue;
                    if(cell_id == ncell_id && ii >= jj) continue;

                    double dx = grp[jj].xpos - grp[ii].xpos;
                    double dy = grp[jj].ypos - grp[ii].ypos;
                    double dz = grp[jj].zpos - grp[ii].zpos;

                    dx = (dx > 0.5 ? dx - 1.e0 : dx);
                    dy = (dy > 0.5 ? dy - 1.e0 : dy);
                    dz = (dz > 0.5 ? dz - 1.e0 : dz);
                    dx = (dx < -0.5 ? dx + 1.e0 : dx);
                    dy = (dy < -0.5 ? dy + 1.e0 : dy);
                    dz = (dz < -0.5 ? dz + 1.e0 : dz);

                    const double dr2 = dx * dx + dy * dy + dz * dz;
                    const int ir = get_r2_index(dr2);
                    if(ir >= 0 && ir < nr) {
                      thr_dd_pair[ir] += 1.0;
                    }
                  }
                }
              }
            }
          } // dx, dy, dz

          if(ithread == 0) {
            progress++;
            if(progress % progress_div == 0) {
              std::cerr << "\r\033[2K DD : " << (double)100.0 * progress / (double)progress_thread << " [%]";
            }
          }
        }
      }
    } // ix, iy, iz

    if(ithread == 0) std::cerr << std::endl;

#pragma omp critical
    {
      for(int ir = 0; ir < nr; ++ir) {
        dd_pair[ir] += thr_dd_pair[ir];
      }
    } // omp critial
  } // omp parallel

  // double V_box = lbox * lbox * lbox;
  double V_box = 1.0;
  double N_pairs = (double)ngrp * (ngrp - 1) / 2.0;
  double dr = (log_scale) ? (log(rmax / rmin) / nr) : ((rmax - rmin) / nr);

  for(int ir = 0; ir < nr; ir++) {
    double r_low = (log_scale) ? (rmin * exp(ir * dr)) : (rmin + ir * dr);
    double r_high = (log_scale) ? (rmin * exp((ir + 1) * dr)) : (rmin + (ir + 1) * dr);
    double shell_volume = (4.0 / 3.0) * M_PI * (r_high * r_high * r_high - r_low * r_low * r_low);
    double norm = N_pairs * shell_volume / V_box;
    xi[ir] = dd_pair[ir] / norm - 1.0;
  }
}

void correlation::calc_xi_LS_direct()
{
  /* Landy SD, Szalay AS 1993, Apj */
  ngrp = grp.size();
  nrand = ngrp;

  dd_pair.resize(nr);
  dr_pair.resize(nr);
  rr_pair.resize(nr);
  xi.resize(nr);

  for(int i = 0; i < nr; i++) {
    rr_pair[i] = 0.0;
    dd_pair[i] = 0.0;
    dr_pair[i] = 0.0;
    xi[i] = 0.0;
  }

  set_random_group();

#pragma omp parallel
  {
    std::vector<double> thr_dd_pair(nr, 0.0);
    std::vector<double> thr_rr_pair(nr, 0.0);
    std::vector<double> thr_dr_pair(nr, 0.0);

    int nthread = omp_get_num_threads();
    int ithread = omp_get_thread_num();
    uint64_t progress = 0;
    uint64_t progress_thread = ngrp / nthread;
    uint64_t progress_div = 1 + progress_thread / 200;

    if(ithread == 0)
      std::cerr << "# ngrp, ngrp_thread = " << ngrp << ", " << progress_thread << " in " << nthread << " threads."
                << std::endl;

    /* calc RR */
// #pragma omp for schedule(auto) nowait
#pragma omp for schedule(dynamic) nowait
    for(uint64_t i = 0; i < nrand; i++) {
      const double xi = rand[i].xpos;
      const double yi = rand[i].ypos;
      const double zi = rand[i].zpos;

      for(uint64_t j = i + 1; j < nrand; j++) {
        double dx = rand[j].xpos - xi;
        double dy = rand[j].ypos - yi;
        double dz = rand[j].zpos - zi;

        dx = (dx > 0.5 ? dx - 1.e0 : dx);
        dy = (dy > 0.5 ? dy - 1.e0 : dy);
        dz = (dz > 0.5 ? dz - 1.e0 : dz);
        dx = (dx < -0.5 ? dx + 1.e0 : dx);
        dy = (dy < -0.5 ? dy + 1.e0 : dy);
        dz = (dz < -0.5 ? dz + 1.e0 : dz);

        const double dr2 = dx * dx + dy * dy + dz * dz;
        const int ir = get_r2_index(dr2);
        if(ir < nr && ir >= 0) {
          thr_rr_pair[ir] += 1.0;
        }
      }

      if(ithread == 0) {
        progress++;
        if(progress % progress_div == 0) {
          // std::cerr << "RR : " << (double)100.0 * progress / (double)progress_thread << " [%]" << "\n";
          std::cerr << "\r\033[2K RR : " << (double)100.0 * progress / (double)progress_thread << " [%]";
        }
      }
    }

    if(ithread == 0) std::cerr << std::endl;
    progress = 0;

    /* calc DR */
// #pragma omp for schedule(auto) nowait
#pragma omp for schedule(dynamic) nowait
    for(uint64_t i = 0; i < ngrp; i++) {
      const double xi = grp[i].xpos;
      const double yi = grp[i].ypos;
      const double zi = grp[i].zpos;

      for(int j = 0; j < nrand; j++) {
        double dx = rand[j].xpos - xi;
        double dy = rand[j].ypos - yi;
        double dz = rand[j].zpos - zi;

        dx = (dx > 0.5 ? dx - 1.e0 : dx);
        dy = (dy > 0.5 ? dy - 1.e0 : dy);
        dz = (dz > 0.5 ? dz - 1.e0 : dz);
        dx = (dx < -0.5 ? dx + 1.e0 : dx);
        dy = (dy < -0.5 ? dy + 1.e0 : dy);
        dz = (dz < -0.5 ? dz + 1.e0 : dz);

        const double dr2 = dx * dx + dy * dy + dz * dz;
        const int ir = get_r2_index(dr2);
        if(ir < nr && ir >= 0) {
          thr_dr_pair[ir] += 0.5;
        }
      }

      if(ithread == 0) {
        progress++;
        if(progress % progress_div == 0) {
          std::cerr << "\r\033[2K DR : " << (double)100.0 * progress / (double)progress_thread << " [%]";
        }
      }
    }

    if(ithread == 0) std::cerr << std::endl;
    progress = 0;

    /* calc DD */
// #pragma omp for schedule(auto) nowait
#pragma omp for schedule(dynamic) nowait
    for(uint64_t i = 0; i < ngrp; i++) {
      const double xi = grp[i].xpos;
      const double yi = grp[i].ypos;
      const double zi = grp[i].zpos;

      for(uint64_t j = i + 1; j < ngrp; j++) {
        double dx = grp[j].xpos - xi;
        double dy = grp[j].ypos - yi;
        double dz = grp[j].zpos - zi;

        dx = (dx > 0.5 ? dx - 1.e0 : dx);
        dy = (dy > 0.5 ? dy - 1.e0 : dy);
        dz = (dz > 0.5 ? dz - 1.e0 : dz);
        dx = (dx < -0.5 ? dx + 1.e0 : dx);
        dy = (dy < -0.5 ? dy + 1.e0 : dy);
        dz = (dz < -0.5 ? dz + 1.e0 : dz);

        const double dr2 = dx * dx + dy * dy + dz * dz;
        const int ir = get_r2_index(dr2);
        if(ir < nr && ir >= 0) {
          thr_dd_pair[ir] += 1.0;
        }
      }

      if(ithread == 0) {
        progress++;
        if(progress % progress_div == 0) {
          std::cerr << "\r\033[2K DD : " << (double)100.0 * progress / (double)progress_thread << " [%]";
        }
      }
    }

    if(ithread == 0) std::cerr << std::endl;

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
    if(rr_pair[ir] != 0.0 && dr_pair[ir] != 0.0) {
      xi[ir] = (dd_pair[ir] * f2 - 2.0 * dr_pair[ir] * f + rr_pair[ir]) / rr_pair[ir];
    }
  }
}

void correlation::calc_xi_LS_cell_list()
{
  /* Landy SD, Szalay AS 1993, Apj */
  /* Advice by chatgpt */
  ngrp = grp.size();
  nrand = ngrp;

  dd_pair.resize(nr);
  dr_pair.resize(nr);
  rr_pair.resize(nr);
  xi.resize(nr);

  for(int i = 0; i < nr; i++) {
    rr_pair[i] = 0.0;
    dd_pair[i] = 0.0;
    dr_pair[i] = 0.0;
    xi[i] = 0.0;
  }

  set_random_group();

  /* Here only the global box size */
  const int ncx = ndiv_1d * std::ceil(1.0 / rmax);
  const int ncy = ndiv_1d * std::ceil(1.0 / rmax);
  const int ncz = ndiv_1d * std::ceil(1.0 / rmax);
  const int nc3 = ncx * ncy * ncz;

  std::vector<std::vector<int>> cell_list(nc3);
  std::vector<std::vector<int>> cell_list_rand(nc3);

  for(int i = 0; i < ngrp; i++) {
    const int ix = get_cell_index(grp[i].xpos, ncx);
    const int iy = get_cell_index(grp[i].ypos, ncy);
    const int iz = get_cell_index(grp[i].zpos, ncz);
    const int cell_id = iz + ncz * (iy + ncy * ix);
    cell_list[cell_id].push_back(i);
  }

  for(int i = 0; i < nrand; i++) {
    const int ix = get_cell_index(rand[i].xpos, ncx);
    const int iy = get_cell_index(rand[i].ypos, ncy);
    const int iz = get_cell_index(rand[i].zpos, ncz);
    const int cell_id = iz + ncz * (iy + ncy * ix);
    cell_list_rand[cell_id].push_back(i);
  }

#pragma omp parallel
  {
    std::vector<double> thr_dd_pair(nr, 0.0);
    std::vector<double> thr_rr_pair(nr, 0.0);
    std::vector<double> thr_dr_pair(nr, 0.0);

    int nthread = omp_get_num_threads();
    int ithread = omp_get_thread_num();
    uint64_t progress = 0;
    uint64_t progress_thread = nc3 / nthread;
    uint64_t progress_div = 1 + progress_thread / 200;

    if(ithread == 0)
      std::cerr << "# nc^3, ngrp_thread = " << nc3 << ", " << progress_thread << " in " << nthread << " threads."
                << std::endl;

/* calc RR */
#pragma omp for collapse(3) schedule(dynamic)
    for(int ix = 0; ix < ncx; ix++) {
      for(int iy = 0; iy < ncy; iy++) {
        for(int iz = 0; iz < ncz; iz++) {

          const int cell_id = iz + ncz * (iy + ncy * ix);
          const auto &clist = cell_list_rand[cell_id];

          for(int jx = -ndiv_1d; jx <= ndiv_1d; jx++) {
            for(int jy = -ndiv_1d; jy <= ndiv_1d; jy++) {
              for(int jz = -ndiv_1d; jz <= ndiv_1d; jz++) {

                const int nix = ((ix + jx) + ncx) % ncx;
                const int niy = ((iy + jy) + ncy) % ncy;
                const int niz = ((iz + jz) + ncz) % ncz;

                const int ncell_id = niz + ncz * (niy + ncy * nix);
                const auto &nlist = cell_list_rand[ncell_id];

                for(int ii : clist) {
                  for(int jj : nlist) {
                    if(cell_id > ncell_id) continue;
                    if(cell_id == ncell_id && ii >= jj) continue;

                    double dx = rand[jj].xpos - rand[ii].xpos;
                    double dy = rand[jj].ypos - rand[ii].ypos;
                    double dz = rand[jj].zpos - rand[ii].zpos;

                    dx = (dx > 0.5 ? dx - 1.e0 : dx);
                    dy = (dy > 0.5 ? dy - 1.e0 : dy);
                    dz = (dz > 0.5 ? dz - 1.e0 : dz);
                    dx = (dx < -0.5 ? dx + 1.e0 : dx);
                    dy = (dy < -0.5 ? dy + 1.e0 : dy);
                    dz = (dz < -0.5 ? dz + 1.e0 : dz);

                    const double dr2 = dx * dx + dy * dy + dz * dz;
                    const int ir = get_r2_index(dr2);
                    if(ir >= 0 && ir < nr) {
                      thr_rr_pair[ir] += 1.0;
                    }
                  }
                }
              }
            }
          } // dx, dy, dz

          if(ithread == 0) {
            progress++;
            if(progress % progress_div == 0) {
              std::cerr << "\r\033[2K RR : " << (double)100.0 * progress / (double)progress_thread << " [%]";
            }
          }
        }
      }
    } // ix, iy, iz

    if(ithread == 0) std::cerr << std::endl;
    progress = 0;

    /* calc DR */
#pragma omp for collapse(3) schedule(dynamic)
    for(int ix = 0; ix < ncx; ix++) {
      for(int iy = 0; iy < ncy; iy++) {
        for(int iz = 0; iz < ncz; iz++) {

          const int cell_id = iz + ncz * (iy + ncy * ix);
          const auto &clist = cell_list[cell_id];

          for(int jx = -ndiv_1d; jx <= ndiv_1d; jx++) {
            for(int jy = -ndiv_1d; jy <= ndiv_1d; jy++) {
              for(int jz = -ndiv_1d; jz <= ndiv_1d; jz++) {

                const int nix = ((ix + jx) + ncx) % ncx;
                const int niy = ((iy + jy) + ncy) % ncy;
                const int niz = ((iz + jz) + ncz) % ncz;

                const int ncell_id = niz + ncz * (niy + ncy * nix);
                const auto &nlist = cell_list_rand[ncell_id];

                for(int ii : clist) {
                  for(int jj : nlist) {
                    // if(cell_id > ncell_id) continue;
                    //  if(cell_id == ncell_id && ii >= jj) continue;

                    double dx = rand[jj].xpos - grp[ii].xpos;
                    double dy = rand[jj].ypos - grp[ii].ypos;
                    double dz = rand[jj].zpos - grp[ii].zpos;

                    dx = (dx > 0.5 ? dx - 1.e0 : dx);
                    dy = (dy > 0.5 ? dy - 1.e0 : dy);
                    dz = (dz > 0.5 ? dz - 1.e0 : dz);
                    dx = (dx < -0.5 ? dx + 1.e0 : dx);
                    dy = (dy < -0.5 ? dy + 1.e0 : dy);
                    dz = (dz < -0.5 ? dz + 1.e0 : dz);

                    const double dr2 = dx * dx + dy * dy + dz * dz;
                    const int ir = get_r2_index(dr2);
                    if(ir >= 0 && ir < nr) {
                      thr_dr_pair[ir] += 0.5;
                    }
                  }
                }
              }
            }
          } // dx, dy, dz

          if(ithread == 0) {
            progress++;
            if(progress % progress_div == 0) {
              std::cerr << "\r\033[2K DR : " << (double)100.0 * progress / (double)progress_thread << " [%]";
            }
          }
        }
      }
    } // ix, iy, iz

    if(ithread == 0) std::cerr << std::endl;
    progress = 0;

    /* calc DD */
#pragma omp for collapse(3) schedule(dynamic)
    for(int ix = 0; ix < ncx; ix++) {
      for(int iy = 0; iy < ncy; iy++) {
        for(int iz = 0; iz < ncz; iz++) {

          const int cell_id = iz + ncz * (iy + ncy * ix);
          const auto &clist = cell_list[cell_id];

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
                    if(cell_id > ncell_id) continue;
                    if(cell_id == ncell_id && ii >= jj) continue;

                    double dx = grp[jj].xpos - grp[ii].xpos;
                    double dy = grp[jj].ypos - grp[ii].ypos;
                    double dz = grp[jj].zpos - grp[ii].zpos;

                    dx = (dx > 0.5 ? dx - 1.e0 : dx);
                    dy = (dy > 0.5 ? dy - 1.e0 : dy);
                    dz = (dz > 0.5 ? dz - 1.e0 : dz);
                    dx = (dx < -0.5 ? dx + 1.e0 : dx);
                    dy = (dy < -0.5 ? dy + 1.e0 : dy);
                    dz = (dz < -0.5 ? dz + 1.e0 : dz);

                    const double dr2 = dx * dx + dy * dy + dz * dz;
                    const int ir = get_r2_index(dr2);
                    if(ir >= 0 && ir < nr) {
                      thr_dd_pair[ir] += 1.0;
                    }
                  }
                }
              }
            }
          } // dx, dy, dz

          if(ithread == 0) {
            progress++;
            if(progress % progress_div == 0) {
              std::cerr << "\r\033[2K DD : " << (double)100.0 * progress / (double)progress_thread << " [%]";
            }
          }
        }
      }
    } // ix, iy, iz

    if(ithread == 0) std::cerr << std::endl;
    progress = 0;

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
    if(rr_pair[ir] != 0.0 && dr_pair[ir] != 0.0) {
      xi[ir] = (dd_pair[ir] * f2 - 2.0 * dr_pair[ir] * f + rr_pair[ir]) / rr_pair[ir];
    }
  }
}

void correlation::calc_xi_jk_LS()
{

  ngrp = grp.size();
  nrand = ngrp;
  jk_dd = ngrp / jk_block; // delete-d
  nblock = (int)ceil(ngrp / jk_dd);

  xi_jk.resize(nblock, std::vector<double>(nr));
  xi_ave.resize(nr);
  xi_sd.resize(nr);
  xi_se.resize(nr);

  dd_pair_jk.resize(nblock, std::vector<double>(nr));
  dr_pair_jk.resize(nblock, std::vector<double>(nr));
  rr_pair_jk.resize(nblock, std::vector<double>(nr));

#pragma omp parellel for collapse(2)
  for(int iblock = 0; iblock < nblock; iblock++) {
    for(int ir = 0; ir < nr; ir++) {
      dd_pair_jk[iblock][ir] = 0.0;
      rr_pair_jk[iblock][ir] = 0.0;
      dr_pair_jk[iblock][ir] = 0.0;
      xi_jk[iblock][ir] = 0.0;
    }
  }

  shuffle_halo_data();
  set_random_group();
  resample_jackknife_LS();
  calc_jk_xi_average();
  calc_jk_xi_error();
}

void correlation::resample_jackknife_LS()
{
  const double logrmin2 = log(rmin * rmin);
  const double dlogbin2 = (2.e0 / (double)nr) * log(rmax / rmin);

  int length = (ngrp - jk_dd) / (nblock - 1);

  for(int iblock = 0; iblock < nblock; iblock++) {
    int delete_block_start = iblock * length;
    int delete_block_end = delete_block_start + jk_dd;

    std::cerr << "# iblock " << iblock << " " << length << " " << delete_block_start << " " << delete_block_end
              << std::endl;

#pragma omp parallel
    {
      std::vector<double> thr_dd_pair(nr, 0.0);
      std::vector<double> thr_rr_pair(nr, 0.0);
      std::vector<double> thr_dr_pair(nr, 0.0);

      int nthread = omp_get_num_threads();
      int ithread = omp_get_thread_num();
      uint64_t progress = 0;
      uint64_t progress_thread = ngrp / nthread;
      uint64_t progress_div = 1 + progress_thread / 200;

      if(ithread == 0)
        std::cerr << "# ngrp, ngrp_thread = " << ngrp << ", " << progress_thread << " in " << nthread << " threads."
                  << std::endl;

// #pragma omp for schedule(auto) nowait
#pragma omp for schedule(dynamic) nowait
      for(uint64_t ii = 0; ii < nrand; ii++) {
        if((delete_block_start <= ii) && (ii < delete_block_end)) continue;

        const double xi = rand[ii].xpos;
        const double yi = rand[ii].ypos;
        const double zi = rand[ii].zpos;

        for(uint64_t jj = 0; jj < nrand; jj++) {
          double dx = rand[jj].xpos - xi;
          double dy = rand[jj].ypos - yi;
          double dz = rand[jj].zpos - zi;
          dx = (dx > 0.5 ? dx - 1.e0 : dx);
          dy = (dy > 0.5 ? dy - 1.e0 : dy);
          dz = (dz > 0.5 ? dz - 1.e0 : dz);
          dx = (dx < -0.5 ? dx + 1.e0 : dx);
          dy = (dy < -0.5 ? dy + 1.e0 : dy);
          dz = (dz < -0.5 ? dz + 1.e0 : dz);
          double dr2 = dx * dx + dy * dy + dz * dz;
          const int ir = (int)floor((log(dr2) - logrmin2) / dlogbin2);
          if(ir < nr && ir >= 0) {
            // thr_rr_pair[ir]+=1.0;  // for jj = ii + 1
            thr_rr_pair[ir] += 0.5; // for jj = 0
          }
        }

        if(ithread == 0) {
          progress++;
          if(progress % progress_div == 0) {
            std::cerr << "\r\033[2K RR : [" << iblock << "/" << nblock
                      << "] block : " << (double)100.0 * progress / (double)progress_thread << " [%]";
          }
        }
      }

      if(ithread == 0) std::cerr << std::endl;
      progress = 0;

// #pragma omp for schedule(auto) nowait
#pragma omp for schedule(dynamic) nowait
      for(uint64_t ii = 0; ii < ngrp; ii++) {
        if((delete_block_start <= ii) && (ii < delete_block_end)) continue;

        const double xi = grp[ii].xpos;
        const double yi = grp[ii].ypos;
        const double zi = grp[ii].zpos;

        for(uint64_t jj = 0; jj < nrand; jj++) {
          double dx = rand[jj].xpos - xi;
          double dy = rand[jj].ypos - yi;
          double dz = rand[jj].zpos - zi;
          dx = (dx > 0.5 ? dx - 1.e0 : dx);
          dy = (dy > 0.5 ? dy - 1.e0 : dy);
          dz = (dz > 0.5 ? dz - 1.e0 : dz);
          dx = (dx < -0.5 ? dx + 1.e0 : dx);
          dy = (dy < -0.5 ? dy + 1.e0 : dy);
          dz = (dz < -0.5 ? dz + 1.e0 : dz);
          double dr2 = dx * dx + dy * dy + dz * dz;
          int ir = (int)floor((log(dr2) - logrmin2) / dlogbin2);
          if(ir < nr && ir >= 0) {
            thr_dr_pair[ir] += 0.5;
          }
        }

        if(ithread == 0) {
          progress++;
          if(progress % progress_div == 0) {
            std::cerr << "\r\033[2K DR : [" << iblock << "/" << nblock
                      << "] block : " << (double)100.0 * progress / (double)progress_thread << " [%]";
          }
        }
      }

      if(ithread == 0) std::cerr << std::endl;
      progress = 0;

// #pragma omp for schedule(auto) nowait
#pragma omp for schedule(dynamic) nowait
      for(uint64_t ii = 0; ii < ngrp; ii++) {
        if((delete_block_start <= ii) && (ii < delete_block_end)) continue;

        const double xi = grp[ii].xpos;
        const double yi = grp[ii].ypos;
        const double zi = grp[ii].zpos;

        for(uint64_t jj = 0; jj < ngrp; jj++) {
          double dx = grp[jj].xpos - xi;
          double dy = grp[jj].ypos - yi;
          double dz = grp[jj].zpos - zi;
          dx = (dx > 0.5 ? dx - 1.e0 : dx);
          dy = (dy > 0.5 ? dy - 1.e0 : dy);
          dz = (dz > 0.5 ? dz - 1.e0 : dz);
          dx = (dx < -0.5 ? dx + 1.e0 : dx);
          dy = (dy < -0.5 ? dy + 1.e0 : dy);
          dz = (dz < -0.5 ? dz + 1.e0 : dz);
          double dr2 = dx * dx + dy * dy + dz * dz;
          int ir = (int)floor((log(dr2) - logrmin2) / dlogbin2);
          if(ir < nr && ir >= 0) {
            // thr_dd_pair[ir]+=1.0; // for jj = ii + 1
            thr_dd_pair[ir] += 0.5; // for jj = 0
          }
        }

        if(ithread == 0) {
          progress++;
          if(progress % progress_div == 0) {
            std::cerr << "\r\033[2K DD : [" << iblock << "/" << nblock
                      << "] block : " << (double)100.0 * progress / (double)progress_thread << " [%]";
          }
        }
      }

      if(ithread == 0) std::cerr << std::endl;

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

  double f = (double)(nrand - jk_dd) / (double)(ngrp - jk_dd);
  double f2 = f * f;
  for(int iblock = 0; iblock < nblock; iblock++) {
    for(int ir = 0; ir < nr; ir++) {
      if(rr_pair_jk[iblock][ir] != 0.0 && dr_pair_jk[iblock][ir] != 0.0) {
        xi_jk[iblock][ir] = (dd_pair_jk[iblock][ir] * f2 - 2.0 * dr_pair_jk[iblock][ir] * f + rr_pair_jk[iblock][ir]) /
                            rr_pair_jk[iblock][ir];
      }
    }
  }
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
    variance[ir] *= (double)(ngrp - jk_dd);
    xi_sd[ir] = sqrt(variance[ir]);
    xi_se[ir] = sqrt(variance[ir] / (double)(jk_dd * nblock));
  }

  std::cerr << "# done " << __func__ << std::endl;
}

void correlation::output_xi(std::string filename)
{
  std::ofstream fout(filename);
  fout << "# Mvir min, max = " << std::scientific << std::setprecision(4) << mmin << ", " << mmax << std::endl;
  fout << "# r[Mpc/h] xi DD" << std::endl;

  for(int ir = 0; ir < nr; ir++) {
    double rad = rcen[ir] * lbox;
    fout << std::scientific << std::setprecision(10) << rad << " " << xi[ir] << " " << dd_pair[ir] << "\n";
  }
  fout.flush();
  fout.close();
  std::cout << "output to " << filename << std::endl;
}

void correlation::output_xi_LS(std::string filename)
{
  std::ofstream fout(filename);
  fout << "# Mvir min, max = " << std::scientific << std::setprecision(4) << mmin << ", " << mmax << std::endl;
  fout << "# r[Mpc/h] xi DD DR RR" << std::endl;

  for(int ir = 0; ir < nr; ir++) {
    double rad = rcen[ir] * lbox;
    fout << std::scientific << std::setprecision(10) << rad << " " << xi[ir] << " " << dd_pair[ir] << " " << dr_pair[ir]
         << " " << rr_pair[ir] << "\n";
  }
  fout.flush();
  fout.close();
  std::cout << "output to " << filename << std::endl;
}

void correlation::output_xi_jk(std::string filename)
{
  std::ofstream fout(filename);
  fout << "# Mvir min, max = " << std::scientific << std::setprecision(4) << mmin << ", " << mmax << std::endl;
  fout << "# r[Mpc/h] xi_ave SD SE block1 block2 block3 ..." << std::endl;

  for(int ir = 0; ir < nr; ir++) {
    double rad = rcen[ir] * lbox;
    fout << std::scientific << std::setprecision(10) << rad << " " << xi_ave[ir] << " " << xi_sd[ir] << " "
         << xi_se[ir];
    for(int i = 0; i < nblock; i++) fout << std::scientific << std::setprecision(10) << " " << xi_jk[i][ir];
    fout << "\n";
  }
  fout.flush();
  fout.close();
  std::cout << "output to " << filename << std::endl;
};
