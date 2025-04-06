#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <cassert>

#include <fftw3.h>
#include "powerspec.hpp"
#include "util.hpp"

struct group {
  float xpos, ypos, zpos;
  float xvel, yvel, zvel;
  float rad, mass;
  float pot;
  int nmem;
  int block_id; // for jackknife
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
  // int ndiv_1d = 1;
  int ndiv_1d = 2;

  int jk_block, jk_level;

  int nrand, ngrp;
  int jk_dd, nblock;
  double mmin, mmax;

  std::vector<group> grp;
  std::vector<group> rand;

  // for position xi
  // support Landy SD, Szalay AS 1993, Apj
  bool use_LS = false;
  std::vector<double> dd_pair, dr_pair, rr_pair;                       // size[nr]
  std::vector<std::vector<double>> dd_pair_jk, dr_pair_jk, rr_pair_jk; // size[block][nr]
  std::vector<double> xi, xi_ave, xi_sd, xi_se;                        // size[nr]
  std::vector<std::vector<double>> xi_jk;                              // size[block][nr]

  std::vector<int64_t> block_start, block_end;
  std::vector<int64_t> rand_block_start, rand_block_end;

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

  void calc_xi();
  void calc_xi_LS();
  void calc_xi_jk(const int = 1);
  void calc_xi_jk_LS(const int = 1);

  void sort_jk_block();
  void sort_jk_rand_block();
  void set_block_edge_id();
  void set_rand_block_edge_id();
  void set_block_edge_id_shuffle();
  void set_rand_block_edge_id_shuffle();

  void resample_jk();
  void resample_jk_LS();
  void calc_jk_xi_average();
  void calc_jk_xi_error();

  template <typename T>
  void calc_xi_ifft(T &, T &);
  template <typename T>
  void calc_xi_jk_ifft(T &, T &);

  void output_xi(std::string);
  template <typename T>
  void output_xi(std::string, T &);
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

  rbin.assign(nr + 1, 0.0);
  rcen.assign(nr, 0.0);

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

void correlation::sort_jk_block()
{
#pragma omp parallel for
  for(int64_t i = 0; i < ngrp; i++) {
    const int ix = get_cell_index(grp[i].xpos, jk_level);
    const int iy = get_cell_index(grp[i].ypos, jk_level);
    const int iz = get_cell_index(grp[i].zpos, jk_level);
    grp[i].block_id = iz + jk_level * (iy + jk_level * ix);
  }
  std::sort(grp.begin(), grp.end(), [](const group &a, const group &b) { return a.block_id < b.block_id; });
}

void correlation::sort_jk_rand_block()
{
#pragma omp parallel for
  for(int64_t i = 0; i < nrand; i++) {
    const int ix = get_cell_index(rand[i].xpos, jk_level);
    const int iy = get_cell_index(rand[i].ypos, jk_level);
    const int iz = get_cell_index(rand[i].zpos, jk_level);
    rand[i].block_id = iz + jk_level * (iy + jk_level * ix);
  }
  std::sort(rand.begin(), rand.end(), [](const group &a, const group &b) { return a.block_id < b.block_id; });
}

void correlation::set_block_edge_id()
{
  block_start.assign(nblock, -1);
  block_end.assign(nblock, -1);

  for(int i = 0; i < ngrp; i++) {
    int bid = grp[i].block_id;
    if(block_start[bid] == -1) {
      block_start[bid] = i;
    }
    block_end[bid] = i + 1;
  }
}

void correlation::set_rand_block_edge_id()
{
  rand_block_start.assign(nblock, -1);
  rand_block_end.assign(nblock, -1);

  for(int i = 0; i < nrand; i++) {
    int bid = rand[i].block_id;
    if(rand_block_start[bid] == -1) {
      rand_block_start[bid] = i;
    }
    rand_block_end[bid] = i + 1;
  }
}

void correlation::set_block_edge_id_shuffle()
{
  block_start.assign(nblock, -1);
  block_end.assign(nblock, -1);

  int length = (ngrp - jk_dd) / (nblock - 1);
  for(int iblock = 0; iblock < nblock; iblock++) {
    block_start[iblock] = iblock * length;
    block_end[iblock] = block_start[iblock] + jk_dd;
  }
}

void correlation::set_rand_block_edge_id_shuffle()
{
  rand_block_start.assign(nblock, -1);
  rand_block_end.assign(nblock, -1);

  int length = (nrand - jk_dd) / (nblock - 1);
  for(int iblock = 0; iblock < nblock; iblock++) {
    rand_block_start[iblock] = iblock * length;
    rand_block_end[iblock] = rand_block_start[iblock] + jk_dd;
  }
}

void correlation::calc_xi()
{
  /* Advice by chatgpt */
  ngrp = grp.size();
  nrand = ngrp;

  dd_pair.assign(nr, 0.0);
  xi.assign(nr, 0.0);

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

void correlation::calc_xi_LS()
{
  /* Landy SD, Szalay AS 1993, Apj */
  /* Advice by chatgpt */
  use_LS = true;
  ngrp = grp.size();
  nrand = ngrp;

  dd_pair.assign(nr, 0.0);
  dr_pair.assign(nr, 0.0);
  rr_pair.assign(nr, 0.0);
  xi.assign(nr, 0.0);

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

#pragma omp unroll
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
                      thr_rr_pair[ir] += 1.0; // for j=i+1
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

#pragma omp unroll
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
                      thr_dr_pair[ir] += 0.5; // for j=0
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
                      thr_dd_pair[ir] += 1.0; // for j=i+1
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

void correlation::calc_xi_jk(const int type)
{
  /*
    type=0 : spaced-block (Something's wrong. LS version is fine.)
    type=1 : shuffle-block
  */
  ngrp = grp.size();
  nrand = ngrp;
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

  if(type == 0) {
    std::cerr << "blocked jackknife by spaced sampling" << std::endl;
    sort_jk_block();
    set_block_edge_id();
  } else {
    std::cerr << "blocked jackknife by shuffled sampling" << std::endl;
  shuffle_halo_data();
    set_block_edge_id_shuffle();
  }

  resample_jk();
  calc_jk_xi_average();
  calc_jk_xi_error();
}

void correlation::calc_xi_jk_LS(const int type)
{
  /*
    type=0 : spaced-block (Something's wrong.)
    type=1 : shuffle-block
  */

  use_LS = true;
  ngrp = grp.size();
  nrand = ngrp;
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

  set_random_group();

  if(type == 0) {
    std::cerr << "blocked jackknife by spaced sampling" << std::endl;
    sort_jk_block();
    set_block_edge_id();
    sort_jk_rand_block();
    set_rand_block_edge_id();
  } else {
    std::cerr << "blocked jackknife by shuffled sampling" << std::endl;
    shuffle_halo_data();
    set_block_edge_id_shuffle();
    set_rand_block_edge_id_shuffle();
  }

  resample_jk_LS();
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

  // #pragma omp parallel for
  for(int ir = 0; ir < nr; ir++) {
    variance[ir] *= (double)(ngrp - jk_dd);
    xi_sd[ir] = sqrt(variance[ir]);
    xi_se[ir] = sqrt(variance[ir] / (double)(jk_dd * nblock));
  }

  std::cerr << "# done " << __func__ << std::endl;
}

void correlation::resample_jk()
{
  /* Here only the global box size */
  const int ncx = ndiv_1d * std::ceil(1.0 / rmax);
  const int ncy = ndiv_1d * std::ceil(1.0 / rmax);
  const int ncz = ndiv_1d * std::ceil(1.0 / rmax);
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
                        thr_dd_pair[ir] += 1.0; // for j=i+1
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
        for(int ir = 0; ir < nr; ir++) {
          dd_pair_jk[iblock][ir] += thr_dd_pair[ir];
        }
      }
    } // end parallel
  } // end nblock loop

  double V_box = 1.0;

  for(int iblock = 0; iblock < nblock; iblock++) {
    int64_t length = block_end[iblock] - block_start[iblock];
    double N_pairs = (double)(ngrp - length) * ((ngrp - length) - 1) / 2.0;
    double dr = (log_scale) ? (log(rmax / rmin) / nr) : ((rmax - rmin) / nr);

    for(int ir = 0; ir < nr; ir++) {
      double r_low = (log_scale) ? (rmin * exp(ir * dr)) : (rmin + ir * dr);
      double r_high = (log_scale) ? (rmin * exp((ir + 1) * dr)) : (rmin + (ir + 1) * dr);
      double shell_volume = (4.0 / 3.0) * M_PI * (r_high * r_high * r_high - r_low * r_low * r_low);
      double norm = N_pairs * shell_volume / V_box;
      xi_jk[iblock][ir] = dd_pair_jk[iblock][ir] / norm - 1.0;
    }
  }
}

void correlation::resample_jk_LS()
{
  use_LS = true;

  /* Here only the global box size */
  const int ncx = ndiv_1d * std::ceil(1.0 / rmax);
  const int ncy = ndiv_1d * std::ceil(1.0 / rmax);
  const int ncz = ndiv_1d * std::ceil(1.0 / rmax);
  const int nc3 = ncx * ncy * ncz;

  for(int iblock = 0; iblock < nblock; iblock++) {
    int64_t delete_block_start = block_start[iblock];
    int64_t delete_block_end = block_end[iblock];

    int64_t delete_rand_block_start = rand_block_start[iblock];
    int64_t delete_rand_block_end = rand_block_end[iblock];

    std::cerr << "# iblock " << iblock << " " << delete_block_start << " " << delete_block_end << std::endl;

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
      const int ix = get_cell_index(rand[i].xpos, ncx);
      const int iy = get_cell_index(rand[i].ypos, ncy);
      const int iz = get_cell_index(rand[i].zpos, ncz);
      const int cell_id = iz + ncz * (iy + ncy * ix);
      if((delete_rand_block_start <= i) && (i < delete_rand_block_end)) continue;
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

#pragma omp unroll
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
                        thr_rr_pair[ir] += 1.0; // for j=i+1
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

/* calc DR */
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
                  const auto &nlist = cell_list_rand[ncell_id];

                  for(int ii : clist) {
                    for(int jj : nlist) {
                      // if(cell_id > ncell_id) continue;
                      // if(cell_id == ncell_id && ii >= jj) continue;

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
                        thr_dr_pair[ir] += 0.5; // for j=0
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
                        thr_dd_pair[ir] += 1.0; // for j=i+1
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
      if(rr_pair_jk[iblock][ir] != 0.0 && dr_pair_jk[iblock][ir] != 0.0) {
        xi_jk[iblock][ir] = (dd_pair_jk[iblock][ir] * f2 - 2.0 * dr_pair_jk[iblock][ir] * f + rr_pair_jk[iblock][ir]) /
                            rr_pair_jk[iblock][ir];
      }
    }
  }
}

template <typename T>
void correlation::calc_xi_ifft(T &mesh, T &weight)
{
  static_assert(!std::is_same<T, float>::value, "Only float is allowed");

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
      fftwf_plan_dft_r2c_3d(nmesh, nmesh, nmesh, mesh.data(), (fftwf_complex *)mesh.data(), FFTW_ESTIMATE);
  fftwf_execute(plan_forward);

  fftwf_complex *mesh_hat = (fftwf_complex *)mesh.data();

#pragma omp parallel for collapse(3)
  for(uint64_t ix = 0; ix < nmesh; ix++) {
    for(uint64_t iy = 0; iy < nmesh; iy++) {
      for(uint64_t iz = 0; iz < nmesh / 2 + 1; iz++) {
        int64_t im = iz + (nmesh / 2 + 1) * (iy + nmesh * ix);

#if 1
        const float kx = (ix < nmesh / 2) ? (float)(ix) : (float)(nmesh - ix);
        const float ky = (iy < nmesh / 2) ? (float)(iy) : (float)(nmesh - iy);
        const float kz = (float)(iz);

        auto d = pi / nmesh;
        auto wx = (ix == 0) ? 1.0 : (sin(d * kx) / (d * kx));
        auto wy = (iy == 0) ? 1.0 : (sin(d * ky) / (d * ky));
        auto wz = (iz == 0) ? 1.0 : (sin(d * kz) / (d * kz));

        // p == 1 ; // for NGP weight w
        if(p == 2) {
          // for CIC weight w^2
          wx *= wx;
          wy *= wy;
          wz *= wz;
        } else if(p == 3) {
          // for TSC weight w^3 ((w^3)^2 in Fourie space)
          wx = (wx * wx * wx);
          wy = (wy * wy * wy);
          wz = (wz * wz * wz);
}

        // for Fourie space
        wx = wx * wx;
        wy = wy * wy;
        wz = wz * wz;

        const auto win2 = (p == 0) ? 1.0 : wx * wy * wz;
#else
        const auto win2 = 1.0;
#endif
        auto power = SQR(c_re(mesh_hat[im])) + SQR(c_im(mesh_hat[im]));
        c_re(mesh_hat[im]) = power / win2;
        c_im(mesh_hat[im]) = 0.0;
      }
    }
  } // ix,iy,iz loop

  /* ifft */
  fftwf_plan plan_backward = fftwf_plan_dft_c2r_3d(nmesh, nmesh, nmesh, mesh_hat, mesh.data(), FFTW_ESTIMATE);
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
void correlation::calc_xi_jk_ifft(T &mesh_orig, T &weight)
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
  ngrp = mesh_orig.size();
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

/* set (rho/rho_bar)-1 to (rho/rho_bar)  */
#pragma omp parallel for collapse(3)
  for(int64_t ix = 0; ix < nmesh; ix++) {
    for(int64_t iy = 0; iy < nmesh; iy++) {
      for(int64_t iz = 0; iz < nmesh; iz++) {
        int64_t im = iz + (nmesh + 2) * (iy + nmesh * ix);
        mesh_orig[im] += 1.0;
      }
    }
  }

  T mesh = mesh_orig; // deep copy
  fftwf_complex *mesh_hat = (fftwf_complex *)mesh.data();

  /* fft */
  fftwf_plan plan_forward =
      fftwf_plan_dft_r2c_3d(nmesh, nmesh, nmesh, mesh.data(), (fftwf_complex *)mesh.data(), FFTW_ESTIMATE);
  fftwf_plan plan_backward = fftwf_plan_dft_c2r_3d(nmesh, nmesh, nmesh, mesh_hat, mesh.data(), FFTW_ESTIMATE);

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

#pragma omp parallel for
    for(int64_t im = 0; im < mesh.size(); im++) {
      mesh[im] = 0.0;
    }

#pragma omp parallel for collapse(3)
    for(int64_t ix = 0; ix < nmesh; ix++) {
      for(int64_t iy = 0; iy < nmesh; iy++) {
        for(int64_t iz = 0; iz < nmesh; iz++) {
          int64_t im = iz + (nmesh + 2) * (iy + nmesh * ix);
          mesh[im] = mesh_orig[im];
        }
      }
    }

/* current mean density=1 */
#pragma omp parallel for collapse(3)
    for(int64_t ix = 0; ix < nmesh; ix++) {
      for(int64_t iy = 0; iy < nmesh; iy++) {
        for(int64_t iz = 0; iz < nmesh; iz++) {
          int64_t im = iz + (nmesh + 2) * (iy + nmesh * ix);
          if((ix >= ix_min && ix < ix_max) && (iy >= iy_min && iy < iy_max) && (iz >= iz_min && iz < iz_max)) {
            mesh[im] = 1.0;
          }
        }
      }
    }

    double mean_with_zero = 0.0;

/* current mean density=1 */
#pragma omp parallel for collapse(3) reduction(+ : mean_with_zero)
    for(int64_t ix = 0; ix < nmesh; ix++) {
      for(int64_t iy = 0; iy < nmesh; iy++) {
        for(int64_t iz = 0; iz < nmesh; iz++) {
          int64_t im = iz + (nmesh + 2) * (iy + nmesh * ix);
          mean_with_zero += mesh[im];
        }
      }
    }

    mean_with_zero = mean_with_zero / (double)(nmesh * nmesh * nmesh);

#pragma omp parallel for collapse(3)
    for(int64_t ix = 0; ix < nmesh; ix++) {
      for(int64_t iy = 0; iy < nmesh; iy++) {
        for(int64_t iz = 0; iz < nmesh; iz++) {
          int64_t im = iz + (nmesh + 2) * (iy + nmesh * ix);
          mesh[im] = mesh[im] / mean_with_zero - 1.0;
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

#if 1
        const float kx = (ix < nmesh / 2) ? (float)(ix) : (float)(nmesh - ix);
        const float ky = (iy < nmesh / 2) ? (float)(iy) : (float)(nmesh - iy);
        const float kz = (float)(iz);

        auto d = pi / nmesh;
        auto wx = (ix == 0) ? 1.0 : (sin(d * kx) / (d * kx));
        auto wy = (iy == 0) ? 1.0 : (sin(d * ky) / (d * ky));
        auto wz = (iz == 0) ? 1.0 : (sin(d * kz) / (d * kz));

        // p == 1 ; // for NGP weight w
        if(p == 2) {
          // for CIC weight w^2
          wx *= wx;
          wy *= wy;
          wz *= wz;
        } else if(p == 3) {
          // for TSC weight w^3 ((w^3)^2 in Fourie space)
          wx = (wx * wx * wx);
          wy = (wy * wy * wy);
          wz = (wz * wz * wz);
        }

        // for Fourie space
        wx = wx * wx;
        wy = wy * wy;
        wz = wz * wz;

        const auto win2 = (p == 0) ? 1.0 : wx * wy * wz;
#else
        const auto win2 = 1.0;
#endif
        auto power = SQR(c_re(mesh_hat[im])) + SQR(c_im(mesh_hat[im]));
        c_re(mesh_hat[im]) = power / win2;
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

void correlation::output_xi(std::string filename)
{
  std::ofstream fout(filename);
  fout << "# Mvir min, max = " << std::scientific << std::setprecision(4) << mmin << ", " << mmax << std::endl;

  if(jk_block <= 1) {
    if(use_LS) {
      fout << "# r[Mpc/h] xi DD DR RR" << std::endl;
      for(int ir = 0; ir < nr; ir++) {
        double rad = rcen[ir] * lbox;
        fout << std::scientific << std::setprecision(10) << rad << " " << xi[ir] << " " << dd_pair[ir] << " "
             << dr_pair[ir] << " " << rr_pair[ir] << "\n";
      }
    } else {
      if(dd_pair.size() > 0) {
        fout << "# r[Mpc/h] xi DD" << std::endl;
  for(int ir = 0; ir < nr; ir++) {
    double rad = rcen[ir] * lbox;
    fout << std::scientific << std::setprecision(10) << rad << " " << xi[ir] << " " << dd_pair[ir] << "\n";
  }
      } else {
        fout << "# r[Mpc/h] xi" << std::endl;
  for(int ir = 0; ir < nr; ir++) {
    double rad = rcen[ir] * lbox;
          fout << std::scientific << std::setprecision(10) << rad << " " << xi[ir] << "\n";
  }
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
  fout << "# Mvir min, max = " << std::scientific << std::setprecision(4) << mmin << ", " << mmax << std::endl;

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
