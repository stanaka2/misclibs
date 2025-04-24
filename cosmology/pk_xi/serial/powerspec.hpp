#pragma once

#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cassert>

#include <omp.h>
#include <fftw3.h>
#include "util.hpp"

#define c_re(c) ((c)[0])
#define c_im(c) ((c)[1])
#define SQR(a) ((a) * (a))
#define CUBE(a) ((a) * (a) * (a))

class powerspec
{
public:
  int nmesh;
  int ellmax = 12;
  double lbox;

  int nk;
  double kmin, kmax, dk;
  double kny, kny2;
  int ikny, ikny2;
  std::vector<float> kbin, kcen; // bin edge and center

  bool log_scale;
  double lin_dk, lin_dk2; // k, k^2 base
  double ratio, ratio2;
  double logratio, logratio2;

  int p = -1; // 0: non correct
              // 1:NGP : nearest grid point
              // 2:CIC : clouds-in-cell//
              // 3:TSC : triangular-shaped cloud
              // 4:PCS : Piecewise Cubic Spline

  bool shotnoise_corr = false;
  double shotnoise = 0.0;
  // No shot noise correction is applied for cross power spectrum.
  // This is because shot noise arises from self-pairs (i == j),
  // which are absent in cross-correlations between independent particle fields.

  void set_kbin(double, double, int, bool);
  void check_kbin();
  void check_p() const;

  template <typename T>
  int get_k_index(T);

  float calc_window(int64_t, int64_t, int64_t);
  template <typename T>
  void set_shotnoise(const T, const double = 1.0);

  template <typename T>
  void calc_power_spec(T &, T &, T &);
  template <typename T>
  void calc_power_spec(T &, T &, T &, T &);

  template <typename T, typename TT>
  void calc_power_spec_ell(T &, TT &, T &);
  template <typename T, typename TT>
  void calc_power_spec_ell(T &, T &, TT &, T &);

  template <typename T>
  void output_pk(T &, T &, std::string);
  template <typename T, typename TT>
  void output_pk_ell(TT &, T &, std::string);
};

void powerspec::set_kbin(double _kmin, double _kmax, int _nk, bool _log_scale = true)
{
  nk = _nk;
  dk = 2.0 * pi;
  kmin = _kmin * lbox;
  kmax = _kmax * lbox;

  kmin /= dk;
  kmax /= dk;

  log_scale = _log_scale;

  kbin.assign(nk + 1, 0.0);
  kcen.assign(nk, 0.0);

  if(kmax > nmesh) {
    kmax = nmesh;
  }

  if(log_scale) {
    if(kmin < 1e-10) kmin = 1e-10;
    ratio = pow(kmax / kmin, 1.0 / (double)(nk));
    logratio = log(ratio);
    for(int ik = 0; ik < nk + 1; ik++) kbin[ik] = kmin * pow(ratio, ik);
    for(int ik = 0; ik < nk; ik++) kcen[ik] = kmin * pow(ratio, ik + 0.5);
  } else {
    kmin = 0.0;
    lin_dk = (kmax - kmin) / (double)(nk);
    for(int ik = 0; ik < nk + 1; ik++) kbin[ik] = kmin + lin_dk * ik;
    for(int ik = 0; ik < nk; ik++) kcen[ik] = kmin + lin_dk * (ik + 0.5);
  }

  kny = (double)nmesh / 2.0;
  kny2 = kny / 2.0;
  ikny = get_k_index(kny);
  ikny2 = get_k_index(kny2);
}

void powerspec::check_kbin()
{
  for(int ik = 0; ik < nk; ik++)
    std::cerr << ik << " " << kcen[ik] << " " << kcen[ik] * dk / lbox << " " << get_k_index(kcen[ik]) << "\n";
  std::cerr << "k  min " << kmin << " max " << kmax << " ny " << kny << "\n";
  std::cerr << "k/L [Mpc/h] min " << kmin / lbox << " max " << kmax / lbox << " ny " << kny / lbox << "\n";
  std::cerr << "ikny " << ikny << " ikny2 " << ikny2 << "\n";
}

template <typename T>
int powerspec::get_k_index(T k)
{
  if(k < kmin || k >= kmax) return -1;
  if(log_scale) return floor(log(k / kmin) / logratio);
  else return floor((k - kmin) / lin_dk);
  return -1;
}

void powerspec::check_p() const
{
  std::cout << "Exponent of the window function used for aliasing removal p = ";
  switch(p) {
  case 1:
    std::cout << "1 (NGP)" << std::endl;
    break;
  case 2:
    std::cout << "2 (CIC)" << std::endl;
    break;
  case 3:
    std::cout << "3 (TSC)" << std::endl;
    break;
  default:
    std::cerr << "Error: p is UNKNOWN (value = " << p << ")" << std::endl;
    exit(EXIT_FAILURE);
  }
}

template <typename T>
void powerspec::output_pk(T &power, T &weight, std::string filename)
{
  std::ofstream fout(filename);
  fout << "# k[h/Mpc] pk weight" << std::endl;
  fout << std::scientific << std::setprecision(10);

  double pk_box_norm = lbox * lbox * lbox;

  for(int32_t ik = 1; ik < nk; ik++) {
    double kk = kcen[ik] * dk / lbox;
    if(kk < kny * dk / lbox) {
      double pk = power[ik] * pk_box_norm;
      fout << kk << " " << pk << " " << weight[ik] << "\n";
    }
  }
  fout.flush();
  fout.close();
  std::cout << "output to " << filename << std::endl;
}

template <typename T, typename TT>
void powerspec::output_pk_ell(TT &multipole_power, T &weight, std::string filename)
{
  std::ofstream fout(filename);
  fout << "# k[h/Mpc] pk_l_0 pk_l_1 pk_l_2 ... weight" << std::endl;
  fout << std::scientific << std::setprecision(10);

  double pk_box_norm = lbox * lbox * lbox;

  for(int32_t ik = 1; ik < nk; ik++) {
    double kk = kcen[ik] * dk / lbox;
    if(kk < kny * dk / lbox) {
      fout << kk;
      for(int ell = 0; ell < ellmax; ell++) {
        double mp_pk = multipole_power[ell][ik] * pk_box_norm;
        fout << " " << mp_pk;
      }
      fout << " " << weight[ik] << "\n";
    }
  }
  fout.flush();
  fout.close();
  std::cout << "output to " << filename << std::endl;
}

template <typename T>
void powerspec::set_shotnoise(const T ngrp, const double vol)
{
  if(shotnoise_corr) {
    auto norm = (double)nmesh * (double)nmesh * (double)nmesh; // N^3
    norm *= norm;                                              // N^6
    shotnoise = vol / (double)ngrp;
    shotnoise *= norm;
  } else {
    shotnoise = 0.0;
  }
}

float powerspec::calc_window(int64_t ix, int64_t iy, int64_t iz)
{
  const float kx = (ix < nmesh / 2) ? (float)(ix) : (float)(nmesh - ix);
  const float ky = (iy < nmesh / 2) ? (float)(iy) : (float)(nmesh - iy);
  const float kz = (float)(iz);

  float d = pi / nmesh;
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

  float win2 = (p == 0) ? 1.0 : wx * wy * wz;
  return win2;
}

template <typename T>
void powerspec::calc_power_spec(T &mesh, T &power, T &weight)
{
  static_assert(!std::is_same<T, float>::value, "Only float is allowed");

  check_p();

  static bool fft_init = false;
  if(fft_init == false) {
    fftwf_init_threads();
    fftwf_plan_with_nthreads(omp_get_max_threads());
    fft_init = true;
  }

  double pk_norm = (double)(nmesh * nmesh) * (double)(nmesh * nmesh) * (double)(nmesh * nmesh);
  // int nk = (int)(nmesh);

  power.assign(nk, 0.0);
  weight.assign(nk, 0.0);

  fftwf_plan plan;
  plan = fftwf_plan_dft_r2c_3d(nmesh, nmesh, nmesh, mesh.data(), (fftwf_complex *)mesh.data(), FFTW_ESTIMATE);
  fftwf_execute(plan);

  fftwf_complex *mesh_hat = (fftwf_complex *)mesh.data();

// #pragma omp parallel for collapse(3) reduction(+ : power[ : nmesh], weight[ : nmesh])
#pragma omp parallel for collapse(3) reduction(vec_float_plus : power, weight)
  for(uint64_t ix = 0; ix < nmesh; ix++) {
    for(uint64_t iy = 0; iy < nmesh; iy++) {
      for(uint64_t iz = 0; iz < nmesh / 2 + 1; iz++) {
        const float kx = (ix < nmesh / 2) ? (float)(ix) : (float)(nmesh - ix);
        const float ky = (iy < nmesh / 2) ? (float)(iy) : (float)(nmesh - iy);
        const float kz = (float)(iz);

        auto k = sqrt(SQR(kx) + SQR(ky) + SQR(kz));
        int32_t ik = get_k_index(k);
        int64_t im = iz + (nmesh / 2 + 1) * (iy + nmesh * ix);
        auto win2 = calc_window(ix, iy, iz);

        if(ik > 0 && ik < nk) {
          auto tmp_power = SQR(c_re(mesh_hat[im])) + SQR(c_im(mesh_hat[im]));
          power[ik] += (tmp_power / win2) - shotnoise;
          weight[ik] += 1.0;
        }
      }
    }
  } // ix,iy,iz loop

#pragma omp parallel for
  for(int ik = 0; ik < nk; ik++) {
    power[ik] /= (weight[ik] + 1e-20);
    power[ik] /= pk_norm;
        }

  fftwf_destroy_plan(plan);
}

template <typename T>
void powerspec::calc_power_spec(T &mesh1, T &mesh2, T &power, T &weight)
{
  static_assert(!std::is_same<T, float>::value, "Only float is allowed");

  if(mesh1.data() == mesh2.data()) {
    calc_power_spec(mesh1, power, weight);
    return;
  }

  assert(mesh1.size() == mesh2.size());

  check_p();

  static bool fft_init = false;
  if(fft_init == false) {
    fftwf_init_threads();
    fftwf_plan_with_nthreads(omp_get_max_threads());
    fft_init = true;
  }

  double pk_norm = (double)(nmesh * nmesh) * (double)(nmesh * nmesh) * (double)(nmesh * nmesh);
  // int nk = (int)(nmesh);

  power.assign(nk, 0.0);
  weight.assign(nk, 0.0);

  fftwf_plan plan;
  plan = fftwf_plan_dft_r2c_3d(nmesh, nmesh, nmesh, mesh1.data(), (fftwf_complex *)mesh1.data(), FFTW_ESTIMATE);
  fftwf_execute_dft_r2c(plan, mesh1.data(), (fftwf_complex *)(mesh1.data()));
  fftwf_execute_dft_r2c(plan, mesh2.data(), (fftwf_complex *)(mesh2.data()));

  fftwf_complex *mesh_hat1 = (fftwf_complex *)mesh1.data();
  fftwf_complex *mesh_hat2 = (fftwf_complex *)mesh2.data();

// #pragma omp parallel for collapse(3) reduction(+ : power[ : nmesh], weight[ : nmesh])
#pragma omp parallel for collapse(3) reduction(vec_float_plus : power, weight)
  for(uint64_t ix = 0; ix < nmesh; ix++) {
    for(uint64_t iy = 0; iy < nmesh; iy++) {
      for(uint64_t iz = 0; iz < nmesh / 2 + 1; iz++) {

        const float kx = (ix < nmesh / 2) ? (float)(ix) : (float)(nmesh - ix);
        const float ky = (iy < nmesh / 2) ? (float)(iy) : (float)(nmesh - iy);
        const float kz = (float)(iz);

        auto k = sqrt(SQR(kx) + SQR(ky) + SQR(kz));
        int32_t ik = get_k_index(k);
        int64_t im = iz + (nmesh / 2 + 1) * (iy + nmesh * ix);
        auto win2 = calc_window(ix, iy, iz);

        if(ik > 0 && ik < nk) {
          auto tmp_power = (c_re(mesh_hat1[im]) * c_re(mesh_hat2[im])) + (c_im(mesh_hat1[im]) * c_im(mesh_hat2[im]));
          power[ik] += tmp_power / win2;
          weight[ik] += 1.0;
        }
      }
    }
  } // ix,iy,iz loop

#pragma omp parallel for
  for(int ik = 0; ik < nk; ik++) {
    power[ik] /= (weight[ik] + 1e-20);
    power[ik] /= pk_norm;
  }

  fftwf_destroy_plan(plan);
}

template <typename T, typename TT>
void powerspec::calc_power_spec_ell(T &mesh, TT &multipole_power, T &weight)
{
  static_assert(!std::is_same<T, float>::value, "Only float is allowed");

  check_p();

  static bool fft_init = false;
  if(fft_init == false) {
    fftwf_init_threads();
    fftwf_plan_with_nthreads(omp_get_max_threads());
    fft_init = true;
  }

  double pk_norm = (double)(nmesh * nmesh) * (double)(nmesh * nmesh) * (double)(nmesh * nmesh);
  // int nk = (int)(nmesh);

  weight.assign(nk, 0.0);
  multipole_power.resize(ellmax, T(nk));

  fftwf_plan plan;
  plan = fftwf_plan_dft_r2c_3d(nmesh, nmesh, nmesh, mesh.data(), (fftwf_complex *)mesh.data(), FFTW_ESTIMATE);
  fftwf_execute(plan);

  fftwf_complex *mesh_hat = (fftwf_complex *)mesh.data();

  for(int ell = 0; ell < ellmax; ell++) {
    T power_ell;
    power_ell.assign(nk, 0.0);

    std::cout << "# calc pk ell=" << ell << std::endl;
    std::fill(weight.begin(), weight.end(), 0.0); // set effective values in the last loop
    std::fill(multipole_power[ell].begin(), multipole_power[ell].end(), 0.0);

// #pragma omp parallel for collapse(3) reduction(vec_float_plus : multipole_power[l], weight)
#pragma omp parallel for collapse(3) reduction(vec_float_plus : power_ell, weight)
    for(uint64_t ix = 0; ix < nmesh; ix++) {
      for(uint64_t iy = 0; iy < nmesh; iy++) {
        for(uint64_t iz = 0; iz < nmesh / 2 + 1; iz++) {

          const float kx = (ix < nmesh / 2) ? (float)(ix) : (float)(nmesh - ix);
          const float ky = (iy < nmesh / 2) ? (float)(iy) : (float)(nmesh - iy);
          const float kz = (float)(iz);

          double k = sqrt(SQR(kx) + SQR(ky) + SQR(kz));
          double mu = (double)kz / k;

          int32_t ik = get_k_index(k);
          int64_t im = iz + (nmesh / 2 + 1) * (iy + nmesh * ix);
          auto win2 = calc_window(ix, iy, iz);

          double tmp_power = SQR(c_re(mesh_hat[im])) + SQR(c_im(mesh_hat[im]));
          tmp_power = (tmp_power / win2) - shotnoise;

          if(ik > 0 && ik < nk) {
            power_ell[ik] += tmp_power * std::legendre(ell, mu);
            weight[ik] += 1.0;
          }
        }
      }
    } // ix,iy,iz-loop

#pragma omp parallel for
    for(int ik = 0; ik < nk; ik++) {
      multipole_power[ell][ik] = power_ell[ik] / (weight[ik] + 1e-20);
      multipole_power[ell][ik] /= pk_norm;
      multipole_power[ell][ik] *= (2.0 * ell + 1.0); // Multiply by 2 since the integral range of mu is not [-1,1] but
                                                     // [0,1] (2.0 * ell + 1.0)/2.0 * 2.0
    }
  } // l-loop

  fftwf_destroy_plan(plan);
}

template <typename T, typename TT>
void powerspec::calc_power_spec_ell(T &mesh1, T &mesh2, TT &multipole_power, T &weight)
{
  static_assert(!std::is_same<T, float>::value, "Only float is allowed");

  if(mesh1.data() == mesh2.data()) {
    calc_power_spec_ell(mesh1, multipole_power, weight);
    return;
  }

  assert(mesh1.size() == mesh2.size());

  check_p();

  static bool fft_init = false;
  if(fft_init == false) {
    fftwf_init_threads();
    fftwf_plan_with_nthreads(omp_get_max_threads());
    fft_init = true;
  }

  double pk_norm = (double)(nmesh * nmesh) * (double)(nmesh * nmesh) * (double)(nmesh * nmesh);
  // int nk = (int)(nmesh);

  weight.assign(nk, 0.0);
  multipole_power.resize(ellmax, T(nk));

  fftwf_plan plan;
  plan = fftwf_plan_dft_r2c_3d(nmesh, nmesh, nmesh, mesh1.data(), (fftwf_complex *)mesh1.data(), FFTW_ESTIMATE);
  fftwf_execute_dft_r2c(plan, mesh1.data(), (fftwf_complex *)(mesh1.data()));
  fftwf_execute_dft_r2c(plan, mesh2.data(), (fftwf_complex *)(mesh2.data()));

  fftwf_complex *mesh_hat1 = (fftwf_complex *)mesh1.data();
  fftwf_complex *mesh_hat2 = (fftwf_complex *)mesh2.data();

  for(int ell = 0; ell < ellmax; ell++) {
    T power_ell;
    power_ell.assign(nk, 0.0);

    std::cout << "# calc pk ell=" << ell << std::endl;
    std::fill(weight.begin(), weight.end(), 0.0); // set effective values in the last loop
    std::fill(multipole_power[ell].begin(), multipole_power[ell].end(), 0.0);

// #pragma omp parallel for collapse(3) reduction(vec_float_plus : multipole_power[l], weight)
#pragma omp parallel for collapse(3) reduction(vec_float_plus : power_ell, weight)
  for(uint64_t ix = 0; ix < nmesh; ix++) {
    for(uint64_t iy = 0; iy < nmesh; iy++) {
      for(uint64_t iz = 0; iz < nmesh / 2 + 1; iz++) {

        const float kx = (ix < nmesh / 2) ? (float)(ix) : (float)(nmesh - ix);
        const float ky = (iy < nmesh / 2) ? (float)(iy) : (float)(nmesh - iy);
        const float kz = (float)(iz);

        double k = sqrt(SQR(kx) + SQR(ky) + SQR(kz));
          double mu = (double)kz / k;

          // int32_t ik = (int)(k);
        int32_t ik = get_k_index(k);
        int64_t im = iz + (nmesh / 2 + 1) * (iy + nmesh * ix);
          auto win2 = calc_window(ix, iy, iz);

          auto tmp_power = (c_re(mesh_hat1[im]) * c_re(mesh_hat2[im])) + (c_im(mesh_hat1[im]) * c_im(mesh_hat2[im]));
          tmp_power /= win2;

        if(ik > 0 && ik < nk) {
            power_ell[ik] += tmp_power * std::legendre(ell, mu);
          weight[ik] += 1.0;
        }
      }
    }
    } // ix,iy,iz-loop

#pragma omp parallel for
  for(int ik = 0; ik < nk; ik++) {
      multipole_power[ell][ik] = power_ell[ik] / (weight[ik] + 1e-20);
      multipole_power[ell][ik] /= pk_norm;
      multipole_power[ell][ik] *= (2.0 * ell + 1.0); // Multiply by 2 since the integral range of mu is not [-1,1] but
                                                     // [0,1] (2.0 * ell + 1.0)/2.0 * 2.0
  }
  } // l-loop

  fftwf_destroy_plan(plan);
  }
