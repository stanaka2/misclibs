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

  void set_kbin(double, double, int, bool);
  void check_kbin();
  void check_p() const;

  template <typename T>
  int get_k_index(T);

  template <typename T>
  void calc_power_spec(T &, T &, T &);
  template <typename T, typename TT>
  void calc_power_spec_ell(T &, TT &, T &);

  template <typename T>
  double fit_power(T &);
  double W(double *, double);
  double C2_correct_factor(double, double, int, double);
  template <typename T>
  void correct_aliasing_pk(T &r, int);

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

/* 3D pk filtering */
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

        // int32_t ik = (int)(sqrt(SQR(kx) + SQR(ky) + SQR(kz)));
        auto k = sqrt(SQR(kx) + SQR(ky) + SQR(kz));
        int32_t ik = get_k_index(k);
        int64_t im = iz + (nmesh / 2 + 1) * (iy + nmesh * ix);

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

        if(ik > 0 && ik < nk) {
          auto tmp_power = SQR(c_re(mesh_hat[im])) + SQR(c_im(mesh_hat[im]));
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

          double k = sqrt(SQR(kx) + SQR(ky) + SQR(kz));
          double mu = (double)kz / k;

          // int32_t ik = (int)(k);
          int32_t ik = get_k_index(k);
          int64_t im = iz + (nmesh / 2 + 1) * (iy + nmesh * ix);

          double tmp_power = SQR(c_re(mesh_hat[im])) + SQR(c_im(mesh_hat[im]));
          tmp_power /= win2;

          if(ik > 0 && ik < nk) {
            // multipole_power[l][ik] += tmp_power * std::legendre(ell, mu);
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
}

#if 0
/* 1D pk filtering */
template <typename T>
void powerspec::calc_power_spec_1d_corrction(T &mesh, T &power, T &weight)
{
  static bool fft_init = false;
  if(fft_init == false) {
    fftwf_init_threads();
    fftwf_plan_with_nthreads(omp_get_max_threads());
    fft_init = true;
  }

  double pk_norm = (double)(nmesh * nmesh) * (double)(nmesh * nmesh) * (double)(nmesh * nmesh);
  // int nk = (int)(nmesh);

  power.assign(nk, 0.0);
  weight.assign(nk, 0);


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

        // int32_t ik = (int)(sqrt(SQR(kx) + SQR(ky) + SQR(kz)));
        double k = sqrt(SQR(kx) + SQR(ky) + SQR(kz));
        int32_t ik = get_k_index(k);
        int64_t im = iz + (nmesh / 2 + 1) * (iy + nmesh * ix);

        if(ik > 0 && ik < nk) {
          power[ik] += SQR(c_re(mesh_hat[im])) + SQR(c_im(mesh_hat[im]));
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

  if(p > 0) {
    std::cout << "# correct aliasing pk" << std::endl;
    correct_aliasing_pk(power, nmesh);
  }
}

template <typename T>
double powerspec::fit_power(T &pk)
{
  // evaluate the power-law slope between k_ny/2 < k < k_ny
  double alpha = log(pk[ikny] / pk[ikny2]) / log(kny / kny2);
  return alpha;
}

double powerspec::W(double *kvec, double k_ny)
{
  double k[3];
  k[0] = pi * kvec[0] / (2.0 * k_ny);
  k[1] = pi * kvec[1] / (2.0 * k_ny);
  k[2] = pi * kvec[2] / (2.0 * k_ny);

  double wk = sin(k[0]) * sin(k[1]) * sin(k[2]) / (k[0] * k[1] * k[2]);

  if(p == 1) return (wk);
  else if(p == 2) return (wk * wk);
  else if(p == 3) return (wk * wk * wk);
  else if(p == 4) return (wk * wk * wk * wk);
  else std::exit(EXIT_FAILURE);
}

double powerspec::C2_correct_factor(double k, double k_ny, int nmax, double alpha)
{
  double kvec[3];
  int nvec[3];
  double sum;
  kvec[0] = kvec[1] = kvec[2] = k / sqrt(3.0);

  sum = 0.0;

#pragma omp parellel for collapse(3) reduction(+ : sum)
  for(nvec[0] = -nmax; nvec[0] <= nmax; nvec[0]++) {
    for(nvec[1] = -nmax; nvec[1] <= nmax; nvec[1]++) {
      for(nvec[2] = -nmax; nvec[2] <= nmax; nvec[2]++) {
        double knvec[3];
        knvec[0] = kvec[0] + 2.0 * k_ny * nvec[0];
        knvec[1] = kvec[1] + 2.0 * k_ny * nvec[1];
        knvec[2] = kvec[2] + 2.0 * k_ny * nvec[2];

        double kn = sqrt(knvec[0] * knvec[0] + knvec[1] * knvec[1] + knvec[2] * knvec[2]);
        sum += SQR(W(knvec, k_ny)) * pow(kn, alpha);
      }
    }
  }

  double kn = sqrt(kvec[0] * kvec[0] + kvec[1] * kvec[1] + kvec[2] * kvec[2]);
  return (sum / pow(kn, alpha));
}

template <typename T>
void powerspec::correct_aliasing_pk(T &power, int nmesh)
{
  double k_nyquist = kny;

  T pk_cor(nk);

  double alpha, alpha_new;
  alpha = fit_power(power);
  alpha_new = alpha;
  do {
    alpha = alpha_new;

#pragma omp parellel for
    for(int ik = 0; ik < nk; ik++) {
      double k = kcen[ik];
      pk_cor[ik] = power[ik] / C2_correct_factor(k, k_nyquist, 2, alpha);
    }

    alpha_new = fit_power(pk_cor);
  } while(fabs(alpha - alpha_new) > 1.0e-5);

  power = pk_cor;
}

#endif
