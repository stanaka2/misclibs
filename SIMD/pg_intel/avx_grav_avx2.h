#pragma once

#include <immintrin.h>
#include "pppm.h"

#undef ALIGNE_SIZE
#define ALIGNE_SIZE (64)
#define NVECT (8) // 4 doubles × 2

// newton raphson
static inline __m256 _mm256_rsqrt_nr1_ps(const __m256 rsq)
{
  const __m256 half = _mm256_set1_ps(0.5f);
  const __m256 three = _mm256_set1_ps(3.0f);
  __m256 y = _mm256_rsqrt_ps(rsq);                    // approximate rsqrt
  __m256 y2 = _mm256_mul_ps(y, y);                    // y^2
  __m256 corr = _mm256_mul_ps(rsq, y2);               // rsq * y^2
  corr = _mm256_sub_ps(three, corr);                  // 3 - rsq*y^2
  return _mm256_mul_ps(half, _mm256_mul_ps(y, corr)); // 0.5*y*(3-rsq*y^2)
}

static inline __m256 _mm256_gfactor_S2_ps(const __m256 _rad, const __m256 _eps_pm)
{
  __m256 _R, _g, _h, _S;
  __m256 _zero = _mm256_set1_ps(0.0f);
  __m256 _one = _mm256_set1_ps(1.0f);
  __m256 _two = _mm256_set1_ps(2.0f);

  // _R = 2 r / eps_pm, clamp to [0,2]
  _R = _mm256_div_ps(_rad, _eps_pm);
  _R = _mm256_mul_ps(_R, _two);
  _R = _mm256_min_ps(_R, _two);

  // _S = max(_R - 1, 0)^6
  _S = _mm256_sub_ps(_R, _one);
  _S = _mm256_max_ps(_S, _zero);
  _S = _mm256_mul_ps(_mm256_mul_ps(_S, _S), _S); // S^3
  _S = _mm256_mul_ps(_S, _S);                    // S^6

  __m256 _coeff0 = _mm256_set1_ps(0.15f);
  __m256 _coeff1 = _mm256_set1_ps(12.0f / 35.0f);
  __m256 _coeff2 = _mm256_set1_ps(-0.5f);
  __m256 _coeff3 = _mm256_set1_ps(1.6f);

  // _g(_R)
  _g = _mm256_fmsub_ps(_coeff0, _R, _coeff1);
  _g = _mm256_fmadd_ps(_g, _R, _coeff2);
  _g = _mm256_fmadd_ps(_g, _R, _coeff3);
  _g = _mm256_mul_ps(_g, _R);
  _g = _mm256_fmsub_ps(_g, _R, _coeff3);
  _g = _mm256_mul_ps(_g, _R);
  _g = _mm256_mul_ps(_g, _R);
  _g = _mm256_mul_ps(_g, _R);

  // _h(_R, _S)
  _coeff0 = _mm256_set1_ps(0.2f);
  _coeff1 = _mm256_set1_ps(18.0f / 35.0f);
  _coeff2 = _mm256_set1_ps(3.0f / 35.0f);

  _h = _mm256_fmadd_ps(_coeff0, _R, _coeff1);
  _h = _mm256_fmadd_ps(_h, _R, _coeff2);
  _h = _mm256_mul_ps(_h, _S);

  _g = _mm256_add_ps(_one, _g);
  _g = _mm256_sub_ps(_g, _h);

  return _g;
}

static inline void calc_grav_avx2(double (*xi)[3], double (*ai)[NVECT], double *mj, double (*xj)[3], int njlist,
                                  double eps2_pp, double eps_pm)
{
  const __m256 _eps_pm = _mm256_set1_ps((float)eps_pm);
  const __m256 _eps2_0 = _mm256_set1_ps((float)eps2_pp);

  __m256d _xi_0 = _mm256_set_pd(xi[3][0], xi[2][0], xi[1][0], xi[0][0]);
  __m256d _yi_0 = _mm256_set_pd(xi[3][1], xi[2][1], xi[1][1], xi[0][1]);
  __m256d _zi_0 = _mm256_set_pd(xi[3][2], xi[2][2], xi[1][2], xi[0][2]);

  __m256d _xi_1 = _mm256_set_pd(xi[7][0], xi[6][0], xi[5][0], xi[4][0]);
  __m256d _yi_1 = _mm256_set_pd(xi[7][1], xi[6][1], xi[5][1], xi[4][1]);
  __m256d _zi_1 = _mm256_set_pd(xi[7][2], xi[6][2], xi[5][2], xi[4][2]);

  __m256d _xacc_0 = _mm256_setzero_pd();
  __m256d _yacc_0 = _mm256_setzero_pd();
  __m256d _zacc_0 = _mm256_setzero_pd();

  __m256d _xacc_1 = _mm256_setzero_pd();
  __m256d _yacc_1 = _mm256_setzero_pd();
  __m256d _zacc_1 = _mm256_setzero_pd();

  for(int j = 0; j < njlist; j++) {
    __m256d _xj = _mm256_set1_pd(xj[j][0]);
    __m256d _yj = _mm256_set1_pd(xj[j][1]);
    __m256d _zj = _mm256_set1_pd(xj[j][2]);

    __m256 _mj = _mm256_set1_ps((float)mj[j]);

    // dx, dy, dz in double
    __m256d _dx_0 = _mm256_sub_pd(_xj, _xi_0);
    __m256d _dy_0 = _mm256_sub_pd(_yj, _yi_0);
    __m256d _dz_0 = _mm256_sub_pd(_zj, _zi_0);

    __m256d _dx_1 = _mm256_sub_pd(_xj, _xi_1);
    __m256d _dy_1 = _mm256_sub_pd(_yj, _yi_1);
    __m256d _dz_1 = _mm256_sub_pd(_zj, _zi_1);

    // 8 doubles -> 8 floats (dx,dy,dz)
    __m128 _dx0_ps = _mm256_cvtpd_ps(_dx_0);
    __m128 _dx1_ps = _mm256_cvtpd_ps(_dx_1);
    __m256 _dx = _mm256_castps128_ps256(_dx0_ps);
    _dx = _mm256_insertf128_ps(_dx, _dx1_ps, 1);

    __m128 _dy0_ps = _mm256_cvtpd_ps(_dy_0);
    __m128 _dy1_ps = _mm256_cvtpd_ps(_dy_1);
    __m256 _dy = _mm256_castps128_ps256(_dy0_ps);
    _dy = _mm256_insertf128_ps(_dy, _dy1_ps, 1);

    __m128 _dz0_ps = _mm256_cvtpd_ps(_dz_0);
    __m128 _dz1_ps = _mm256_cvtpd_ps(_dz_1);
    __m256 _dz = _mm256_castps128_ps256(_dz0_ps);
    _dz = _mm256_insertf128_ps(_dz, _dz1_ps, 1);

    // rad2, rsq
    __m256 _rad2 = _mm256_mul_ps(_dx, _dx);
    _rad2 = _mm256_fmadd_ps(_dy, _dy, _rad2);
    _rad2 = _mm256_fmadd_ps(_dz, _dz, _rad2);
    __m256 _rsq = _mm256_add_ps(_rad2, _eps2_0);

    __m256 _rinv = _mm256_rsqrt_nr1_ps(_rsq);
    //__m256 _rinv = _mm256_rsqrt_ps(_rsq);

    __m256 _rad = _mm256_sqrt_ps(_rad2);
    __m256 _gfact = _mm256_gfactor_S2_ps(_rad, _eps_pm);

    __m256 _rinv2 = _mm256_mul_ps(_rinv, _rinv);
    __m256 _rinv3 = _mm256_mul_ps(_rinv2, _rinv);
    __m256 _mrinv3 = _mm256_mul_ps(_mj, _rinv3);
    _mrinv3 = _mm256_mul_ps(_mrinv3, _gfact);

    // 8 floats -> 2×4 doubles
    __m128 _m0_ps = _mm256_castps256_ps128(_mrinv3);
    __m128 _m1_ps = _mm256_extractf128_ps(_mrinv3, 1);
    __m256d _mrinv3_0 = _mm256_cvtps_pd(_m0_ps);
    __m256d _mrinv3_1 = _mm256_cvtps_pd(_m1_ps);

    _xacc_0 = _mm256_fmadd_pd(_mrinv3_0, _dx_0, _xacc_0);
    _yacc_0 = _mm256_fmadd_pd(_mrinv3_0, _dy_0, _yacc_0);
    _zacc_0 = _mm256_fmadd_pd(_mrinv3_0, _dz_0, _zacc_0);

    _xacc_1 = _mm256_fmadd_pd(_mrinv3_1, _dx_1, _xacc_1);
    _yacc_1 = _mm256_fmadd_pd(_mrinv3_1, _dy_1, _yacc_1);
    _zacc_1 = _mm256_fmadd_pd(_mrinv3_1, _dz_1, _zacc_1);
  }

  _mm256_store_pd(&ai[0][0], _xacc_0);
  _mm256_store_pd(&ai[0][4], _xacc_1);
  _mm256_store_pd(&ai[1][0], _yacc_0);
  _mm256_store_pd(&ai[1][4], _yacc_1);
  _mm256_store_pd(&ai[2][0], _zacc_0);
  _mm256_store_pd(&ai[2][4], _zacc_1);
}

void calc_grav_simd_thread(EPItreepm *iptcl, Result_treepm *ppforce, const int ni, double (*xj)[3], double *mj,
                           const int nj, double eps2_pp, double eps_pm)
{
  double xii[NVECT][3] __attribute__((aligned(ALIGNE_SIZE)));
  double aii[3][NVECT] __attribute__((aligned(ALIGNE_SIZE)));

  for(int i = 0; i < ni; i += NVECT) {
    int nn = NVECT;
    if((i + NVECT) > ni) nn = ni - i;

    for(int ii = 0; ii < nn; ii++) {
      int iii = i + ii;
      xii[ii][0] = iptcl[iii].pos[0];
      xii[ii][1] = iptcl[iii].pos[1];
      xii[ii][2] = iptcl[iii].pos[2];
    }

    for(int ii = nn; ii < NVECT; ii++) {
      xii[ii][0] = xii[nn - 1][0];
      xii[ii][1] = xii[nn - 1][1];
      xii[ii][2] = xii[nn - 1][2];
    }

    calc_grav_avx2(xii, aii, mj, xj, nj, eps2_pp, eps_pm);

    for(int ii = 0; ii < nn; ii++) {
      int iii = i + ii;
      ppforce[iii].acc[0] += aii[0][ii];
      ppforce[iii].acc[1] += aii[1][ii];
      ppforce[iii].acc[2] += aii[2][ii];
    }
  }
}
