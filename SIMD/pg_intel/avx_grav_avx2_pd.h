#pragma once

#include <immintrin.h>
#include "pppm.h"

#undef ALIGNE_SIZE
#define ALIGNE_SIZE (64)
#define NVECT (4) // 4 doubles × 1

// S2 softening g-factor (force) : double / 4 lanes
static inline __m256d _mm256_gfactor_S2_pd(const __m256d _rad, const __m256d _eps_pm)
{
  __m256d _R, _g, _h, _S;
  const __m256d _zero = _mm256_set1_pd(0.0);
  const __m256d _one = _mm256_set1_pd(1.0);
  const __m256d _two = _mm256_set1_pd(2.0);

  // R = 2 r / eps_pm, clamp to [0,2]
  _R = _mm256_div_pd(_rad, _eps_pm);
  _R = _mm256_mul_pd(_R, _two);
  _R = _mm256_min_pd(_R, _two);

  // S = max(R - 1, 0)^6
  _S = _mm256_sub_pd(_R, _one);
  _S = _mm256_max_pd(_S, _zero);
  _S = _mm256_mul_pd(_mm256_mul_pd(_S, _S), _S); // S^3
  _S = _mm256_mul_pd(_S, _S);                    // S^6

  __m256d _coeff0 = _mm256_set1_pd(0.15); // 3/20
  __m256d _coeff1 = _mm256_set1_pd(12.0 / 35.0);
  __m256d _coeff2 = _mm256_set1_pd(-0.5);
  __m256d _coeff3 = _mm256_set1_pd(1.6);

  // g(R)
  _g = _mm256_fmsub_pd(_coeff0, _R, _coeff1);
  _g = _mm256_fmadd_pd(_g, _R, _coeff2);
  _g = _mm256_fmadd_pd(_g, _R, _coeff3);
  _g = _mm256_mul_pd(_g, _R);
  _g = _mm256_fmsub_pd(_g, _R, _coeff3);
  _g = _mm256_mul_pd(_g, _R);
  _g = _mm256_mul_pd(_g, _R);
  _g = _mm256_mul_pd(_g, _R);

  // h(R,S)
  _coeff0 = _mm256_set1_pd(0.2); // 1/5
  _coeff1 = _mm256_set1_pd(18.0 / 35.0);
  _coeff2 = _mm256_set1_pd(3.0 / 35.0);

  _h = _mm256_fmadd_pd(_coeff0, _R, _coeff1);
  _h = _mm256_fmadd_pd(_h, _R, _coeff2);
  _h = _mm256_mul_pd(_h, _S);

  _g = _mm256_add_pd(_one, _g);
  _g = _mm256_sub_pd(_g, _h);

  return _g;
}

// AVX2, double only, 4 i-particles / batch
static inline void calc_grav_avx2(double (*xi)[3], double (*ai)[NVECT], double *mj, double (*xj)[3], int njlist,
                                  double eps2_pp, double eps_pm)
{
  const __m256d _eps_pm = _mm256_set1_pd(eps_pm);
  const __m256d _eps2 = _mm256_set1_pd(eps2_pp);
  const __m256d _one = _mm256_set1_pd(1.0);

  // load 4 i-particles: lane0 -> xi[0], lane1 -> xi[1], ...
  __m256d _xi = _mm256_set_pd(xi[3][0], xi[2][0], xi[1][0], xi[0][0]);
  __m256d _yi = _mm256_set_pd(xi[3][1], xi[2][1], xi[1][1], xi[0][1]);
  __m256d _zi = _mm256_set_pd(xi[3][2], xi[2][2], xi[1][2], xi[0][2]);

  __m256d _xacc = _mm256_setzero_pd();
  __m256d _yacc = _mm256_setzero_pd();
  __m256d _zacc = _mm256_setzero_pd();

  for(int j = 0; j < njlist; j++) {
    __m256d _xj = _mm256_set1_pd(xj[j][0]);
    __m256d _yj = _mm256_set1_pd(xj[j][1]);
    __m256d _zj = _mm256_set1_pd(xj[j][2]);

    __m256d _mj = _mm256_set1_pd(mj[j]);

    __m256d _dx = _mm256_sub_pd(_xj, _xi);
    __m256d _dy = _mm256_sub_pd(_yj, _yi);
    __m256d _dz = _mm256_sub_pd(_zj, _zi);

    __m256d _rad2 = _mm256_mul_pd(_dx, _dx);
    _rad2 = _mm256_fmadd_pd(_dy, _dy, _rad2);
    _rad2 = _mm256_fmadd_pd(_dz, _dz, _rad2);

    __m256d _rsq = _mm256_add_pd(_rad2, _eps2);
    __m256d _rad = _mm256_sqrt_pd(_rad2);

    __m256d _gfact = _mm256_gfactor_S2_pd(_rad, _eps_pm);

    __m256d _rinv = _mm256_div_pd(_one, _mm256_sqrt_pd(_rsq));
    __m256d _rinv2 = _mm256_mul_pd(_rinv, _rinv);
    __m256d _rinv3 = _mm256_mul_pd(_rinv2, _rinv);
    __m256d _mrinv3 = _mm256_mul_pd(_mj, _rinv3);
    _mrinv3 = _mm256_mul_pd(_mrinv3, _gfact);

    _xacc = _mm256_fmadd_pd(_mrinv3, _dx, _xacc);
    _yacc = _mm256_fmadd_pd(_mrinv3, _dy, _yacc);
    _zacc = _mm256_fmadd_pd(_mrinv3, _dz, _zacc);
  }

  _mm256_store_pd(&ai[0][0], _xacc);
  _mm256_store_pd(&ai[1][0], _yacc);
  _mm256_store_pd(&ai[2][0], _zacc);
}

void calc_grav_simd_thread(EPItreepm *iptcl, Result_treepm *ppforce, const int ni, double (*xj)[3], double *mj,
                           const int nj, double eps2_pp, double eps_pm)
{
  double xii[NVECT][3] __attribute__((aligned(ALIGNE_SIZE)));
  double aii[3][NVECT] __attribute__((aligned(ALIGNE_SIZE)));

  for(int i = 0; i < ni; i += NVECT) {
    int nn = NVECT;
    if((i + NVECT) > ni) nn = ni - i;

    // 有効な粒子だけ詰める
    for(int ii = 0; ii < nn; ii++) {
      int iii = i + ii;
      xii[ii][0] = iptcl[iii].pos[0];
      xii[ii][1] = iptcl[iii].pos[1];
      xii[ii][2] = iptcl[iii].pos[2];
    }

    // 残りレーンは最後の粒子で埋めて UB 回避（結果は後で捨てる）
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
