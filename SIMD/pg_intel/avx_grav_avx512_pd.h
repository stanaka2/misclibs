#pragma once

#include <x86intrin.h>
#include "pppm.h"

#undef ALIGNE_SIZE

#define ALIGNE_SIZE (64)
#define NVECT (8) // 8 iptcls * 1

static inline __m512d _mm512_gfactor_S2_pd(const __m512d _rad, const __m512d _eps_pm)
{
  __m512d _R, _g, _h, _S;
  const __m512d _zero = _mm512_set1_pd(0.0);
  const __m512d _one = _mm512_set1_pd(1.0);
  const __m512d _two = _mm512_set1_pd(2.0);

  // R = 2 r / eps_pm, clipped to [0, 2]
  _R = _mm512_div_pd(_rad, _eps_pm);
  _R = _mm512_mul_pd(_R, _two);
  _R = _mm512_min_pd(_R, _two);

  // S = max(R - 1, 0)^6
  _S = _mm512_sub_pd(_R, _one);
  _S = _mm512_max_pd(_S, _zero);
  _S = _mm512_mul_pd(_mm512_mul_pd(_S, _S), _S); // S^3
  _S = _mm512_mul_pd(_S, _S);                    // S^6

  __m512d _coeff0 = _mm512_set1_pd(0.15); // 3/20
  __m512d _coeff1 = _mm512_set1_pd(12.0 / 35.0);
  __m512d _coeff2 = _mm512_set1_pd(-0.5);
  __m512d _coeff3 = _mm512_set1_pd(1.6);

  // g(R) part
  _g = _mm512_fmsub_pd(_coeff0, _R, _coeff1);
  _g = _mm512_fmadd_pd(_g, _R, _coeff2);
  _g = _mm512_fmadd_pd(_g, _R, _coeff3);
  _g = _mm512_mul_pd(_g, _R);
  _g = _mm512_fmsub_pd(_g, _R, _coeff3);
  _g = _mm512_mul_pd(_g, _R);
  _g = _mm512_mul_pd(_g, _R);
  _g = _mm512_mul_pd(_g, _R);

  // h(R) part
  _coeff0 = _mm512_set1_pd(0.2); // 1/5
  _coeff1 = _mm512_set1_pd(18.0 / 35.0);
  _coeff2 = _mm512_set1_pd(3.0 / 35.0);

  _h = _mm512_fmadd_pd(_coeff0, _R, _coeff1);
  _h = _mm512_fmadd_pd(_h, _R, _coeff2);
  _h = _mm512_mul_pd(_h, _S);

  _g = _mm512_add_pd(_one, _g);
  _g = _mm512_sub_pd(_g, _h);

  return _g;
}

static inline __m512d _mm512_gfactor_pot_S2_pd(__m512d _rad, const __m512d _eps_pm)
{
  __m512d _R, _R2;
  const __m512d _zero = _mm512_set1_pd(0.0);
  const __m512d _one = _mm512_set1_pd(1.0);
  const __m512d _two = _mm512_set1_pd(2.0);

  _rad = _mm512_min_pd(_rad, _eps_pm); // r = min(r, sft_pm)
  _R = _mm512_div_pd(_rad, _eps_pm);
  _R = _mm512_mul_pd(_R, _two);
  _R2 = _mm512_mul_pd(_R, _R);

  const __m512d _c4 = _mm512_set1_pd(4.0);
  const __m512d _c8 = _mm512_set1_pd(8.0);
  const __m512d _c14 = _mm512_set1_pd(14.0);
  __m512d _f = _mm512_div_pd(_one, _c14); // f2 = 1/14

  // g1 branch (R <= 1)
  __m512d _g1 = _mm512_fmsub_pd(_R, _mm512_set1_pd(3.0), _c8);                 // -8 + 3R
  _g1 = _mm512_fmsub_pd(_R, _mm512_mul_pd(_f, _g1), _one);                     // -1 + R(f2*g1)
  _g1 = _mm512_fmadd_pd(_R, _g1, _c4);                                         // 4 + R*g1
  _g1 = _mm512_fmsub_pd(_R2, _g1, _c8);                                        // -8 + R2*g1
  _g1 = _mm512_fmadd_pd(_R2, _mm512_mul_pd(_c14, _g1), _mm512_set1_pd(208.0)); // 208 + R2*(14*g1)

  // g2 branch (R > 1)
  __m512d _g2 = _mm512_mul_pd(_R, _mm512_sub_pd(_c8, _R));                    // R(8-R)
  _g2 = _mm512_fmsub_pd(_f, _g2, _one);                                       // -1 + f2*g2
  _g2 = _mm512_fmsub_pd(_R, _g2, _c4);                                        // -4 + R*g2
  _g2 = _mm512_fmadd_pd(_R, _g2, _mm512_set1_pd(20.0));                       // 20 + R*g2
  _g2 = _mm512_fmsub_pd(_R, _g2, _mm512_set1_pd(32.0));                       // -32 + R*g2
  _g2 = _mm512_fmadd_pd(_R, _g2, _mm512_set1_pd(16.0));                       // 16 + R*g2
  _g2 = _mm512_fmadd_pd(_c14, _mm512_mul_pd(_R, _g2), _mm512_set1_pd(128.0)); // 128 + 14R*g2
  _g2 = _mm512_add_pd(_g2, _mm512_div_pd(_mm512_set1_pd(12.0), _R));          // 12/R + g2

  // g = g1 (R<=1) or g2 (R>1)
  __mmask8 mRle1 = _mm512_cmp_pd_mask(_R, _one, _MM_CMPINT_LE);
  _g1 = _mm512_mask_blend_pd(mRle1, _g2, _g1);

  _f = _mm512_div_pd(_rad, _mm512_mul_pd(_mm512_set1_pd(70.0), _eps_pm)); // f = r/(70 eps_pm)
  _g1 = _mm512_fnmadd_pd(_f, _g1, _one);                                  // g = 1 - f*g1

  __mmask8 mRge2 = _mm512_cmp_pd_mask(_R, _mm512_set1_pd(1.999), _MM_CMPINT_GE);
  _g1 = _mm512_mask_blend_pd(mRge2, _g1, _zero);

  return _g1;
}

static inline void calc_grav_avx512(double (*xi)[3], double (*ai)[NVECT], double *mj, double (*xj)[3], int njlist,
                                    double eps2_pp, double eps_pm)
{
  const __m512d _eps_pm = _mm512_set1_pd(eps_pm);
  const __m512d _eps2 = _mm512_set1_pd(eps2_pp);
  const __m512d _one = _mm512_set1_pd(1.0);

  // load 8 i-particles
  __m512d _xi = _mm512_set_pd(xi[7][0], xi[6][0], xi[5][0], xi[4][0], xi[3][0], xi[2][0], xi[1][0], xi[0][0]);
  __m512d _yi = _mm512_set_pd(xi[7][1], xi[6][1], xi[5][1], xi[4][1], xi[3][1], xi[2][1], xi[1][1], xi[0][1]);
  __m512d _zi = _mm512_set_pd(xi[7][2], xi[6][2], xi[5][2], xi[4][2], xi[3][2], xi[2][2], xi[1][2], xi[0][2]);

  __m512d _xacc = _mm512_setzero_pd();
  __m512d _yacc = _mm512_setzero_pd();
  __m512d _zacc = _mm512_setzero_pd();

  for(int j = 0; j < njlist; j++) {
    const __m512d _xj = _mm512_set1_pd(xj[j][0]);
    const __m512d _yj = _mm512_set1_pd(xj[j][1]);
    const __m512d _zj = _mm512_set1_pd(xj[j][2]);
    const __m512d _mj = _mm512_set1_pd(mj[j]);

    __m512d _dx = _mm512_sub_pd(_xj, _xi);
    __m512d _dy = _mm512_sub_pd(_yj, _yi);
    __m512d _dz = _mm512_sub_pd(_zj, _zi);

    __m512d _rad2 = _mm512_mul_pd(_dx, _dx);
    _rad2 = _mm512_fmadd_pd(_dy, _dy, _rad2);
    _rad2 = _mm512_fmadd_pd(_dz, _dz, _rad2);

    __m512d _rsq = _mm512_add_pd(_rad2, _eps2);
    __m512d _rad = _mm512_sqrt_pd(_rad2);

    // softened force factor
    __m512d _gfact = _mm512_gfactor_S2_pd(_rad, _eps_pm);

    // rinv^3
    __m512d _rinv = _mm512_div_pd(_one, _mm512_sqrt_pd(_rsq));
    __m512d _rinv2 = _mm512_mul_pd(_rinv, _rinv);
    __m512d _mrinv3 = _mm512_mul_pd(_mj, _mm512_mul_pd(_rinv2, _rinv));
    _mrinv3 = _mm512_mul_pd(_mrinv3, _gfact);

    _xacc = _mm512_fmadd_pd(_mrinv3, _dx, _xacc);
    _yacc = _mm512_fmadd_pd(_mrinv3, _dy, _yacc);
    _zacc = _mm512_fmadd_pd(_mrinv3, _dz, _zacc);
  }

  _mm512_store_pd(&ai[0][0], _xacc);
  _mm512_store_pd(&ai[1][0], _yacc);
  _mm512_store_pd(&ai[2][0], _zacc);
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

    calc_grav_avx512(xii, aii, mj, xj, nj, eps2_pp, eps_pm);

    for(int ii = 0; ii < nn; ii++) {
      int iii = i + ii;
      ppforce[iii].acc[0] += aii[0][ii];
      ppforce[iii].acc[1] += aii[1][ii];
      ppforce[iii].acc[2] += aii[2][ii];
    }
  }
}

#ifdef OUTPUT_POT

/*-------------------------------------------------------------
 * Wendland C4 kernel & derivative (density use), double / 8 lanes
 *------------------------------------------------------------*/
static inline __m512d _mm512_WendlandQuinticC4_pd(const __m512d _rad, const __m512d _h)
{
  const __m512d _zero = _mm512_set1_pd(0.0);
  const __m512d _one = _mm512_set1_pd(1.0);
  const __m512d _two = _mm512_set1_pd(2.0);

  __m512d _tmp1 = _mm512_mul_pd(_mm512_set1_pd(M_PI), _mm512_mul_pd(_h, _mm512_mul_pd(_h, _h)));
  __m512d _a = _mm512_div_pd(_mm512_set1_pd(495.0), _mm512_mul_pd(_mm512_set1_pd(256.0), _tmp1));
  __m512d _q = _mm512_div_pd(_rad, _h);

  _tmp1 = _mm512_fnmadd_pd(_mm512_set1_pd(0.5), _q, _one);   // 1 - 0.5 q
  _tmp1 = _mm512_mul_pd(_tmp1, _mm512_mul_pd(_tmp1, _tmp1)); // cube
  _tmp1 = _mm512_mul_pd(_tmp1, _tmp1);                       // square of cube

  __m512d _tmp3 = _mm512_set1_pd(35.0 / 12.0);
  __m512d _tmp2 = _mm512_fmadd_pd(_q, _mm512_set1_pd(3.0), _one);
  _tmp2 = _mm512_fmadd_pd(_mm512_mul_pd(_q, _q), _tmp3, _tmp2);

  _tmp1 = _mm512_mul_pd(_tmp1, _tmp2);

  // q <= 2 ? w : 0
  __mmask8 mqle2 = _mm512_cmp_pd_mask(_q, _two, _MM_CMPINT_LE);
  _tmp1 = _mm512_mask_blend_pd(mqle2, _zero, _tmp1);

  return _mm512_mul_pd(_a, _tmp1);
}

static inline __m512d _mm512_WendlandQuinticC4_dwdq_pd(const __m512d _rad, const __m512d _h)
{
  const __m512d _zero = _mm512_set1_pd(0.0);
  const __m512d _one = _mm512_set1_pd(1.0);
  const __m512d _two = _mm512_set1_pd(2.0);

  __m512d _tmp1 = _mm512_mul_pd(_mm512_set1_pd(M_PI), _mm512_mul_pd(_h, _mm512_mul_pd(_h, _h)));
  __m512d _a = _mm512_div_pd(_mm512_set1_pd(495.0), _mm512_mul_pd(_mm512_set1_pd(256.0), _tmp1));
  __m512d _q = _mm512_div_pd(_rad, _h);

  // SQR(1-0.5q) * CUBE(1-0.5q)
  _tmp1 = _mm512_fnmadd_pd(_mm512_set1_pd(0.5), _q, _one); // t = 1-0.5q
  __m512d _t2 = _mm512_mul_pd(_tmp1, _tmp1);
  __m512d _t3 = _mm512_mul_pd(_t2, _tmp1);
  _tmp1 = _mm512_mul_pd(_t2, _t3); // t^5

  __m512d _tmp3 = _mm512_fmadd_pd(_mm512_set1_pd(2.5), _q, _one); // 1+2.5q
  __m512d _tmp2 = _mm512_set1_pd(-14.0 / 3.0);
  _tmp2 = _mm512_mul_pd(_mm512_mul_pd(_tmp2, _q), _tmp3); // (-14/3) q (1+2.5q)

  _tmp1 = _mm512_mul_pd(_tmp1, _tmp2);

  // q <= 2 ? dwdq : 0
  __mmask8 mqle2 = _mm512_cmp_pd_mask(_q, _two, _MM_CMPINT_LE);
  _tmp1 = _mm512_mask_blend_pd(mqle2, _zero, _tmp1);

  return _mm512_mul_pd(_a, _tmp1);
}

static inline __m512d _mm512_density_kernel_pd(const __m512d _rad, const __m512d _h)
{
  return _mm512_WendlandQuinticC4_pd(_rad, _h);
}

static inline __m512d _mm512_density_grad_kernel_pd(const __m512d _rad, const __m512d _h)
{
  // Multiply this by dr component-wise.
  __m512d _irh = _mm512_div_pd(_mm512_set1_pd(1.0), _mm512_mul_pd(_rad, _h));

  __mmask8 m = _mm512_cmp_pd_mask(_rad, _mm512_set1_pd(1.0e-30), _MM_CMPINT_GE);
  _irh = _mm512_mask_blend_pd(m, _mm512_set1_pd(0.0), _irh); // (rad>=1e-30)? irh : 0

  return _mm512_mul_pd(_mm512_WendlandQuinticC4_dwdq_pd(_rad, _h), _irh);
}

/*-------------------------------------------------------------
 * Gravity + potential + density (Wendland C4), double / 8 lanes
 *------------------------------------------------------------*/
static inline void calc_grav_avx512_with_potdens(double (*xi)[3], double (*ai)[NVECT], double *poti, double *densi,
                                                 double (*dgi)[NVECT], double *hi, double *mj, double (*xj)[3],
                                                 int njlist, double eps2_pp, double eps_pm)
{
  const __m512d _eps_pm = _mm512_set1_pd(eps_pm);
  const __m512d _eps2 = _mm512_set1_pd(eps2_pp);
  const __m512d _one = _mm512_set1_pd(1.0);

  // 8 i-particles
  __m512d _xi = _mm512_set_pd(xi[7][0], xi[6][0], xi[5][0], xi[4][0], xi[3][0], xi[2][0], xi[1][0], xi[0][0]);
  __m512d _yi = _mm512_set_pd(xi[7][1], xi[6][1], xi[5][1], xi[4][1], xi[3][1], xi[2][1], xi[1][1], xi[0][1]);
  __m512d _zi = _mm512_set_pd(xi[7][2], xi[6][2], xi[5][2], xi[4][2], xi[3][2], xi[2][2], xi[1][2], xi[0][2]);

  // smoothing length
  __m512d _hi = _mm512_load_pd(hi);
  const __m512d _hmean = _mm512_set1_pd(0.2 / (double)NPART_1D); // 0.2 x mean 1D dist

  // hi < 0 ? hmean : hi
  __mmask8 mneg = _mm512_cmp_pd_mask(_hi, _mm512_set1_pd(0.0), _CMP_LT_OQ);
  _hi = _mm512_mask_blend_pd(mneg, _hi, _hmean);

  __m512d _xacc = _mm512_setzero_pd();
  __m512d _yacc = _mm512_setzero_pd();
  __m512d _zacc = _mm512_setzero_pd();

  __m512d _pot = _mm512_setzero_pd();
  __m512d _dens = _mm512_setzero_pd();

  __m512d _xdg = _mm512_setzero_pd();
  __m512d _ydg = _mm512_setzero_pd();
  __m512d _zdg = _mm512_setzero_pd();

  for(int j = 0; j < njlist; j++) {
    const __m512d _xj = _mm512_set1_pd(xj[j][0]);
    const __m512d _yj = _mm512_set1_pd(xj[j][1]);
    const __m512d _zj = _mm512_set1_pd(xj[j][2]);
    const __m512d _mj = _mm512_set1_pd(mj[j]);

    __m512d _dx = _mm512_sub_pd(_xj, _xi);
    __m512d _dy = _mm512_sub_pd(_yj, _yi);
    __m512d _dz = _mm512_sub_pd(_zj, _zi);

    __m512d _rad2 = _mm512_mul_pd(_dx, _dx);
    _rad2 = _mm512_fmadd_pd(_dy, _dy, _rad2);
    _rad2 = _mm512_fmadd_pd(_dz, _dz, _rad2);

    __m512d _rsq = _mm512_add_pd(_rad2, _eps2);
    __m512d _rad = _mm512_sqrt_pd(_rad2);

    __m512d _rinv = _mm512_div_pd(_one, _mm512_sqrt_pd(_rsq));
    __m512d _rinv2 = _mm512_mul_pd(_rinv, _rinv);

    // force
    __m512d _gfact = _mm512_gfactor_S2_pd(_rad, _eps_pm);
    __m512d _mrinv3 = _mm512_mul_pd(_mj, _mm512_mul_pd(_rinv2, _rinv));
    _mrinv3 = _mm512_mul_pd(_mrinv3, _gfact);

    _xacc = _mm512_fmadd_pd(_mrinv3, _dx, _xacc);
    _yacc = _mm512_fmadd_pd(_mrinv3, _dy, _yacc);
    _zacc = _mm512_fmadd_pd(_mrinv3, _dz, _zacc);

    // potential
    __m512d _gpot = _mm512_gfactor_pot_S2_pd(_rad, _eps_pm);
    __mmask8 mrsq = _mm512_cmp_pd_mask(_rsq, _mm512_set1_pd(1.0e-30), _MM_CMPINT_GE);
    _gpot = _mm512_mask_blend_pd(mrsq, _mm512_set1_pd(0.0), _gpot);

    __m512d _mrinv = _mm512_mul_pd(_mj, _mm512_mul_pd(_gpot, _rinv));
    _pot = _mm512_sub_pd(_pot, _mrinv);

    // density & grad
    __m512d _wk = _mm512_density_kernel_pd(_rad, _hi);
    __m512d _wgrad = _mm512_density_grad_kernel_pd(_rad, _hi);

    __m512d _dens_j = _mm512_mul_pd(_mj, _wk);
    __m512d _densg_j = _mm512_mul_pd(_mj, _wgrad);

    _dens = _mm512_add_pd(_dens, _dens_j);

    _xdg = _mm512_fmadd_pd(_densg_j, _dx, _xdg);
    _ydg = _mm512_fmadd_pd(_densg_j, _dy, _ydg);
    _zdg = _mm512_fmadd_pd(_densg_j, _dz, _zdg);
  }

  _mm512_store_pd(&ai[0][0], _xacc);
  _mm512_store_pd(&ai[1][0], _yacc);
  _mm512_store_pd(&ai[2][0], _zacc);

  _mm512_store_pd(&poti[0], _pot);
  _mm512_store_pd(&densi[0], _dens);

  _mm512_store_pd(&dgi[0][0], _xdg);
  _mm512_store_pd(&dgi[1][0], _ydg);
  _mm512_store_pd(&dgi[2][0], _zdg);
}

void calc_grav_simd_thread_with_pot(EPItreepm *iptcl, Result_treepm *ppforce, const int ni, double (*xj)[3], double *mj,
                                    const int nj, double eps2_pp, double eps_pm)
{
  double xii[NVECT][3] __attribute__((aligned(ALIGNE_SIZE)));
  double aii[3][NVECT] __attribute__((aligned(ALIGNE_SIZE)));
  double poti[NVECT] __attribute__((aligned(ALIGNE_SIZE)));
  double densi[NVECT] __attribute__((aligned(ALIGNE_SIZE)));
  double dgi[3][NVECT] __attribute__((aligned(ALIGNE_SIZE)));
  double hi[NVECT] __attribute__((aligned(ALIGNE_SIZE)));

  for(int i = 0; i < ni; i += NVECT) {
    int nn = NVECT;
    if((i + NVECT) > ni) nn = ni - i;

    for(int ii = 0; ii < nn; ii++) {
      int iii = i + ii;
      xii[ii][0] = iptcl[iii].pos[0];
      xii[ii][1] = iptcl[iii].pos[1];
      xii[ii][2] = iptcl[iii].pos[2];
      hi[ii] = iptcl[iii].smooth_l;
    }

    calc_grav_avx512_with_potdens(xii, aii, poti, densi, dgi, hi, mj, xj, nj, eps2_pp, eps_pm);

    for(int ii = 0; ii < nn; ii++) {
      int iii = i + ii;
      ppforce[iii].acc[0] += aii[0][ii];
      ppforce[iii].acc[1] += aii[1][ii];
      ppforce[iii].acc[2] += aii[2][ii];
      ppforce[iii].pot += poti[ii];
      ppforce[iii].dens += densi[ii];
      ppforce[iii].dens_grad[0] += dgi[0][ii];
      ppforce[iii].dens_grad[1] += dgi[1][ii];
      ppforce[iii].dens_grad[2] += dgi[2][ii];
    }
  }
}
#endif
