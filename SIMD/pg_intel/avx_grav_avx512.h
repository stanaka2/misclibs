#pragma once

#include <x86intrin.h>
#include "pppm.h"

#undef ALIGNE_SIZE

#define ALIGNE_SIZE (64)
#define NVECT (16) // 8 iptcls * 2

/* Intrinsics function */
#if defined(__INTEL_LLVM_COMPILER)

// Intel LLVM compiler defines following two intrinsic functions
// __m512d _mm512_cvtpslo_pd(__m512 src_s16)
// __m512 _mm512_cvtpd_pslo(__m512d src_d8)

static inline __m512d _mm512_cvtpshi_pd(__m512 src_s16)
{
  __m512d ret_d8;
  // src_s16[15:8] -> (YMM float [7:0]) -> ret_d8[7:0]
  ret_d8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(src_s16, 1));
  return ret_d8;
}

static inline __m512 _mm512_cvtpd_pshi_dst(__m512 dst_s16, __m512d src_d8)
{
  __mmask16 mask = 0b1111111100000000;
  __m512 ret_s16 = _mm512_cvtpd_pslo(src_d8); // __m512d DC -> __m512 00DC
  ret_s16 = _mm512_permutexvar_ps(_mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                                  ret_s16);               // __m512 00DC -> __m512 DCxx
  ret_s16 = _mm512_mask_blend_ps(mask, dst_s16, ret_s16); // __m512 xxBA +  DCxx -> DCBA
  return ret_s16;
}

static inline __m512 _mm512_cvtpd_pslo_dst(__m512 dst_s16, __m512d src_d8)
{
  __mmask16 mask = 0b0000000011111111;
  __m512 ret_s16 = _mm512_cvtpd_pslo(src_d8);             // __m512d BA -> __m512 00BA
  ret_s16 = _mm512_mask_blend_ps(mask, dst_s16, ret_s16); // __m512 DCba +  00BA -> DCBA
  return ret_s16;
}

#elif defined(__INTEL_COMPILER)

// Intel classic compiler defines following three instrinsic functions
// __m512d _mm512_cvtpslo_pd(__m512 src_s16)
// __m512 _mm512_cvtpd_pslo(__m512d src_d8)
// __m512 _mm512_permute4f128_ps()

static inline __m512d _mm512_cvtpshi_pd(__m512 src_s16)
{
  __m512d ret_d8;
  src_s16 = _mm512_permute4f128_ps(src_s16, _MM_PERM_DCDC); // __m512 DCBA -> __m512 DCDC
  ret_d8 = _mm512_cvtpslo_pd(src_s16);                      // __m512 xxDC -> __m512d DC
  return ret_d8;
}

static inline __m512 _mm512_cvtpd_pshi_dst(__m512 dst_s16, __m512d src_d8)
{
  __mmask16 mask = 0b1111111100000000;
  __m512 ret_s16 = _mm512_cvtpd_pslo(src_d8);               // __m512d DC -> __m512 00DC
  ret_s16 = _mm512_permute4f128_ps(ret_s16, _MM_PERM_BABA); // __m512 00DC -> __m512 DCDC
  ret_s16 = _mm512_mask_blend_ps(mask, dst_s16, ret_s16);   // __m512 xxBA +  DCxx -> DCBA
  return ret_s16;
}

static inline __m512 _mm512_cvtpd_pslo_dst(__m512 dst_s16, __m512d src_d8)
{
  __mmask16 mask = 0b0000000011111111;
  __m512 ret_s16 = _mm512_cvtpd_pslo(src_d8);             // __m512d BA -> __m512 00BA
  ret_s16 = _mm512_mask_blend_ps(mask, dst_s16, ret_s16); // __m512 DCba +  00BA -> DCBA
  return ret_s16;
}

#else /* GNU compiler */

static inline __m512d _mm512_cvtpslo_pd(__m512 src_s16)
{
  __m512d ret_d8;
  /* src_s16[7:0] -> (YMM float [7:0]) -> ret_d8[7:0] */
  ret_d8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(src_s16, 0));
  return ret_d8;
}

static inline __m512 _mm512_cvtpd_pslo(__m512d src_d8)
{
  __m512 ret_s16 = _mm512_setzero_ps();
  /* ret_s16[7:0] = ( src_d8[7:0]->(YMM float [7:0]) ) */
  ret_s16 = _mm512_insertf32x8(ret_s16, _mm512_cvtpd_ps(src_d8), 0);
  return ret_s16;
}

static inline __m512d _mm512_cvtpshi_pd(__m512 src_s16)
{
  __m512d ret_d8;
  // src_s16[15:8] -> (YMM float [7:0]) -> ret_d8[7:0]
  ret_d8 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(src_s16, 1));
  return ret_d8;
}

static inline __m512 _mm512_cvtpd_pshi_dst(__m512 dst_s16, __m512d src_d8)
{
  __mmask16 mask = 0b1111111100000000;
  __m512 ret_s16 = _mm512_cvtpd_pslo(src_d8); // __m512d DC -> __m512 00DC
  ret_s16 = _mm512_permutexvar_ps(_mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                                  ret_s16);               // __m512 00DC -> __m512 DCxx
  ret_s16 = _mm512_mask_blend_ps(mask, dst_s16, ret_s16); // __m512 xxBA +  DCxx -> DCBA
  return ret_s16;
}

static inline __m512 _mm512_cvtpd_pslo_dst(__m512 dst_s16, __m512d src_d8)
{
  __mmask16 mask = 0b0000000011111111;
  __m512 ret_s16 = _mm512_cvtpd_pslo(src_d8);             // __m512d BA -> __m512 00BA
  ret_s16 = _mm512_mask_blend_ps(mask, dst_s16, ret_s16); // __m512 DCba +  00BA -> DCBA
  return ret_s16;
}

#endif

static inline __m512 _mm512_gfactor_S2_ps(const __m512 _rad, const __m512 _eps_pm)
{
  __m512 _R, _g, _h, _S;
  __m512 _zero = _mm512_set1_ps(0.0f);
  __m512 _one = _mm512_set1_ps(1.0f);
  __m512 _two = _mm512_set1_ps(2.0f);

  _R = _mm512_div_ps(_rad, _eps_pm);
  _R = _mm512_mul_ps(_R, _two);
  _R = _mm512_min_ps(_R, _two);
  _S = _mm512_sub_ps(_R, _one);
  _S = _mm512_max_ps(_S, _zero);
  _S = _mm512_mul_ps(_mm512_mul_ps(_S, _S), _S); // S^3
  _S = _mm512_mul_ps(_S, _S);                    // S^6

  __m512 _coeff0 = _mm512_set1_ps(0.15f);
  __m512 _coeff1 = _mm512_set1_ps(12.0f / 35.0f);
  __m512 _coeff2 = _mm512_set1_ps(-0.5f);
  __m512 _coeff3 = _mm512_set1_ps(1.6f);

  _g = _mm512_fmsub_ps(_coeff0, _R, _coeff1);
  _g = _mm512_fmadd_ps(_g, _R, _coeff2);
  _g = _mm512_fmadd_ps(_g, _R, _coeff3);
  _g = _mm512_mul_ps(_g, _R);
  _g = _mm512_fmsub_ps(_g, _R, _coeff3);
  _g = _mm512_mul_ps(_g, _R);
  _g = _mm512_mul_ps(_g, _R);
  _g = _mm512_mul_ps(_g, _R);

  _coeff0 = _mm512_set1_ps(0.2f);
  _coeff1 = _mm512_set1_ps(18.0f / 35.0f);
  _coeff2 = _mm512_set1_ps(3.0f / 35.0f);

  _h = _mm512_fmadd_ps(_coeff0, _R, _coeff1);
  _h = _mm512_fmadd_ps(_h, _R, _coeff2);
  _h = _mm512_mul_ps(_h, _S);

  _g = _mm512_add_ps(_one, _g);
  _g = _mm512_sub_ps(_g, _h);

  return _g;
}

static inline __m512 _mm512_gfactor_pot_S2_ps(__m512 _rad, const __m512 _eps_pm)
{
  __m512 _R, _R2;
  __m512 _zero = _mm512_set1_ps(0.0f);
  __m512 _one = _mm512_set1_ps(1.0f);
  __m512 _two = _mm512_set1_ps(2.0f);

  _rad = _mm512_min_ps(_rad, _eps_pm); // r = min(r,sft_pm), R=(2,R) is also satisfied
  _R = _mm512_div_ps(_rad, _eps_pm);
  _R = _mm512_mul_ps(_R, _two);
  // _R = _mm512_min_ps(_R, _two);
  _R2 = _mm512_mul_ps(_R, _R);

  __m512 _c4 = _mm512_set1_ps(4.0f);
  __m512 _c8 = _mm512_set1_ps(8.0f);
  __m512 _c14 = _mm512_set1_ps(14.0f);
  __m512 _f = _mm512_div_ps(_one, _c14); // f2 = 1/14

  __m512 _g1 = _mm512_fmsub_ps(_R, _mm512_set1_ps(3.0f), _c8);                  // g1 = -8.0 + 3.0*R
  _g1 = _mm512_fmsub_ps(_R, _mm512_mul_ps(_f, _g1), _one);                      // g1 = -1.0 + R*(f2*g1)
  _g1 = _mm512_fmadd_ps(_R, _g1, _c4);                                          // g1 = 4.0 + R*g1
  _g1 = _mm512_fmsub_ps(_R2, _g1, _c8);                                         // g1 = -8.0 + R2*g1
  _g1 = _mm512_fmadd_ps(_R2, _mm512_mul_ps(_c14, _g1), _mm512_set1_ps(208.0f)); // g1 = 208 + R2*(14*g1)

  __m512 _g2 = _mm512_mul_ps(_R, _mm512_sub_ps(_c8, _R));                      // g2 = R*(8.0-R)
  _g2 = _mm512_fmsub_ps(_f, _g2, _one);                                        // g2 = -1 + f2*g2
  _g2 = _mm512_fmsub_ps(_R, _g2, _c4);                                         // g2 = -4 + R*g2
  _g2 = _mm512_fmadd_ps(_R, _g2, _mm512_set1_ps(20.0f));                       // g2 = 20 + R*g2
  _g2 = _mm512_fmsub_ps(_R, _g2, _mm512_set1_ps(32.0f));                       // g2 = -32 + R*g2
  _g2 = _mm512_fmadd_ps(_R, _g2, _mm512_set1_ps(16.0f));                       // g2 = 16 + R*g2
  _g2 = _mm512_fmadd_ps(_c14, _mm512_mul_ps(_R, _g2), _mm512_set1_ps(128.0f)); // g2 = 128 + 14*R*g2
  _g2 = _mm512_add_ps(_g2, _mm512_div_ps(_mm512_set1_ps(12.0f), _R));          // g2 = 12/R + g2

  // g1 (R<=1, true) or g2 (R>1, false)
  _g1 = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(_R, _one, _MM_CMPINT_LE), _g2,
                             _g1); // mask=0(false) select src1(_g2), mask=1(true) select src2(_g1)
  _f = _mm512_div_ps(_rad, _mm512_mul_ps(_mm512_set1_ps(70.0f), _eps_pm)); //  f = rad / (70 * eps_pm);
  _g1 = _mm512_fnmadd_ps(_f, _g1, _one);                                   //    g = 1.0 - f * g;

  // _g1 = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(_R, _two, _MM_CMPINT_GE), _g1, _zero);
  _g1 = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(_R, _mm512_set1_ps(1.999f), _MM_CMPINT_GE), _g1,
                             _zero); //  For numerical stability

  return _g1;
}

static inline void calc_grav_avx512(double (*xi)[3], double (*ai)[NVECT], double *mj, double (*xj)[3], int njlist,
                                    double eps2_pp, double eps_pm)
{
  __m512 _eps_pm = _mm512_set1_ps((float)eps_pm);

  __m512d _xi_0 = _mm512_set_pd(xi[7][0], xi[6][0], xi[5][0], xi[4][0], xi[3][0], xi[2][0], xi[1][0], xi[0][0]);
  __m512d _yi_0 = _mm512_set_pd(xi[7][1], xi[6][1], xi[5][1], xi[4][1], xi[3][1], xi[2][1], xi[1][1], xi[0][1]);
  __m512d _zi_0 = _mm512_set_pd(xi[7][2], xi[6][2], xi[5][2], xi[4][2], xi[3][2], xi[2][2], xi[1][2], xi[0][2]);

  __m512d _xi_1 = _mm512_set_pd(xi[15][0], xi[14][0], xi[13][0], xi[12][0], xi[11][0], xi[10][0], xi[9][0], xi[8][0]);
  __m512d _yi_1 = _mm512_set_pd(xi[15][1], xi[14][1], xi[13][1], xi[12][1], xi[11][1], xi[10][1], xi[9][1], xi[8][1]);
  __m512d _zi_1 = _mm512_set_pd(xi[15][2], xi[14][2], xi[13][2], xi[12][2], xi[11][2], xi[10][2], xi[9][2], xi[8][2]);

  __m512 _eps2_0 = _mm512_set1_ps((float)eps2_pp);

  __m512d _xacc_0 = _mm512_setzero_pd();
  __m512d _yacc_0 = _mm512_setzero_pd();
  __m512d _zacc_0 = _mm512_setzero_pd();

  __m512d _xacc_1 = _mm512_setzero_pd();
  __m512d _yacc_1 = _mm512_setzero_pd();
  __m512d _zacc_1 = _mm512_setzero_pd();

  for(int j = 0; j < njlist; j++) {
    __m512d _xj, _yj, _zj;

    _xj = _mm512_set1_pd(xj[j][0]);
    _yj = _mm512_set1_pd(xj[j][1]);
    _zj = _mm512_set1_pd(xj[j][2]);

    __m512 _mj = _mm512_set1_ps((float)mj[j]);

    __m512d _dx_0, _dy_0, _dz_0;
    __m512d _dx_1, _dy_1, _dz_1;
    _dx_0 = _mm512_sub_pd(_xj, _xi_0);
    _dy_0 = _mm512_sub_pd(_yj, _yi_0);
    _dz_0 = _mm512_sub_pd(_zj, _zi_0);

    _dx_1 = _mm512_sub_pd(_xj, _xi_1);
    _dy_1 = _mm512_sub_pd(_yj, _yi_1);
    _dz_1 = _mm512_sub_pd(_zj, _zi_1);

    __m512 _dx, _dy, _dz;
    __m512 _tmp0;

    _dx = _mm512_cvtpd_pslo(_dx_0);
    _dx = _mm512_cvtpd_pshi_dst(_dx, _dx_1);

    _dy = _mm512_cvtpd_pslo(_dy_0);
    _dy = _mm512_cvtpd_pshi_dst(_dy, _dy_1);

    _dz = _mm512_cvtpd_pslo(_dz_0);
    _dz = _mm512_cvtpd_pshi_dst(_dz, _dz_1);

    __m512 _rsq, _rad2;
    _rad2 = _mm512_mul_ps(_dx, _dx);
    _rad2 = _mm512_fmadd_ps(_dy, _dy, _rad2);
    _rad2 = _mm512_fmadd_ps(_dz, _dz, _rad2);
    _rsq = _mm512_add_ps(_rad2, _eps2_0);

#ifdef __AVX512ER__
    __m512 _rinv = _mm512_rsqrt28_ps(_rsq);
#else
    __m512 _rinv = _mm512_rsqrt14_ps(_rsq);
#endif

    __m512 _rad = _mm512_sqrt_ps(_rad2);
    __m512 _gfact = _mm512_gfactor_S2_ps(_rad, _eps_pm);
    __m512 _mrinv3 = _mm512_mul_ps(_rinv, _mm512_mul_ps(_rinv, _rinv));
    _mrinv3 = _mm512_mul_ps(_mj, _mrinv3);
    _mrinv3 = _mm512_mul_ps(_mrinv3, _gfact);

    __m512d _mrinv3d;
    _mrinv3d = _mm512_cvtpslo_pd(_mrinv3);

    _xacc_0 = _mm512_fmadd_pd(_mrinv3d, _dx_0, _xacc_0);
    _yacc_0 = _mm512_fmadd_pd(_mrinv3d, _dy_0, _yacc_0);
    _zacc_0 = _mm512_fmadd_pd(_mrinv3d, _dz_0, _zacc_0);

    _mrinv3d = _mm512_cvtpshi_pd(_mrinv3);

    _xacc_1 = _mm512_fmadd_pd(_mrinv3d, _dx_1, _xacc_1);
    _yacc_1 = _mm512_fmadd_pd(_mrinv3d, _dy_1, _yacc_1);
    _zacc_1 = _mm512_fmadd_pd(_mrinv3d, _dz_1, _zacc_1);
  }

  _mm512_store_pd(&ai[0][0], _xacc_0);
  _mm512_store_pd(&ai[0][8], _xacc_1);
  _mm512_store_pd(&ai[1][0], _yacc_0);
  _mm512_store_pd(&ai[1][8], _yacc_1);
  _mm512_store_pd(&ai[2][0], _zacc_0);
  _mm512_store_pd(&ai[2][8], _zacc_1);
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

// for density assignment
// <https://pysph.readthedocs.io/en/latest/reference/kernels.html>

static inline __m512 _mm512_WendlandQuinticC4_ps(const __m512 _rad, const __m512 _h)
{
  /*
  a = 495.0 / (256.0 * M_PI * CUBE(h));
  q = r / h;
  w = 0.0;
  if(q <= 2) w = SQR(CUBE(1.0 - 0.5 * q)) * (SQR(q) * 35.0 / 12.0 + 3.0 * q + 1.0);
  return a * w;
 */

  __m512 _zero = _mm512_set1_ps(0.0f);
  __m512 _one = _mm512_set1_ps(1.0f);
  __m512 _two = _mm512_set1_ps(2.0f);
  __m512 _tmp1, _tmp2, _tmp3;
  _tmp1 = _mm512_mul_ps(_mm512_set1_ps(M_PI), _mm512_mul_ps(_h, _mm512_mul_ps(_h, _h)));
  __m512 _a = _mm512_div_ps(_mm512_set1_ps(495.0f), _mm512_mul_ps(_mm512_set1_ps(256.0f), _tmp1));
  __m512 _q = _mm512_div_ps(_rad, _h);

  _tmp1 = _mm512_fnmadd_ps(_mm512_set1_ps(0.5f), _q, _one);  // 1.0 - 0.5 * q
  _tmp1 = _mm512_mul_ps(_tmp1, _mm512_mul_ps(_tmp1, _tmp1)); // CUBE(1.0 - 0.5 * q)
  _tmp1 = _mm512_mul_ps(_tmp1, _tmp1);                       // SQR(CUBE(1.0 - 0.5 * q))

  _tmp3 = _mm512_set1_ps(35.0f / 12.0f);
  _tmp2 = _mm512_fmadd_ps(_q, _mm512_set1_ps(3.0f), _one);
  _tmp2 = _mm512_fmadd_ps(_mm512_mul_ps(_q, _q), _tmp3, _tmp2);

  _tmp1 = _mm512_mul_ps(_tmp1, _tmp2);
  _tmp1 = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(_q, _two, _MM_CMPINT_LE), _zero, _tmp1);
  return _mm512_mul_ps(_a, _tmp1);
}

static inline __m512 _mm512_WendlandQuinticC4_dwdq_ps(const __m512 _rad, const __m512 _h)
{
  /*
  a = 495.0 / (256.0 * M_PI * CUBE(h));
  q = r / h;
  dwdq = 0.0;
  if(q <= 2) dwdq = (SQR(SQR(1.0 - 0.5 * q)) * (1.0 - 0.5 * q)) * (-14.0 / 3.0) * q * (1.0 + 2.5 * q);
  return a * dwdq;
  */
  __m512 _zero = _mm512_set1_ps(0.0f);
  __m512 _one = _mm512_set1_ps(1.0f);
  __m512 _two = _mm512_set1_ps(2.0f);
  __m512 _tmp1, _tmp2, _tmp3;
  _tmp1 = _mm512_mul_ps(_mm512_set1_ps(M_PI), _mm512_mul_ps(_h, _mm512_mul_ps(_h, _h)));
  __m512 _a = _mm512_div_ps(_mm512_set1_ps(495.0f), _mm512_mul_ps(_mm512_set1_ps(256.0f), _tmp1));
  __m512 _q = _mm512_div_ps(_rad, _h);

  _tmp1 = _mm512_fnmadd_ps(_mm512_set1_ps(0.5f), _q, _one); // 1.0 - 0.5 * q
  _tmp2 = _mm512_mul_ps(_tmp1, _mm512_mul_ps(_tmp1, _tmp1));
  _tmp1 = _mm512_mul_ps(_mm512_mul_ps(_tmp1, _tmp1), _tmp2); // SQR(1.0 - 0.5 * q)*CUBE(1.0 - 0.5 * q)

  _tmp3 = _mm512_fmadd_ps(_mm512_set1_ps(2.5f), _q, _one); // 1.0 + 2.5 * q
  _tmp2 = _mm512_set1_ps(-14.0f / 3.0f);
  _tmp2 = _mm512_mul_ps(_mm512_mul_ps(_tmp2, _q), _tmp3); // (-14.0 / 3.0) * q * (1.0 + 2.5 * q)

  _tmp1 = _mm512_mul_ps(_tmp1, _tmp2);
  _tmp1 = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(_q, _two, _MM_CMPINT_LE), _zero, _tmp1);
  return _mm512_mul_ps(_a, _tmp1);
}

static inline __m512 _mm512_density_kernel_ps(const __m512 _rad, const __m512 _h)
{
  return _mm512_WendlandQuinticC4_ps(_rad, _h);
}

static inline __m512 _mm512_density_grad_kernel_ps(const __m512 _rad, const __m512 _h)
{
  /* Multiply this by the distance in each axes. */
  // if(_rad < 1e-30) return 0.0;
  __m512 _irh = _mm512_div_ps(_mm512_set1_ps(1.0f), _mm512_mul_ps(_rad, _h));
  _irh = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(_rad, _mm512_set1_ps(1.0e-30f), _MM_CMPINT_GE), _mm512_set1_ps(0.0f),
                              _irh); // irh = (rad >= 1e-30) ? irh : 0.0;
  return _mm512_mul_ps(_mm512_WendlandQuinticC4_dwdq_ps(_rad, _h), _irh);
}

static inline void calc_grav_avx512_with_potdens(double (*xi)[3], double (*ai)[NVECT], double *poti, double *densi,
                                                 double (*dgi)[NVECT], float *hi, double *mj, double (*xj)[3],
                                                 int njlist, double eps2_pp, double eps_pm)
{
  __m512 _eps_pm = _mm512_set1_ps((float)eps_pm);

  __m512d _xi_0 = _mm512_set_pd(xi[7][0], xi[6][0], xi[5][0], xi[4][0], xi[3][0], xi[2][0], xi[1][0], xi[0][0]);
  __m512d _yi_0 = _mm512_set_pd(xi[7][1], xi[6][1], xi[5][1], xi[4][1], xi[3][1], xi[2][1], xi[1][1], xi[0][1]);
  __m512d _zi_0 = _mm512_set_pd(xi[7][2], xi[6][2], xi[5][2], xi[4][2], xi[3][2], xi[2][2], xi[1][2], xi[0][2]);

  __m512d _xi_1 = _mm512_set_pd(xi[15][0], xi[14][0], xi[13][0], xi[12][0], xi[11][0], xi[10][0], xi[9][0], xi[8][0]);
  __m512d _yi_1 = _mm512_set_pd(xi[15][1], xi[14][1], xi[13][1], xi[12][1], xi[11][1], xi[10][1], xi[9][1], xi[8][1]);
  __m512d _zi_1 = _mm512_set_pd(xi[15][2], xi[14][2], xi[13][2], xi[12][2], xi[11][2], xi[10][2], xi[9][2], xi[8][2]);

  __m512 _hi = _mm512_load_ps(hi);
  __m512 _h = _mm512_div_ps(_mm512_set1_ps(0.2f),
                            _mm512_set1_ps((float)NPART_1D)); // kernel length = 0.2 x mean particle dist. in 1D
  _hi = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(_hi, _mm512_set1_ps(0.0f), _CMP_LT_OQ), _hi,
                             _h); // _hi = (_hi < 0) ? _h : _hi;

  __m512 _eps2_0 = _mm512_set1_ps((float)eps2_pp);

  __m512d _xacc_0 = _mm512_setzero_pd();
  __m512d _yacc_0 = _mm512_setzero_pd();
  __m512d _zacc_0 = _mm512_setzero_pd();
  __m512d _xacc_1 = _mm512_setzero_pd();
  __m512d _yacc_1 = _mm512_setzero_pd();
  __m512d _zacc_1 = _mm512_setzero_pd();

  __m512d _pot_0 = _mm512_setzero_pd();
  __m512d _pot_1 = _mm512_setzero_pd();

  __m512d _dens_0 = _mm512_setzero_pd();
  __m512d _dens_1 = _mm512_setzero_pd();

  __m512d _xdg_0 = _mm512_setzero_pd();
  __m512d _ydg_0 = _mm512_setzero_pd();
  __m512d _zdg_0 = _mm512_setzero_pd();
  __m512d _xdg_1 = _mm512_setzero_pd();
  __m512d _ydg_1 = _mm512_setzero_pd();
  __m512d _zdg_1 = _mm512_setzero_pd();

  for(int j = 0; j < njlist; j++) {
    __m512d _xj, _yj, _zj;

    _xj = _mm512_set1_pd(xj[j][0]);
    _yj = _mm512_set1_pd(xj[j][1]);
    _zj = _mm512_set1_pd(xj[j][2]);

    __m512 _mj = _mm512_set1_ps((float)mj[j]);

    __m512d _dx_0, _dy_0, _dz_0;
    __m512d _dx_1, _dy_1, _dz_1;
    _dx_0 = _mm512_sub_pd(_xj, _xi_0);
    _dy_0 = _mm512_sub_pd(_yj, _yi_0);
    _dz_0 = _mm512_sub_pd(_zj, _zi_0);

    _dx_1 = _mm512_sub_pd(_xj, _xi_1);
    _dy_1 = _mm512_sub_pd(_yj, _yi_1);
    _dz_1 = _mm512_sub_pd(_zj, _zi_1);

    __m512 _dx, _dy, _dz;
    __m512 _tmp0;

    _dx = _mm512_cvtpd_pslo(_dx_0);
    _dx = _mm512_cvtpd_pshi_dst(_dx, _dx_1);

    _dy = _mm512_cvtpd_pslo(_dy_0);
    _dy = _mm512_cvtpd_pshi_dst(_dy, _dy_1);

    _dz = _mm512_cvtpd_pslo(_dz_0);
    _dz = _mm512_cvtpd_pshi_dst(_dz, _dz_1);

    __m512 _rsq, _rad2;
    _rad2 = _mm512_mul_ps(_dx, _dx);
    _rad2 = _mm512_fmadd_ps(_dy, _dy, _rad2);
    _rad2 = _mm512_fmadd_ps(_dz, _dz, _rad2);
    _rsq = _mm512_add_ps(_rad2, _eps2_0);

#ifdef __AVX512ER__
    __m512 _rinv = _mm512_rsqrt28_ps(_rsq);
#else
    __m512 _rinv = _mm512_rsqrt14_ps(_rsq);
#endif

    __m512 _rad = _mm512_sqrt_ps(_rad2);
    __m512 _gfact = _mm512_gfactor_S2_ps(_rad, _eps_pm);
    __m512 _mrinv3 = _mm512_mul_ps(_rinv, _mm512_mul_ps(_rinv, _rinv));
    _mrinv3 = _mm512_mul_ps(_mj, _mrinv3);
    _mrinv3 = _mm512_mul_ps(_mrinv3, _gfact);

    __m512d _mrinv3d;
    _mrinv3d = _mm512_cvtpslo_pd(_mrinv3);
    _xacc_0 = _mm512_fmadd_pd(_mrinv3d, _dx_0, _xacc_0);
    _yacc_0 = _mm512_fmadd_pd(_mrinv3d, _dy_0, _yacc_0);
    _zacc_0 = _mm512_fmadd_pd(_mrinv3d, _dz_0, _zacc_0);

    _mrinv3d = _mm512_cvtpshi_pd(_mrinv3);
    _xacc_1 = _mm512_fmadd_pd(_mrinv3d, _dx_1, _xacc_1);
    _yacc_1 = _mm512_fmadd_pd(_mrinv3d, _dy_1, _yacc_1);
    _zacc_1 = _mm512_fmadd_pd(_mrinv3d, _dz_1, _zacc_1);

    // pot = pot - mj*gfact_pot*(1/(r+e))
    _gfact = _mm512_gfactor_pot_S2_ps(_rad, _eps_pm);
    _gfact = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(_rsq, _mm512_set1_ps(1.0e-30f), _MM_CMPINT_GE),
                                  _mm512_set1_ps(0.0f), _gfact); // gfact = (rsq >= 1e-30) ? gfact : 0.0;

    __m512 _mrinv = _mm512_mul_ps(_mm512_mul_ps(_mj, _gfact), _rinv);

    __m512d _mrinvd;
    _mrinvd = _mm512_cvtpslo_pd(_mrinv);
    _pot_0 = _mm512_sub_pd(_pot_0, _mrinvd);
    _mrinvd = _mm512_cvtpshi_pd(_mrinv);
    _pot_1 = _mm512_sub_pd(_pot_1, _mrinvd);

    // dens = dens + mj * density_kernel // calculate i=j particle.
    // dens_grad[3] = dens_grad[3] + mj * dr[3] * density_grad_kernel
    __m512 _dens = _mm512_mul_ps(_mj, _mm512_density_kernel_ps(_rad, _hi));
    __m512 _dens_grad = _mm512_mul_ps(_mj, _mm512_density_grad_kernel_ps(_rad, _hi));

    __m512d _densd;
    _densd = _mm512_cvtpslo_pd(_dens);
    _dens_0 = _mm512_add_pd(_dens_0, _densd);
    _densd = _mm512_cvtpshi_pd(_dens);
    _dens_1 = _mm512_add_pd(_dens_1, _densd);

    _densd = _mm512_cvtpslo_pd(_dens_grad);
    _xdg_0 = _mm512_fmadd_pd(_densd, _dx_0, _xdg_0);
    _ydg_0 = _mm512_fmadd_pd(_densd, _dy_0, _ydg_0);
    _zdg_0 = _mm512_fmadd_pd(_densd, _dz_0, _zdg_0);

    _densd = _mm512_cvtpshi_pd(_dens_grad);
    _xdg_1 = _mm512_fmadd_pd(_densd, _dx_1, _xdg_1);
    _ydg_1 = _mm512_fmadd_pd(_densd, _dy_1, _ydg_1);
    _zdg_1 = _mm512_fmadd_pd(_densd, _dz_1, _zdg_1);
  }

  _mm512_store_pd(&ai[0][0], _xacc_0);
  _mm512_store_pd(&ai[0][8], _xacc_1);
  _mm512_store_pd(&ai[1][0], _yacc_0);
  _mm512_store_pd(&ai[1][8], _yacc_1);
  _mm512_store_pd(&ai[2][0], _zacc_0);
  _mm512_store_pd(&ai[2][8], _zacc_1);

  _mm512_store_pd(&poti[0], _pot_0);
  _mm512_store_pd(&poti[8], _pot_1);

  _mm512_store_pd(&densi[0], _dens_0);
  _mm512_store_pd(&densi[8], _dens_1);

  _mm512_store_pd(&dgi[0][0], _xdg_0);
  _mm512_store_pd(&dgi[0][8], _xdg_1);
  _mm512_store_pd(&dgi[1][0], _ydg_0);
  _mm512_store_pd(&dgi[1][8], _ydg_1);
  _mm512_store_pd(&dgi[2][0], _zdg_0);
  _mm512_store_pd(&dgi[2][8], _zdg_1);
}

void calc_grav_simd_thread_with_pot(EPItreepm *iptcl, Result_treepm *ppforce, const int ni, double (*xj)[3], double *mj,
                                    const int nj, double eps2_pp, double eps_pm)
{
  double xii[NVECT][3] __attribute__((aligned(ALIGNE_SIZE)));
  double aii[3][NVECT] __attribute__((aligned(ALIGNE_SIZE)));
  double poti[NVECT] __attribute__((aligned(ALIGNE_SIZE)));
  double densi[NVECT] __attribute__((aligned(ALIGNE_SIZE)));
  double dgi[3][NVECT] __attribute__((aligned(ALIGNE_SIZE)));
  float hi[NVECT] __attribute__((aligned(ALIGNE_SIZE)));

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
      ppforce[iii].dens_grad[0] -= dgi[0][ii];
      ppforce[iii].dens_grad[1] -= dgi[1][ii];
      ppforce[iii].dens_grad[2] -= dgi[2][ii];
    }
  }
}
#endif
