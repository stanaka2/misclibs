#pragma once

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif // __ARM_FEATURE_SVE

#include <float.h>
#include "pppm.h"

#ifndef SQR
#define SQR(x) ((x) * (x))
#endif

#ifndef CUBE
#define CUBE(x) ((x) * (x) * (x))
#endif

#undef ALIGNE_SIZE
#define NIMAX (1024)
#define ALIGNE_SIZE (64)
#define NVECT (16) // 8 iptcls * 2

struct iptcl {
  float xpos[NIMAX] __attribute__((aligned(ALIGNE_SIZE)));
  float ypos[NIMAX] __attribute__((aligned(ALIGNE_SIZE)));
  float zpos[NIMAX] __attribute__((aligned(ALIGNE_SIZE)));

  float xacc[NIMAX] __attribute__((aligned(ALIGNE_SIZE)));
  float yacc[NIMAX] __attribute__((aligned(ALIGNE_SIZE)));
  float zacc[NIMAX] __attribute__((aligned(ALIGNE_SIZE)));

#ifdef OUTPUT_POT
  float pot[NIMAX] __attribute__((aligned(ALIGNE_SIZE)));
  float dens[NIMAX] __attribute__((aligned(ALIGNE_SIZE)));
  float xdg[NIMAX] __attribute__((aligned(ALIGNE_SIZE)));
  float ydg[NIMAX] __attribute__((aligned(ALIGNE_SIZE)));
  float zdg[NIMAX] __attribute__((aligned(ALIGNE_SIZE)));
  float smooth_l[NIMAX] __attribute__((aligned(ALIGNE_SIZE)));
#endif
};

struct jlist_t {
  float xpos, ypos, zpos;
  float mass;
};

static inline svfloat32_t rsqrt_acc(svbool_t pg, svfloat32_t _x)
{
  svfloat32_t _res = svrsqrte(_x);
  _res = svmul_x(pg, svrsqrts(svmul_x(pg, _res, _res), _x), _res); // higher accuracy
  _res = svmul_x(pg, svrsqrts(svmul_x(pg, _res, _res), _x), _res); // higher accuracy
  _res = svmul_x(pg, svrsqrts(svmul_x(pg, _res, _res), _x), _res); // higher accuracy
  return _res;
}

static inline svfloat32_t gfactor_acc_sve(svbool_t pg, svfloat32_t _rad, svfloat32_t _rsft_pm)
{
  const float32_t zero = 0.0;
  const float32_t one = 1.0;
  const float32_t two = 2.0;

  const float32_t coeff0 = 0.15;
  const float32_t coeff1 = 12.0 / 35.0;
  const float32_t coeff2 = -0.5;
  const float32_t coeff3 = 1.6;
  const float32_t coeff4 = 0.2;
  const float32_t coeff5 = 18.0 / 35.0;
  const float32_t coeff6 = 3.0 / 35.0;

  svfloat32_t _zero = svdup_f32(zero);
  svfloat32_t _one = svdup_f32(one);
  svfloat32_t _two = svdup_f32(two);

  svfloat32_t _g, _h, _R, _R3, _S;

  _R = svmul_x(pg, svmul_x(pg, _rad, _two), _rsft_pm);
  _R = svmin_x(pg, _R, _two);

  _R3 = svmul_x(pg, svmul_x(pg, _R, _R), _R);

  _S = svmax_x(pg, svsub_x(pg, _R, _one), zero);
  _S = svmul_x(pg, svmul_x(pg, _S, _S), _S); // S^3
  _S = svmul_x(pg, _S, _S);                  // S^6

  _g = svnmsb_x(pg, svdup_f32(coeff0), _R, coeff1); // _g = -(c1 - c0*R) =  c0*R - c1
  _g = svmad_x(pg, _g, _R, coeff2);                 // _g = c2 + g*R
  _g = svmad_x(pg, _g, _R, coeff3);                 // _g = c3 + g*R
  _g = svmul_x(pg, _g, _R);                         // _g = g*R
  _g = svnmsb_x(pg, _g, _R, coeff3);                // _g = -(c3 - g*R) = g*R - c3
  _g = svmul_x(pg, _g, _R3);                        // _g = g*R3

  _h = svmad_x(pg, svdup_f32(coeff4), _R, coeff5); // _h = c5 + c4*R
  _h = svmad_x(pg, _h, _R, coeff6);                // _h = c6 + h*R
  _h = svmul_x(pg, _h, _S);                        // _h = _h*S  (this S is S^6)
  _h = svsub_x(pg, _one, _h);                      // _h = 1 - _h

  _g = svadd_x(pg, _g, _h); // _g = _g + _h

  return _g;
}

static inline svfloat32_t gfactor_pot_acc_sve(svbool_t pg, svfloat32_t _rad, svfloat32_t _rsft_pm)
{
  const float32_t zero = 0.0;
  const float32_t one = 1.0;
  const float32_t two = 2.0;
  svfloat32_t _zero = svdup_f32(zero);
  svfloat32_t _one = svdup_f32(one);
  svfloat32_t _two = svdup_f32(two);

  svfloat32_t _c4 = svdup_f32(4.0f);
  svfloat32_t _c8 = svdup_f32(8.0f);
  svfloat32_t _c14 = svdup_f32(14.0f);
  svfloat32_t _f = svdup_f32(1.0f / 14.0f);

  svfloat32_t _g1, _g2, _R, _R2;

  _rad = svmin_x(pg, _rad, svdiv_x(pg, _one, _rsft_pm)); // r = min(r,sft_pm), R=(2,R) is also satisfied
  _R = svmul_x(pg, svmul_x(pg, _rad, _two), _rsft_pm);
  // _R = svmin_x(pg, _R, _two);
  _R2 = svmul_x(pg, _R, _R);

  _g1 = svnmsb_x(pg, svdup_f32(3.0f), _R, _c8);           // g1 = -8.0 + 3.0*R
  _g1 = svnmsb_x(pg, svmul_x(pg, _f, _g1), _R, _one);     // g1 = -1.0 + R*(f2*g1)
  _g1 = svmad_x(pg, _R, _g1, _c4);                        // g1 = 4.0 + R*g1
  _g1 = svnmsb_x(pg, _R2, _g1, _c8);                      // g1 = -8.0 + R2*g1
  _g1 = svmad_x(pg, _R2, svmul_x(pg, _c14, _g1), 208.0f); // g1 = 208 + R2*(14*g1)

  _g2 = svmul_x(pg, _R, svsub_x(pg, _c8, _R));               // g2 = R*(8.0-R)
  _g2 = svnmsb_x(pg, _f, _g2, _one);                         // g2 = -1 + f2*g2
  _g2 = svnmsb_x(pg, _R, _g2, _c4);                          // g2 = -4 + R*g2
  _g2 = svmad_x(pg, _R, _g2, 20.0f);                         // g2 = 20 + R*g2
  _g2 = svnmsb_x(pg, _R, _g2, 32.0f);                        // g2 = -32 + R*g2
  _g2 = svmad_x(pg, _R, _g2, 16.0f);                         // g2 = 16 + R*g2
  _g2 = svmad_x(pg, _c14, svmul_x(pg, _R, _g2), 128.0f);     // g2 = 128 + 14*R*g2
  _g2 = svadd_x(pg, _g2, svdiv_x(pg, svdup_f32(12.0f), _R)); // g2 = 12/R + g2

  _f = svmul_x(pg, svdiv_x(pg, _rad, 70.0f), _rsft_pm); // f = rad / (70 * eps_pm);
  _g1 = svsel(svcmple(pg, _R, _one), _g1, _g2);         // if(R <= 1) g = g1; else g = g2;

  _g1 = svmsb_x(pg, _f, _g1, _one); // g = 1.0 - f * g;

  // _g1 = svsel(svcmpge(pg, _R, _two), _zero, _g1); // check if(R >= 2) // 0 ? (R>=2) : _g1;
  _g1 = svsel(svcmpge(pg, _R, svdup_f32(1.999f)), _zero, _g1); // 0 ? (R>=1.999f) : _g1; For numerical stability

  return _g1;
}

static inline void calc_grav_S2_with_sve_i16_j1_unroll16(struct iptcl *ip, const int ni, struct jlist_t *jp,
                                                         const int nj, float sft2_pp, float sft_pm)
{
  int i = 0;
  svbool_t pg = svwhilelt_b32(i, ni);
  svbool_t pg_j = svwhilelt_b32(0, 4);

  float rsft_pm = 1.0 / sft_pm;

  int nj16, mod_nj16;
  mod_nj16 = nj % 16;
  nj16 = nj - mod_nj16;

  do {
    svfloat32_t _xacc = svdup_f32(0.0f);
    svfloat32_t _yacc = svdup_f32(0.0f);
    svfloat32_t _zacc = svdup_f32(0.0f);

    svfloat32_t _xi = svld1(pg, ip->xpos + i);
    svfloat32_t _yi = svld1(pg, ip->ypos + i);
    svfloat32_t _zi = svld1(pg, ip->zpos + i);

    // svfloat32_t _eps2 = svld1(pg, ip->eps+i);
    //_eps2 = svmul_x(pg, _eps2, _eps2);
    svfloat32_t _eps2 = svdup_f32(sft2_pp);
    svfloat32_t _rsft_pm = svdup_f32(rsft_pm);

    svfloat32_t _mj, _xj, _yj, _zj;
    svfloat32_t _dx, _dy, _dz;
    svfloat32_t _rsq, _rad, _gfact, _mrinv3, _rinv;

    for(int j = 0; j < nj16; j += 16) {
      _mj = svld1(pg_j, (float32_t *)(jp + j));
      _xj = svdup_lane(_mj, 0); // xpos
      _yj = svdup_lane(_mj, 1); // ypos
      _zj = svdup_lane(_mj, 2); // zpos
      _mj = svdup_lane(_mj, 3); // mass

      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);

      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);

      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
      _gfact = gfactor_acc_sve(pg, _rad, _rsft_pm);
      _rinv = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2)); // 1.0/sqrt(r^2 + eps^2)

      _mrinv3 = svmul_x(pg, svmul_x(pg, _rinv, _rinv), _rinv);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);

      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      _mj = svld1(pg_j, (float32_t *)(jp + j + 1));
      _xj = svdup_lane(_mj, 0);
      _yj = svdup_lane(_mj, 1);
      _zj = svdup_lane(_mj, 2);
      _mj = svdup_lane(_mj, 3);
      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);
      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
      _gfact = gfactor_acc_sve(pg, _rad, _rsft_pm);
      _rinv = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
      _mrinv3 = svmul_x(pg, svmul_x(pg, _rinv, _rinv), _rinv);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);
      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      _mj = svld1(pg_j, (float32_t *)(jp + j + 2));
      _xj = svdup_lane(_mj, 0);
      _yj = svdup_lane(_mj, 1);
      _zj = svdup_lane(_mj, 2);
      _mj = svdup_lane(_mj, 3);
      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);
      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
      _gfact = gfactor_acc_sve(pg, _rad, _rsft_pm);
      _rinv = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
      _mrinv3 = svmul_x(pg, svmul_x(pg, _rinv, _rinv), _rinv);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);
      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      _mj = svld1(pg_j, (float32_t *)(jp + j + 3));
      _xj = svdup_lane(_mj, 0);
      _yj = svdup_lane(_mj, 1);
      _zj = svdup_lane(_mj, 2);
      _mj = svdup_lane(_mj, 3);
      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);
      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
      _gfact = gfactor_acc_sve(pg, _rad, _rsft_pm);
      _rinv = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
      _mrinv3 = svmul_x(pg, svmul_x(pg, _rinv, _rinv), _rinv);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);
      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      _mj = svld1(pg_j, (float32_t *)(jp + j + 4));
      _xj = svdup_lane(_mj, 0);
      _yj = svdup_lane(_mj, 1);
      _zj = svdup_lane(_mj, 2);
      _mj = svdup_lane(_mj, 3);
      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);
      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
      _gfact = gfactor_acc_sve(pg, _rad, _rsft_pm);
      _rinv = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
      _mrinv3 = svmul_x(pg, svmul_x(pg, _rinv, _rinv), _rinv);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);
      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      _mj = svld1(pg_j, (float32_t *)(jp + j + 5));
      _xj = svdup_lane(_mj, 0);
      _yj = svdup_lane(_mj, 1);
      _zj = svdup_lane(_mj, 2);
      _mj = svdup_lane(_mj, 3);
      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);
      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
      _gfact = gfactor_acc_sve(pg, _rad, _rsft_pm);
      _rinv = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
      _mrinv3 = svmul_x(pg, svmul_x(pg, _rinv, _rinv), _rinv);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);
      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      _mj = svld1(pg_j, (float32_t *)(jp + j + 6));
      _xj = svdup_lane(_mj, 0);
      _yj = svdup_lane(_mj, 1);
      _zj = svdup_lane(_mj, 2);
      _mj = svdup_lane(_mj, 3);
      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);
      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
      _gfact = gfactor_acc_sve(pg, _rad, _rsft_pm);
      _rinv = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
      _mrinv3 = svmul_x(pg, svmul_x(pg, _rinv, _rinv), _rinv);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);
      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      _mj = svld1(pg_j, (float32_t *)(jp + j + 7));
      _xj = svdup_lane(_mj, 0);
      _yj = svdup_lane(_mj, 1);
      _zj = svdup_lane(_mj, 2);
      _mj = svdup_lane(_mj, 3);
      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);
      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
      _gfact = gfactor_acc_sve(pg, _rad, _rsft_pm);
      _rinv = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
      _mrinv3 = svmul_x(pg, svmul_x(pg, _rinv, _rinv), _rinv);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);
      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      _mj = svld1(pg_j, (float32_t *)(jp + j + 8));
      _xj = svdup_lane(_mj, 0);
      _yj = svdup_lane(_mj, 1);
      _zj = svdup_lane(_mj, 2);
      _mj = svdup_lane(_mj, 3);
      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);
      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
      _gfact = gfactor_acc_sve(pg, _rad, _rsft_pm);
      _rinv = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
      _mrinv3 = svmul_x(pg, svmul_x(pg, _rinv, _rinv), _rinv);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);
      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      _mj = svld1(pg_j, (float32_t *)(jp + j + 9));
      _xj = svdup_lane(_mj, 0);
      _yj = svdup_lane(_mj, 1);
      _zj = svdup_lane(_mj, 2);
      _mj = svdup_lane(_mj, 3);
      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);
      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
      _gfact = gfactor_acc_sve(pg, _rad, _rsft_pm);
      _rinv = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
      _mrinv3 = svmul_x(pg, svmul_x(pg, _rinv, _rinv), _rinv);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);
      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      _mj = svld1(pg_j, (float32_t *)(jp + j + 10));
      _xj = svdup_lane(_mj, 0);
      _yj = svdup_lane(_mj, 1);
      _zj = svdup_lane(_mj, 2);
      _mj = svdup_lane(_mj, 3);
      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);
      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
      _gfact = gfactor_acc_sve(pg, _rad, _rsft_pm);
      _rinv = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
      _mrinv3 = svmul_x(pg, svmul_x(pg, _rinv, _rinv), _rinv);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);
      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      _mj = svld1(pg_j, (float32_t *)(jp + j + 11));
      _xj = svdup_lane(_mj, 0);
      _yj = svdup_lane(_mj, 1);
      _zj = svdup_lane(_mj, 2);
      _mj = svdup_lane(_mj, 3);
      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);
      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
      _gfact = gfactor_acc_sve(pg, _rad, _rsft_pm);
      _rinv = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
      _mrinv3 = svmul_x(pg, svmul_x(pg, _rinv, _rinv), _rinv);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);
      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      _mj = svld1(pg_j, (float32_t *)(jp + j + 12));
      _xj = svdup_lane(_mj, 0);
      _yj = svdup_lane(_mj, 1);
      _zj = svdup_lane(_mj, 2);
      _mj = svdup_lane(_mj, 3);
      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);
      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
      _gfact = gfactor_acc_sve(pg, _rad, _rsft_pm);
      _rinv = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
      _mrinv3 = svmul_x(pg, svmul_x(pg, _rinv, _rinv), _rinv);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);
      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      _mj = svld1(pg_j, (float32_t *)(jp + j + 13));
      _xj = svdup_lane(_mj, 0);
      _yj = svdup_lane(_mj, 1);
      _zj = svdup_lane(_mj, 2);
      _mj = svdup_lane(_mj, 3);
      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);
      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
      _gfact = gfactor_acc_sve(pg, _rad, _rsft_pm);
      _rinv = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
      _mrinv3 = svmul_x(pg, svmul_x(pg, _rinv, _rinv), _rinv);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);
      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      _mj = svld1(pg_j, (float32_t *)(jp + j + 14));
      _xj = svdup_lane(_mj, 0);
      _yj = svdup_lane(_mj, 1);
      _zj = svdup_lane(_mj, 2);
      _mj = svdup_lane(_mj, 3);
      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);
      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
      _gfact = gfactor_acc_sve(pg, _rad, _rsft_pm);
      _rinv = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
      _mrinv3 = svmul_x(pg, svmul_x(pg, _rinv, _rinv), _rinv);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);
      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      _mj = svld1(pg_j, (float32_t *)(jp + j + 15));
      _xj = svdup_lane(_mj, 0);
      _yj = svdup_lane(_mj, 1);
      _zj = svdup_lane(_mj, 2);
      _mj = svdup_lane(_mj, 3);
      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);
      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
      _gfact = gfactor_acc_sve(pg, _rad, _rsft_pm);
      _rinv = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
      _mrinv3 = svmul_x(pg, svmul_x(pg, _rinv, _rinv), _rinv);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);
      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);
    }

    for(int j = nj16; j < nj; j++) {
      _mj = svld1(pg_j, (float32_t *)(jp + j));
      _xj = svdup_lane(_mj, 0); // xpos
      _yj = svdup_lane(_mj, 1); // ypos
      _zj = svdup_lane(_mj, 2); // zpos
      _mj = svdup_lane(_mj, 3); // mass

      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);

      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);

      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
      _gfact = gfactor_acc_sve(pg, _rad, _rsft_pm);
      _rinv = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2)); // 1.0/sqrt(r^2 + eps^2)

      _mrinv3 = svmul_x(pg, svmul_x(pg, _rinv, _rinv), _rinv);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);

      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);
    }

    svst1(pg, ip->xacc + i, _xacc);
    svst1(pg, ip->yacc + i, _yacc);
    svst1(pg, ip->zacc + i, _zacc);

    i += svcntw(); // for word (float)
    pg = svwhilelt_b32(i, ni);
  } while(svptest_any(svptrue_b32(), pg));
}

static inline float gfactor_acc(float rad, float eps_pm)
{
  float R, S, g;

  const float zero = 0.0;
  const float one = 1.0;
  const float two = 2.0;

  const float coeff0 = 0.15;
  const float coeff1 = 12.0 / 35.0;
  const float coeff2 = -0.5;
  const float coeff3 = 1.6;
  const float coeff4 = 0.2;
  const float coeff5 = 18.0 / 35.0;
  const float coeff6 = 3.0 / 35.0;

  R = two * rad / eps_pm;
  S = R - one;
  S = (S > zero) ? S : zero;

  g = one + CUBE(R) * (-coeff3 + SQR(R) * (coeff3 + R * (coeff2 + R * (coeff0 * R - coeff1)))) -
      CUBE(S) * CUBE(S) * (coeff6 + R * (coeff5 + coeff4 * R));

  if(R >= two) g = zero;

  return g;
}

static inline void calc_grav_S2_wo_sve(struct iptcl *ip, const int ni, struct jlist_t *jp, const int nj, float sft2_pp,
                                       float sft_pm)
{
  for(int i = 0; i < ni; i++) {
    float xacc, yacc, zacc;
    xacc = yacc = zacc = 0.0;

    for(int j = 0; j < nj; j++) {
      float dx = jp[j].xpos - ip->xpos[i];
      float dy = jp[j].ypos - ip->ypos[i];
      float dz = jp[j].zpos - ip->zpos[i];

      float rsq = dx * dx + dy * dy + dz * dz;
      float rad = sqrtf(rsq);
      // float rinv = 1.0f/sqrtf(rsq+SQR(ip->eps[i]));
      float rinv = 1.0f / sqrtf(rsq + sft2_pp);
      float mrinv3 = jp[j].mass * CUBE(rinv);

      float gfact = gfactor_acc(rad, sft_pm);

      xacc += mrinv3 * dx * gfact;
      yacc += mrinv3 * dy * gfact;
      zacc += mrinv3 * dz * gfact;
    }

    ip->xacc[i] = xacc;
    ip->yacc[i] = yacc;
    ip->zacc[i] = zacc;
  }
}

void calc_grav_simd_thread(EPItreepm *iptcl, Result_treepm *ppforce, const int ni, struct jlist_t *jlist, const int nj,
                           double eps2_pp, double eps_pm)
{
  struct iptcl ip;
  float sft_eps2_pp, sft_eps_pm;
  sft_eps2_pp = eps2_pp;
  sft_eps_pm = eps_pm;

  for(int i = 0; i < ni; i += NIMAX) {
    int nn = NIMAX;
    if((i + NIMAX) > ni) nn = ni - i;

    for(int ii = 0; ii < nn; ii++) {
      int iii = i + ii;
      ip.xpos[ii] = iptcl[iii].pos[0];
      ip.ypos[ii] = iptcl[iii].pos[1];
      ip.zpos[ii] = iptcl[iii].pos[2];
    }

#ifdef PG_SVE_DOUBLE
    calc_grav_S2_with_sve_i8_j1_unroll8_d(&ip, nn, jlist, nj, sft_eps2_pp, sft_eps_pm);
#else
    // calc_grav_S2_wo_sve(&ip, nn, jlist, nj, sft_eps2_pp, sft_eps_pm);
    calc_grav_S2_with_sve_i16_j1_unroll16(&ip, nn, jlist, nj, sft_eps2_pp, sft_eps_pm);
#endif

    for(int ii = 0; ii < nn; ii++) {
      int iii = i + ii;
      ppforce[iii].acc[0] += ip.xacc[ii];
      ppforce[iii].acc[1] += ip.yacc[ii];
      ppforce[iii].acc[2] += ip.zacc[ii];
    }
  }
}

#ifdef OUTPUT_POT

// for density assignment
// <https://pysph.readthedocs.io/en/latest/reference/kernels.html>
static inline svfloat32_t WendlandQuinticC4_sve(svbool_t pg, svfloat32_t _rad, svfloat32_t _h)
{
  /*
  a = 495.0 / (256.0 * M_PI * CUBE(h));
  q = r / h;
  w = 0.0;
  if(q <= 2) w = SQR(CUBE(1.0 - 0.5 * q)) * (SQR(q) * 35.0 / 12.0 + 3.0 * q + 1.0);
  return a * w;
  */
  svfloat32_t _zero = svdup_f32(0.0f);
  svfloat32_t _two = svdup_f32(2.0f);

  svfloat32_t _q = svdiv_x(pg, _rad, _h);                // CUBE(h)
  svfloat32_t _a = svmul_x(pg, _h, svmul_x(pg, _h, _h)); // CUBE(h)
  _a = svmul_x(pg, _a, (float32_t)(256.0f * M_PI));
  _a = svdiv_x(pg, svdup_f32(495.0f), _a);

  svfloat32_t _tmp1, _tmp2, _tmp3;
  _tmp1 = svmsb_x(pg, svdup_f32(0.5f), _q, 1.0f);        // -0.5*q + 1
  _tmp1 = svmul_x(pg, _tmp1, svmul_x(pg, _tmp1, _tmp1)); // CUBE(tmp1)
  _tmp1 = svmul_x(pg, _tmp1, _tmp1);                     // SQR(CUBE(tmp1))

  _tmp3 = svdup_f32(35.0f / 12.0f);
  _tmp2 = svmad_x(pg, _q, svdup_f32(3.0f), 1.0f);
  _tmp2 = svmad_x(pg, svmul_x(pg, _q, _q), _tmp3, _tmp2);

  _tmp1 = svmul_x(pg, _tmp1, _tmp2);
  _tmp1 = svsel(svcmple(pg, _q, _two), _tmp1, _zero); // if(q <= 2) t = t else t = 0
  return svmul_x(pg, _a, _tmp1);
}

static inline svfloat32_t WendlandQuinticC4_dwdq_sve(svbool_t pg, svfloat32_t _rad, svfloat32_t _h)
{
  /*
  a = 495.0 / (256.0 * M_PI * CUBE(h));
  q = r / h;
  dwdq = 0.0;
  if(q <= 2) dwdq = (SQR(SQR(1.0 - 0.5 * q)) * (1.0 - 0.5 * q)) * (-14.0 / 3.0) * q * (1.0 + 2.5 * q);
  return a * dwdq;
  */

  svfloat32_t _zero = svdup_f32(0.0f);
  svfloat32_t _two = svdup_f32(2.0f);

  svfloat32_t _q = svdiv_x(pg, _rad, _h);                // CUBE(h)
  svfloat32_t _a = svmul_x(pg, _h, svmul_x(pg, _h, _h)); // CUBE(h)
  _a = svmul_x(pg, _a, (float32_t)(256.0f * M_PI));
  _a = svdiv_x(pg, svdup_f32(495.0f), _a);

  svfloat32_t _tmp1, _tmp2, _tmp3;
  _tmp1 = svmsb_x(pg, svdup_f32(0.5f), _q, 1.0f);        // -0.5*q + 1
  _tmp2 = svmul_x(pg, _tmp1, svmul_x(pg, _tmp1, _tmp1)); // CUBE(tmp1)
  _tmp1 = svmul_x(pg, svmul_x(pg, _tmp1, _tmp1), _tmp2); // SQR(tmp1)*CUBE(tmp1)

  _tmp3 = svmad_x(pg, _q, svdup_f32(2.5f), 1.0f);
  _tmp2 = svdup_f32(-14.0f / 3.0f);
  _tmp2 = svmul_x(pg, svmul_x(pg, _tmp2, _q), _tmp3); // (-14.0 / 3.0) * q * (1.0 + 2.5 * q)

  _tmp1 = svmul_x(pg, _tmp1, _tmp2);
  _tmp1 = svsel(svcmple(pg, _q, _two), _tmp1, _zero); // if(q <= 2) t = t else t = 0
  return svmul_x(pg, _a, _tmp1);
}

static inline svfloat32_t density_kernel_sve(svbool_t pg, svfloat32_t _rad, svfloat32_t _h)
{
  return WendlandQuinticC4_sve(pg, _rad, _h);
}

static inline svfloat32_t density_grad_kernel_sve(svbool_t pg, svfloat32_t _rad, svfloat32_t _h)
{
  /* Multiply this by the distance in each axes. */
  // if(r < 1e-30) return 0.0;
  svfloat32_t _irh = svdiv_f32_x(pg, svdup_f32(1.0f), svmul_f32_x(pg, _rad, _h));
  _irh = svsel_f32(svcmpge(pg, _rad, svdup_f32(1.0e-30f)), _irh, svdup_f32(0.0f)); // irh = (rad >= 1e-30) ? irh : 0.0;
  return svmul_x(pg, WendlandQuinticC4_dwdq_sve(pg, _rad, _h), _irh);
}

static inline void calc_grav_S2_with_sve_i16_j1_unroll16_with_potdens(struct iptcl *ip, const int ni,
                                                                      struct jlist_t *jp, const int nj, float sft2_pp,
                                                                      float sft_pm)
{
  int i = 0;
  svbool_t pg = svwhilelt_b32(i, ni);
  svbool_t pg_j = svwhilelt_b32(0, 4);

  float rsft_pm = 1.0 / sft_pm;
  float h = 0.2 / (float)(NPART_1D);

  int nj16, mod_nj16;
  mod_nj16 = nj % 16;
  nj16 = nj - mod_nj16;

  do {
    svfloat32_t _xacc = svdup_f32(0.0f);
    svfloat32_t _yacc = svdup_f32(0.0f);
    svfloat32_t _zacc = svdup_f32(0.0f);
    svfloat32_t _pot = svdup_f32(0.0f);

    svfloat32_t _dens = svdup_f32(0.0f);
    svfloat32_t _xdg = svdup_f32(0.0f);
    svfloat32_t _ydg = svdup_f32(0.0f);
    svfloat32_t _zdg = svdup_f32(0.0f);

    svfloat32_t _xi = svld1(pg, ip->xpos + i);
    svfloat32_t _yi = svld1(pg, ip->ypos + i);
    svfloat32_t _zi = svld1(pg, ip->zpos + i);

    svfloat32_t _h = svld1(pg, ip->smooth_l + i);
    _h = svsel(svcmplt(pg, _h, 0.0f), svdup_f32(h), _h); // _h = (_h < 0) ? h : _h;

    // svfloat32_t _eps2 = svld1(pg, ip->eps+i);
    //_eps2 = svmul_x(pg, _eps2, _eps2);
    svfloat32_t _eps2 = svdup_f32(sft2_pp);
    svfloat32_t _rsft_pm = svdup_f32(rsft_pm);

    svfloat32_t _mj, _xj, _yj, _zj;
    svfloat32_t _dx, _dy, _dz;
    svfloat32_t _rsq, _rad, _gfact, _mrinv3;
    svfloat32_t _rinv;
    svfloat32_t _dens_grad;

    svfloat32_t _tiny = svdup_f32(1.0e-30f);

    for(int j = 0; j < nj16; j += 16) {
      _mj = svld1(pg_j, (float32_t *)(jp + j));
      _xj = svdup_lane(_mj, 0); // xpos
      _yj = svdup_lane(_mj, 1); // ypos
      _zj = svdup_lane(_mj, 2); // zpos
      _mj = svdup_lane(_mj, 3); // mass

      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);

      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);

      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
      _gfact = gfactor_acc_sve(pg, _rad, _rsft_pm);
      _rinv = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2)); // 1.0/sqrt(r^2 + eps^2)

      _mrinv3 = svmul_x(pg, svmul_x(pg, _rinv, _rinv), _rinv);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);

      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      /* for potential */
      // pot = pot - mj*gfact_pot*(1/(r+e))
      _gfact = gfactor_pot_acc_sve(pg, _rad, _rsft_pm);
      _gfact = svsel(svcmpge(pg, _rsq, _tiny), _gfact,
                     svdup_f32(0.0f)); // gfact = (rsq >= 1e-30) ? gfact : 0.0;
      _pot = svmsb_x(pg, svmul_x(pg, _mj, _gfact), _rinv, _pot);

      /* for density */
      // dens = dens + mj * density_kernel // calculate i=j particle.
      // dens_grad[3] = dens_grad[3] + mj * dr[3] * density_grad_kernel
      _dens = svmad_x(pg, _mj, density_kernel_sve(pg, _rad, _h), _dens);
      _dens_grad = svmul_x(pg, _mj, density_grad_kernel_sve(pg, _rad, _h));
      _xdg = svmad_x(pg, _dens_grad, _dx, _xdg);
      _ydg = svmad_x(pg, _dens_grad, _dy, _ydg);
      _zdg = svmad_x(pg, _dens_grad, _dz, _zdg);

      _mj = svld1(pg_j, (float32_t *)(jp + j + 1));
      _xj = svdup_lane(_mj, 0);
      _yj = svdup_lane(_mj, 1);
      _zj = svdup_lane(_mj, 2);
      _mj = svdup_lane(_mj, 3);
      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);
      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
      _gfact = gfactor_acc_sve(pg, _rad, _rsft_pm);
      _rinv = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
      _mrinv3 = svmul_x(pg, svmul_x(pg, _rinv, _rinv), _rinv);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);
      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      // pot = pot - mj*gfact_pot*(1/(r+e))
      _gfact = gfactor_pot_acc_sve(pg, _rad, _rsft_pm);
      _gfact = svsel(svcmpge(pg, _rsq, _tiny), _gfact,
                     svdup_f32(0.0f)); // gfact = (rsq >= 1e-30) ? gfact : 0.0;
      _pot = svmsb_x(pg, svmul_x(pg, _mj, _gfact), _rinv, _pot);

      /* for density */
      _dens = svmad_x(pg, _mj, density_kernel_sve(pg, _rad, _h), _dens);
      _dens_grad = svmul_x(pg, _mj, density_grad_kernel_sve(pg, _rad, _h));

      _xdg = svmad_x(pg, _dens_grad, _dx, _xdg);
      _ydg = svmad_x(pg, _dens_grad, _dy, _ydg);
      _zdg = svmad_x(pg, _dens_grad, _dz, _zdg);

      _mj = svld1(pg_j, (float32_t *)(jp + j + 2));
      _xj = svdup_lane(_mj, 0);
      _yj = svdup_lane(_mj, 1);
      _zj = svdup_lane(_mj, 2);
      _mj = svdup_lane(_mj, 3);
      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);
      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
      _gfact = gfactor_acc_sve(pg, _rad, _rsft_pm);
      _rinv = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
      _mrinv3 = svmul_x(pg, svmul_x(pg, _rinv, _rinv), _rinv);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);
      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      // pot = pot - mj*gfact_pot*(1/(r+e))
      _gfact = gfactor_pot_acc_sve(pg, _rad, _rsft_pm);
      _gfact = svsel(svcmpge(pg, _rsq, _tiny), _gfact,
                     svdup_f32(0.0f)); // gfact = (rsq >= 1e-30) ? gfact : 0.0;
      _pot = svmsb_x(pg, svmul_x(pg, _mj, _gfact), _rinv, _pot);

      /* for density */
      _dens = svmad_x(pg, _mj, density_kernel_sve(pg, _rad, _h), _dens);
      _dens_grad = svmul_x(pg, _mj, density_grad_kernel_sve(pg, _rad, _h));

      _xdg = svmad_x(pg, _dens_grad, _dx, _xdg);
      _ydg = svmad_x(pg, _dens_grad, _dy, _ydg);
      _zdg = svmad_x(pg, _dens_grad, _dz, _zdg);

      _mj = svld1(pg_j, (float32_t *)(jp + j + 3));
      _xj = svdup_lane(_mj, 0);
      _yj = svdup_lane(_mj, 1);
      _zj = svdup_lane(_mj, 2);
      _mj = svdup_lane(_mj, 3);
      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);
      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
      _gfact = gfactor_acc_sve(pg, _rad, _rsft_pm);
      _rinv = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
      _mrinv3 = svmul_x(pg, svmul_x(pg, _rinv, _rinv), _rinv);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);
      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      // pot = pot - mj*gfact_pot*(1/(r+e))
      _gfact = gfactor_pot_acc_sve(pg, _rad, _rsft_pm);
      _gfact = svsel(svcmpge(pg, _rsq, _tiny), _gfact,
                     svdup_f32(0.0f)); // gfact = (rsq >= 1e-30) ? gfact : 0.0;
      _pot = svmsb_x(pg, svmul_x(pg, _mj, _gfact), _rinv, _pot);

      /* for density */
      _dens = svmad_x(pg, _mj, density_kernel_sve(pg, _rad, _h), _dens);
      _dens_grad = svmul_x(pg, _mj, density_grad_kernel_sve(pg, _rad, _h));

      _xdg = svmad_x(pg, _dens_grad, _dx, _xdg);
      _ydg = svmad_x(pg, _dens_grad, _dy, _ydg);
      _zdg = svmad_x(pg, _dens_grad, _dz, _zdg);

      _mj = svld1(pg_j, (float32_t *)(jp + j + 4));
      _xj = svdup_lane(_mj, 0);
      _yj = svdup_lane(_mj, 1);
      _zj = svdup_lane(_mj, 2);
      _mj = svdup_lane(_mj, 3);
      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);
      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
      _gfact = gfactor_acc_sve(pg, _rad, _rsft_pm);
      _rinv = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
      _mrinv3 = svmul_x(pg, svmul_x(pg, _rinv, _rinv), _rinv);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);
      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      // pot = pot - mj*gfact_pot*(1/(r+e))
      _gfact = gfactor_pot_acc_sve(pg, _rad, _rsft_pm);
      _gfact = svsel(svcmpge(pg, _rsq, _tiny), _gfact,
                     svdup_f32(0.0f)); // gfact = (rsq >= 1e-30) ? gfact : 0.0;
      _pot = svmsb_x(pg, svmul_x(pg, _mj, _gfact), _rinv, _pot);

      /* for density */
      _dens = svmad_x(pg, _mj, density_kernel_sve(pg, _rad, _h), _dens);
      _dens_grad = svmul_x(pg, _mj, density_grad_kernel_sve(pg, _rad, _h));

      _xdg = svmad_x(pg, _dens_grad, _dx, _xdg);
      _ydg = svmad_x(pg, _dens_grad, _dy, _ydg);
      _zdg = svmad_x(pg, _dens_grad, _dz, _zdg);

      _mj = svld1(pg_j, (float32_t *)(jp + j + 5));
      _xj = svdup_lane(_mj, 0);
      _yj = svdup_lane(_mj, 1);
      _zj = svdup_lane(_mj, 2);
      _mj = svdup_lane(_mj, 3);
      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);
      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
      _gfact = gfactor_acc_sve(pg, _rad, _rsft_pm);
      _rinv = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
      _mrinv3 = svmul_x(pg, svmul_x(pg, _rinv, _rinv), _rinv);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);
      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      // pot = pot - mj*gfact_pot*(1/(r+e))
      _gfact = gfactor_pot_acc_sve(pg, _rad, _rsft_pm);
      _gfact = svsel(svcmpge(pg, _rsq, _tiny), _gfact,
                     svdup_f32(0.0f)); // gfact = (rsq >= 1e-30) ? gfact : 0.0;
      _pot = svmsb_x(pg, svmul_x(pg, _mj, _gfact), _rinv, _pot);

      /* for density */
      _dens = svmad_x(pg, _mj, density_kernel_sve(pg, _rad, _h), _dens);
      _dens_grad = svmul_x(pg, _mj, density_grad_kernel_sve(pg, _rad, _h));

      _xdg = svmad_x(pg, _dens_grad, _dx, _xdg);
      _ydg = svmad_x(pg, _dens_grad, _dy, _ydg);
      _zdg = svmad_x(pg, _dens_grad, _dz, _zdg);

      _mj = svld1(pg_j, (float32_t *)(jp + j + 6));
      _xj = svdup_lane(_mj, 0);
      _yj = svdup_lane(_mj, 1);
      _zj = svdup_lane(_mj, 2);
      _mj = svdup_lane(_mj, 3);
      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);
      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
      _gfact = gfactor_acc_sve(pg, _rad, _rsft_pm);
      _rinv = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
      _mrinv3 = svmul_x(pg, svmul_x(pg, _rinv, _rinv), _rinv);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);
      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      // pot = pot - mj*gfact_pot*(1/(r+e))
      _gfact = gfactor_pot_acc_sve(pg, _rad, _rsft_pm);
      _gfact = svsel(svcmpge(pg, _rsq, _tiny), _gfact,
                     svdup_f32(0.0f)); // gfact = (rsq >= 1e-30) ? gfact : 0.0;
      _pot = svmsb_x(pg, svmul_x(pg, _mj, _gfact), _rinv, _pot);

      /* for density */
      _dens = svmad_x(pg, _mj, density_kernel_sve(pg, _rad, _h), _dens);
      _dens_grad = svmul_x(pg, _mj, density_grad_kernel_sve(pg, _rad, _h));

      _xdg = svmad_x(pg, _dens_grad, _dx, _xdg);
      _ydg = svmad_x(pg, _dens_grad, _dy, _ydg);
      _zdg = svmad_x(pg, _dens_grad, _dz, _zdg);

      _mj = svld1(pg_j, (float32_t *)(jp + j + 7));
      _xj = svdup_lane(_mj, 0);
      _yj = svdup_lane(_mj, 1);
      _zj = svdup_lane(_mj, 2);
      _mj = svdup_lane(_mj, 3);
      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);
      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
      _gfact = gfactor_acc_sve(pg, _rad, _rsft_pm);
      _rinv = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
      _mrinv3 = svmul_x(pg, svmul_x(pg, _rinv, _rinv), _rinv);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);
      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      // pot = pot - mj*gfact_pot*(1/(r+e))
      _gfact = gfactor_pot_acc_sve(pg, _rad, _rsft_pm);
      _gfact = svsel(svcmpge(pg, _rsq, _tiny), _gfact,
                     svdup_f32(0.0f)); // gfact = (rsq >= 1e-30) ? gfact : 0.0;
      _pot = svmsb_x(pg, svmul_x(pg, _mj, _gfact), _rinv, _pot);

      /* for density */
      _dens = svmad_x(pg, _mj, density_kernel_sve(pg, _rad, _h), _dens);
      _dens_grad = svmul_x(pg, _mj, density_grad_kernel_sve(pg, _rad, _h));

      _xdg = svmad_x(pg, _dens_grad, _dx, _xdg);
      _ydg = svmad_x(pg, _dens_grad, _dy, _ydg);
      _zdg = svmad_x(pg, _dens_grad, _dz, _zdg);

      _mj = svld1(pg_j, (float32_t *)(jp + j + 8));
      _xj = svdup_lane(_mj, 0);
      _yj = svdup_lane(_mj, 1);
      _zj = svdup_lane(_mj, 2);
      _mj = svdup_lane(_mj, 3);
      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);
      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
      _gfact = gfactor_acc_sve(pg, _rad, _rsft_pm);
      _rinv = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
      _mrinv3 = svmul_x(pg, svmul_x(pg, _rinv, _rinv), _rinv);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);
      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      // pot = pot - mj*gfact_pot*(1/(r+e))
      _gfact = gfactor_pot_acc_sve(pg, _rad, _rsft_pm);
      _gfact = svsel(svcmpge(pg, _rsq, _tiny), _gfact,
                     svdup_f32(0.0f)); // gfact = (rsq >= 1e-30) ? gfact : 0.0;
      _pot = svmsb_x(pg, svmul_x(pg, _mj, _gfact), _rinv, _pot);

      /* for density */
      _dens = svmad_x(pg, _mj, density_kernel_sve(pg, _rad, _h), _dens);
      _dens_grad = svmul_x(pg, _mj, density_grad_kernel_sve(pg, _rad, _h));

      _xdg = svmad_x(pg, _dens_grad, _dx, _xdg);
      _ydg = svmad_x(pg, _dens_grad, _dy, _ydg);
      _zdg = svmad_x(pg, _dens_grad, _dz, _zdg);

      _mj = svld1(pg_j, (float32_t *)(jp + j + 9));
      _xj = svdup_lane(_mj, 0);
      _yj = svdup_lane(_mj, 1);
      _zj = svdup_lane(_mj, 2);
      _mj = svdup_lane(_mj, 3);
      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);
      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
      _gfact = gfactor_acc_sve(pg, _rad, _rsft_pm);
      _rinv = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
      _mrinv3 = svmul_x(pg, svmul_x(pg, _rinv, _rinv), _rinv);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);
      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      // pot = pot - mj*gfact_pot*(1/(r+e))
      _gfact = gfactor_pot_acc_sve(pg, _rad, _rsft_pm);
      _gfact = svsel(svcmpge(pg, _rsq, _tiny), _gfact,
                     svdup_f32(0.0f)); // gfact = (rsq >= 1e-30) ? gfact : 0.0;
      _pot = svmsb_x(pg, svmul_x(pg, _mj, _gfact), _rinv, _pot);

      /* for density */
      _dens = svmad_x(pg, _mj, density_kernel_sve(pg, _rad, _h), _dens);
      _dens_grad = svmul_x(pg, _mj, density_grad_kernel_sve(pg, _rad, _h));

      _xdg = svmad_x(pg, _dens_grad, _dx, _xdg);
      _ydg = svmad_x(pg, _dens_grad, _dy, _ydg);
      _zdg = svmad_x(pg, _dens_grad, _dz, _zdg);

      _mj = svld1(pg_j, (float32_t *)(jp + j + 10));
      _xj = svdup_lane(_mj, 0);
      _yj = svdup_lane(_mj, 1);
      _zj = svdup_lane(_mj, 2);
      _mj = svdup_lane(_mj, 3);
      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);
      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
      _gfact = gfactor_acc_sve(pg, _rad, _rsft_pm);
      _rinv = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
      _mrinv3 = svmul_x(pg, svmul_x(pg, _rinv, _rinv), _rinv);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);
      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      // pot = pot - mj*gfact_pot*(1/(r+e))
      _gfact = gfactor_pot_acc_sve(pg, _rad, _rsft_pm);
      _gfact = svsel(svcmpge(pg, _rsq, _tiny), _gfact,
                     svdup_f32(0.0f)); // gfact = (rsq >= 1e-30) ? gfact : 0.0;
      _pot = svmsb_x(pg, svmul_x(pg, _mj, _gfact), _rinv, _pot);

      /* for density */
      _dens = svmad_x(pg, _mj, density_kernel_sve(pg, _rad, _h), _dens);
      _dens_grad = svmul_x(pg, _mj, density_grad_kernel_sve(pg, _rad, _h));

      _xdg = svmad_x(pg, _dens_grad, _dx, _xdg);
      _ydg = svmad_x(pg, _dens_grad, _dy, _ydg);
      _zdg = svmad_x(pg, _dens_grad, _dz, _zdg);

      _mj = svld1(pg_j, (float32_t *)(jp + j + 11));
      _xj = svdup_lane(_mj, 0);
      _yj = svdup_lane(_mj, 1);
      _zj = svdup_lane(_mj, 2);
      _mj = svdup_lane(_mj, 3);
      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);
      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
      _gfact = gfactor_acc_sve(pg, _rad, _rsft_pm);
      _rinv = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
      _mrinv3 = svmul_x(pg, svmul_x(pg, _rinv, _rinv), _rinv);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);
      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      // pot = pot - mj*gfact_pot*(1/(r+e))
      _gfact = gfactor_pot_acc_sve(pg, _rad, _rsft_pm);
      _gfact = svsel(svcmpge(pg, _rsq, _tiny), _gfact,
                     svdup_f32(0.0f)); // gfact = (rsq >= 1e-30) ? gfact : 0.0;
      _pot = svmsb_x(pg, svmul_x(pg, _mj, _gfact), _rinv, _pot);

      /* for density */
      _dens = svmad_x(pg, _mj, density_kernel_sve(pg, _rad, _h), _dens);
      _dens_grad = svmul_x(pg, _mj, density_grad_kernel_sve(pg, _rad, _h));

      _xdg = svmad_x(pg, _dens_grad, _dx, _xdg);
      _ydg = svmad_x(pg, _dens_grad, _dy, _ydg);
      _zdg = svmad_x(pg, _dens_grad, _dz, _zdg);

      _mj = svld1(pg_j, (float32_t *)(jp + j + 12));
      _xj = svdup_lane(_mj, 0);
      _yj = svdup_lane(_mj, 1);
      _zj = svdup_lane(_mj, 2);
      _mj = svdup_lane(_mj, 3);
      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);
      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
      _gfact = gfactor_acc_sve(pg, _rad, _rsft_pm);
      _rinv = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
      _mrinv3 = svmul_x(pg, svmul_x(pg, _rinv, _rinv), _rinv);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);
      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      // pot = pot - mj*gfact_pot*(1/(r+e))
      _gfact = gfactor_pot_acc_sve(pg, _rad, _rsft_pm);
      _gfact = svsel(svcmpge(pg, _rsq, _tiny), _gfact,
                     svdup_f32(0.0f)); // gfact = (rsq >= 1e-30) ? gfact : 0.0;
      _pot = svmsb_x(pg, svmul_x(pg, _mj, _gfact), _rinv, _pot);

      /* for density */
      _dens = svmad_x(pg, _mj, density_kernel_sve(pg, _rad, _h), _dens);
      _dens_grad = svmul_x(pg, _mj, density_grad_kernel_sve(pg, _rad, _h));

      _xdg = svmad_x(pg, _dens_grad, _dx, _xdg);
      _ydg = svmad_x(pg, _dens_grad, _dy, _ydg);
      _zdg = svmad_x(pg, _dens_grad, _dz, _zdg);

      _mj = svld1(pg_j, (float32_t *)(jp + j + 13));
      _xj = svdup_lane(_mj, 0);
      _yj = svdup_lane(_mj, 1);
      _zj = svdup_lane(_mj, 2);
      _mj = svdup_lane(_mj, 3);
      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);
      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
      _gfact = gfactor_acc_sve(pg, _rad, _rsft_pm);
      _rinv = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
      _mrinv3 = svmul_x(pg, svmul_x(pg, _rinv, _rinv), _rinv);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);
      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      // pot = pot - mj*gfact_pot*(1/(r+e))
      _gfact = gfactor_pot_acc_sve(pg, _rad, _rsft_pm);
      _gfact = svsel(svcmpge(pg, _rsq, _tiny), _gfact,
                     svdup_f32(0.0f)); // gfact = (rsq >= 1e-30) ? gfact : 0.0;
      _pot = svmsb_x(pg, svmul_x(pg, _mj, _gfact), _rinv, _pot);

      /* for density */
      _dens = svmad_x(pg, _mj, density_kernel_sve(pg, _rad, _h), _dens);
      _dens_grad = svmul_x(pg, _mj, density_grad_kernel_sve(pg, _rad, _h));

      _xdg = svmad_x(pg, _dens_grad, _dx, _xdg);
      _ydg = svmad_x(pg, _dens_grad, _dy, _ydg);
      _zdg = svmad_x(pg, _dens_grad, _dz, _zdg);

      _mj = svld1(pg_j, (float32_t *)(jp + j + 14));
      _xj = svdup_lane(_mj, 0);
      _yj = svdup_lane(_mj, 1);
      _zj = svdup_lane(_mj, 2);
      _mj = svdup_lane(_mj, 3);
      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);
      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
      _gfact = gfactor_acc_sve(pg, _rad, _rsft_pm);
      _rinv = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
      _mrinv3 = svmul_x(pg, svmul_x(pg, _rinv, _rinv), _rinv);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);
      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      // pot = pot - mj*gfact_pot*(1/(r+e))
      _gfact = gfactor_pot_acc_sve(pg, _rad, _rsft_pm);
      _gfact = svsel(svcmpge(pg, _rsq, _tiny), _gfact,
                     svdup_f32(0.0f)); // gfact = (rsq >= 1e-30) ? gfact : 0.0;
      _pot = svmsb_x(pg, svmul_x(pg, _mj, _gfact), _rinv, _pot);

      /* for density */
      _dens = svmad_x(pg, _mj, density_kernel_sve(pg, _rad, _h), _dens);
      _dens_grad = svmul_x(pg, _mj, density_grad_kernel_sve(pg, _rad, _h));

      _xdg = svmad_x(pg, _dens_grad, _dx, _xdg);
      _ydg = svmad_x(pg, _dens_grad, _dy, _ydg);
      _zdg = svmad_x(pg, _dens_grad, _dz, _zdg);

      _mj = svld1(pg_j, (float32_t *)(jp + j + 15));
      _xj = svdup_lane(_mj, 0);
      _yj = svdup_lane(_mj, 1);
      _zj = svdup_lane(_mj, 2);
      _mj = svdup_lane(_mj, 3);
      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);
      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
      _gfact = gfactor_acc_sve(pg, _rad, _rsft_pm);
      _rinv = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
      _mrinv3 = svmul_x(pg, svmul_x(pg, _rinv, _rinv), _rinv);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);
      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      // pot = pot - mj*gfact_pot*(1/(r+e))
      _gfact = gfactor_pot_acc_sve(pg, _rad, _rsft_pm);
      _gfact = svsel(svcmpge(pg, _rsq, _tiny), _gfact,
                     svdup_f32(0.0f)); // gfact = (rsq >= 1e-30) ? gfact : 0.0;
      _pot = svmsb_x(pg, svmul_x(pg, _mj, _gfact), _rinv, _pot);

      /* for density */
      _dens = svmad_x(pg, _mj, density_kernel_sve(pg, _rad, _h), _dens);
      _dens_grad = svmul_x(pg, _mj, density_grad_kernel_sve(pg, _rad, _h));

      _xdg = svmad_x(pg, _dens_grad, _dx, _xdg);
      _ydg = svmad_x(pg, _dens_grad, _dy, _ydg);
      _zdg = svmad_x(pg, _dens_grad, _dz, _zdg);
    }

    for(int j = nj16; j < nj; j++) {
      _mj = svld1(pg_j, (float32_t *)(jp + j));
      _xj = svdup_lane(_mj, 0); // xpos
      _yj = svdup_lane(_mj, 1); // ypos
      _zj = svdup_lane(_mj, 2); // zpos
      _mj = svdup_lane(_mj, 3); // mass

      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);

      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);

      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
      _gfact = gfactor_acc_sve(pg, _rad, _rsft_pm);
      _rinv = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2)); // 1.0/sqrt(r^2 + eps^2)

      _mrinv3 = svmul_x(pg, svmul_x(pg, _rinv, _rinv), _rinv);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);

      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      // pot = pot - mj*gfact_pot*(1/(r+e))
      _gfact = gfactor_pot_acc_sve(pg, _rad, _rsft_pm);
      _gfact = svsel(svcmpge(pg, _rsq, _tiny), _gfact,
                     svdup_f32(0.0f)); // gfact = (rsq >= 1e-30) ? gfact : 0.0;
      _pot = svmsb_x(pg, svmul_x(pg, _mj, _gfact), _rinv, _pot);

      /* for density */
      _dens = svmad_x(pg, _mj, density_kernel_sve(pg, _rad, _h), _dens);
      _dens_grad = svmul_x(pg, _mj, density_grad_kernel_sve(pg, _rad, _h));

      _xdg = svmad_x(pg, _dens_grad, _dx, _xdg);
      _ydg = svmad_x(pg, _dens_grad, _dy, _ydg);
      _zdg = svmad_x(pg, _dens_grad, _dz, _zdg);
    }

    svst1(pg, ip->xacc + i, _xacc);
    svst1(pg, ip->yacc + i, _yacc);
    svst1(pg, ip->zacc + i, _zacc);

    svst1(pg, ip->pot + i, _pot);

    svst1(pg, ip->dens + i, _dens);
    svst1(pg, ip->xdg + i, _xdg);
    svst1(pg, ip->ydg + i, _ydg);
    svst1(pg, ip->zdg + i, _zdg);

    i += svcntw(); // for word (float)
    pg = svwhilelt_b32(i, ni);
  } while(svptest_any(svptrue_b32(), pg));
}

void calc_grav_simd_thread_with_pot(EPItreepm *iptcl, Result_treepm *ppforce, const int ni, struct jlist_t *jlist,
                                    const int nj, double eps2_pp, double eps_pm)
{
  struct iptcl ip;
  float sft_eps2_pp, sft_eps_pm;
  sft_eps2_pp = eps2_pp;
  sft_eps_pm = eps_pm;

  for(int i = 0; i < ni; i += NIMAX) {
    int nn = NIMAX;
    if((i + NIMAX) > ni) nn = ni - i;

    for(int ii = 0; ii < nn; ii++) {
      int iii = i + ii;
      ip.xpos[ii] = iptcl[iii].pos[0];
      ip.ypos[ii] = iptcl[iii].pos[1];
      ip.zpos[ii] = iptcl[iii].pos[2];
      ip.smooth_l[ii] = iptcl[iii].smooth_l;
    }

    calc_grav_S2_with_sve_i16_j1_unroll16_with_potdens(&ip, nn, jlist, nj, sft_eps2_pp, sft_eps_pm);

    for(int ii = 0; ii < nn; ii++) {
      int iii = i + ii;
      ppforce[iii].acc[0] += ip.xacc[ii];
      ppforce[iii].acc[1] += ip.yacc[ii];
      ppforce[iii].acc[2] += ip.zacc[ii];
      ppforce[iii].pot += ip.pot[ii];

      ppforce[iii].dens += ip.dens[ii];
      ppforce[iii].dens_grad[0] -= ip.xdg[ii];
      ppforce[iii].dens_grad[1] -= ip.ydg[ii];
      ppforce[iii].dens_grad[2] -= ip.zdg[ii];
    }
  }
}

#endif
