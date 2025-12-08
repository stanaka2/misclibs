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
#define NVECT (8) // 8 iptcls * 1 (double)

struct iptcl {
  double xpos[NIMAX] __attribute__((aligned(ALIGNE_SIZE)));
  double ypos[NIMAX] __attribute__((aligned(ALIGNE_SIZE)));
  double zpos[NIMAX] __attribute__((aligned(ALIGNE_SIZE)));

  double xacc[NIMAX] __attribute__((aligned(ALIGNE_SIZE)));
  double yacc[NIMAX] __attribute__((aligned(ALIGNE_SIZE)));
  double zacc[NIMAX] __attribute__((aligned(ALIGNE_SIZE)));

#ifdef OUTPUT_POT
  double pot[NIMAX] __attribute__((aligned(ALIGNE_SIZE)));
  double dens[NIMAX] __attribute__((aligned(ALIGNE_SIZE)));
  double xdg[NIMAX] __attribute__((aligned(ALIGNE_SIZE)));
  double ydg[NIMAX] __attribute__((aligned(ALIGNE_SIZE)));
  double zdg[NIMAX] __attribute__((aligned(ALIGNE_SIZE)));
  double smooth_l[NIMAX] __attribute__((aligned(ALIGNE_SIZE)));
#endif
};

struct jlist_t {
  double xpos, ypos, zpos;
  double mass;
};

static inline svfloat64_t rsqrt_acc_d(svbool_t pg, svfloat64_t _x)
{
  svfloat64_t _res = svrsqrte(_x);
  _res = svmul_x(pg, svrsqrts(svmul_x(pg, _res, _res), _x), _res);
  _res = svmul_x(pg, svrsqrts(svmul_x(pg, _res, _res), _x), _res);
  _res = svmul_x(pg, svrsqrts(svmul_x(pg, _res, _res), _x), _res);
  _res = svmul_x(pg, svrsqrts(svmul_x(pg, _res, _res), _x), _res);
  return _res;
}

static inline svfloat64_t gfactor_acc_sve_d(svbool_t pg, svfloat64_t _rad, svfloat64_t _rsft_pm)
{
  const double zero = 0.0;
  const double one = 1.0;
  const double two = 2.0;

  const double coeff0 = 0.15;
  const double coeff1 = 12.0 / 35.0;
  const double coeff2 = -0.5;
  const double coeff3 = 1.6;
  const double coeff4 = 0.2;
  const double coeff5 = 18.0 / 35.0;
  const double coeff6 = 3.0 / 35.0;

  svfloat64_t _zero = svdup_f64(zero);
  svfloat64_t _one = svdup_f64(one);
  svfloat64_t _two = svdup_f64(two);

  svfloat64_t _g, _h, _R, _R3, _S;

  _R = svmul_x(pg, svmul_x(pg, _rad, _two), _rsft_pm);
  _R = svmin_x(pg, _R, _two);

  _R3 = svmul_x(pg, svmul_x(pg, _R, _R), _R);

  _S = svmax_x(pg, svsub_x(pg, _R, _one), zero);
  _S = svmul_x(pg, svmul_x(pg, _S, _S), _S); // S^3
  _S = svmul_x(pg, _S, _S);                  // S^6

  _g = svnmsb_x(pg, svdup_f64(coeff0), _R, coeff1); // c0*R - c1
  _g = svmad_x(pg, _g, _R, coeff2);                 // _g*R + c2
  _g = svmad_x(pg, _g, _R, coeff3);                 // _g*R + c3
  _g = svmul_x(pg, _g, _R);                         // _g*R
  _g = svnmsb_x(pg, _g, _R, coeff3);                // _g*R - c3
  _g = svmul_x(pg, _g, _R3);                        // _g*R3

  _h = svmad_x(pg, svdup_f64(coeff4), _R, coeff5); // c4*R + c5
  _h = svmad_x(pg, _h, _R, coeff6);                // _h*R + c6
  _h = svmul_x(pg, _h, _S);                        // _h*S^6
  _h = svsub_x(pg, _one, _h);                      // 1 - _h

  _g = svadd_x(pg, _g, _h);

  return _g;
}

static inline svfloat64_t gfactor_pot_acc_sve_d(svbool_t pg, svfloat64_t _rad, svfloat64_t _rsft_pm)
{
  const double zero = 0.0;
  const double one = 1.0;
  const double two = 2.0;

  svfloat64_t _zero = svdup_f64(zero);
  svfloat64_t _one = svdup_f64(one);
  svfloat64_t _two = svdup_f64(two);

  svfloat64_t _c4 = svdup_f64(4.0);
  svfloat64_t _c8 = svdup_f64(8.0);
  svfloat64_t _c14 = svdup_f64(14.0);
  svfloat64_t _f = svdup_f64(1.0 / 14.0);

  svfloat64_t _g1, _g2, _R, _R2;

  _rad = svmin_x(pg, _rad, svdiv_x(pg, _one, _rsft_pm)); // r = min(r, eps_pm)
  _R = svmul_x(pg, svmul_x(pg, _rad, _two), _rsft_pm);   // R = 2 r / eps_pm
  _R2 = svmul_x(pg, _R, _R);

  _g1 = svnmsb_x(pg, svdup_f64(3.0), _R, _c8);           // -8 + 3R
  _g1 = svnmsb_x(pg, svmul_x(pg, _f, _g1), _R, _one);    // -1 + R*(f*g1)
  _g1 = svmad_x(pg, _R, _g1, _c4);                       // 4 + R*g1
  _g1 = svnmsb_x(pg, _R2, _g1, _c8);                     // -8 + R2*g1
  _g1 = svmad_x(pg, _R2, svmul_x(pg, _c14, _g1), 208.0); // 208 + R2*(14*g1)

  _g2 = svmul_x(pg, _R, svsub_x(pg, _c8, _R));              // R(8-R)
  _g2 = svnmsb_x(pg, _f, _g2, _one);                        // -1 + f*g2
  _g2 = svnmsb_x(pg, _R, _g2, _c4);                         // -4 + R*g2
  _g2 = svmad_x(pg, _R, _g2, 20.0);                         // 20 + R*g2
  _g2 = svnmsb_x(pg, _R, _g2, 32.0);                        // -32 + R*g2
  _g2 = svmad_x(pg, _R, _g2, 16.0);                         // 16 + R*g2
  _g2 = svmad_x(pg, _c14, svmul_x(pg, _R, _g2), 128.0);     // 128 + 14 R g2
  _g2 = svadd_x(pg, _g2, svdiv_x(pg, svdup_f64(12.0), _R)); // 12/R + g2

  _f = svmul_x(pg, svdiv_x(pg, _rad, 70.0), _rsft_pm); // f = r / (70 eps_pm)
  _g1 = svsel(svcmple(pg, _R, _one), _g1, _g2);        // R<=1 ? g1 : g2
  _g1 = svmsb_x(pg, _f, _g1, _one);                    // 1 - f g

  _g1 = svsel(svcmpge(pg, _R, svdup_f64(1.99999)), _zero, _g1);

  return _g1;
}

static inline void calc_grav_S2_with_sve_i8_j1_d(struct iptcl *ip, const int64_t ni, struct jlist_t *jp,
                                                 const int64_t nj, double sft2_pp, double sft_pm)
{
  int64_t i = 0;
  svbool_t pg = svwhilelt_b64(i, ni);

  const double rsft_pm = 1.0 / sft_pm;

  do {
    svfloat64_t _xacc = svdup_f64(0.0);
    svfloat64_t _yacc = svdup_f64(0.0);
    svfloat64_t _zacc = svdup_f64(0.0);

    svfloat64_t _xi = svld1(pg, ip->xpos + i);
    svfloat64_t _yi = svld1(pg, ip->ypos + i);
    svfloat64_t _zi = svld1(pg, ip->zpos + i);

    svfloat64_t _eps2 = svdup_f64(sft2_pp);
    svfloat64_t _rsft_pm = svdup_f64(rsft_pm);

    for(int64_t j = 0; j < nj; j++) {
      const double xj = jp[j].xpos;
      const double yj = jp[j].ypos;
      const double zj = jp[j].zpos;
      const double mj = jp[j].mass;

      svfloat64_t _xj = svdup_f64(xj);
      svfloat64_t _yj = svdup_f64(yj);
      svfloat64_t _zj = svdup_f64(zj);
      svfloat64_t _mj = svdup_f64(mj);

      svfloat64_t _dx = svsub_x(pg, _xj, _xi);
      svfloat64_t _dy = svsub_x(pg, _yj, _yi);
      svfloat64_t _dz = svsub_x(pg, _zj, _zi);

      svfloat64_t _rsq = svmad_x(pg, _dx, _dx, svdup_f64(DBL_MIN));
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);

      svfloat64_t _rad = svmul_x(pg, rsqrt_acc_d(pg, _rsq), _rsq);
      svfloat64_t _gfact = gfactor_acc_sve_d(pg, _rad, _rsft_pm);
      svfloat64_t _rinv = rsqrt_acc_d(pg, svadd_x(pg, _rsq, _eps2));

      svfloat64_t _mrinv3 = svmul_x(pg, svmul_x(pg, _rinv, _rinv), _rinv);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);

      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);
    }

    svst1(pg, ip->xacc + i, _xacc);
    svst1(pg, ip->yacc + i, _yacc);
    svst1(pg, ip->zacc + i, _zacc);

    i += svcntd();
    pg = svwhilelt_b64(i, ni);
  } while(svptest_any(svptrue_b64(), pg));
}

#ifdef OUTPUT_POT

// SPH kernel (Wendland C4) for density / grad (double)
static inline svfloat64_t WendlandQuinticC4_sve_d(svbool_t pg, svfloat64_t _rad, svfloat64_t _h)
{
  svfloat64_t _zero = svdup_f64(0.0);
  svfloat64_t _two = svdup_f64(2.0);

  svfloat64_t _q = svdiv_x(pg, _rad, _h);                // q = r/h
  svfloat64_t _a = svmul_x(pg, _h, svmul_x(pg, _h, _h)); // h^3
  _a = svmul_x(pg, _a, 256.0 * M_PI);
  _a = svdiv_x(pg, svdup_f64(495.0), _a); // a = 495/(256π h^3)

  svfloat64_t _tmp1, _tmp2;

  _tmp1 = svmsb_x(pg, svdup_f64(0.5), _q, 1.0);          // 1 - 0.5 q
  _tmp1 = svmul_x(pg, _tmp1, svmul_x(pg, _tmp1, _tmp1)); // (1 - 0.5q)^3
  _tmp1 = svmul_x(pg, _tmp1, _tmp1);                     // ( ... )^6

  _tmp2 = svmad_x(pg, _q, svdup_f64(3.0), 1.0);                            // 1 + 3q
  _tmp2 = svmad_x(pg, svmul_x(pg, _q, _q), svdup_f64(35.0 / 12.0), _tmp2); // 35/12 q^2 + 3q +1

  _tmp1 = svmul_x(pg, _tmp1, _tmp2);
  _tmp1 = svsel(svcmple(pg, _q, _two), _tmp1, _zero);

  return svmul_x(pg, _a, _tmp1);
}

static inline svfloat64_t WendlandQuinticC4_dwdq_sve_d(svbool_t pg, svfloat64_t _rad, svfloat64_t _h)
{
  svfloat64_t _zero = svdup_f64(0.0);
  svfloat64_t _two = svdup_f64(2.0);

  svfloat64_t _q = svdiv_x(pg, _rad, _h);
  svfloat64_t _a = svmul_x(pg, _h, svmul_x(pg, _h, _h));
  _a = svmul_x(pg, _a, 256.0 * M_PI);
  _a = svdiv_x(pg, svdup_f64(495.0), _a);

  svfloat64_t _tmp1, _tmp2;

  _tmp1 = svmsb_x(pg, svdup_f64(0.5), _q, 1.0);                    // 1 - 0.5q
  svfloat64_t _t3 = svmul_x(pg, _tmp1, svmul_x(pg, _tmp1, _tmp1)); // (1-0.5q)^3
  _tmp1 = svmul_x(pg, svmul_x(pg, _tmp1, _tmp1), _t3);             // (1-0.5q)^5

  _tmp2 = svmad_x(pg, _q, svdup_f64(2.5), 1.0); // 1 + 2.5 q
  _tmp2 = svmul_x(pg, svmul_x(pg, svdup_f64(-14.0 / 3.0), _q), _tmp2);

  _tmp1 = svmul_x(pg, _tmp1, _tmp2);
  _tmp1 = svsel(svcmple(pg, _q, _two), _tmp1, _zero);

  return svmul_x(pg, _a, _tmp1);
}

static inline svfloat64_t density_kernel_sve_d(svbool_t pg, svfloat64_t _rad, svfloat64_t _h)
{
  return WendlandQuinticC4_sve_d(pg, _rad, _h);
}

static inline svfloat64_t density_grad_kernel_sve_d(svbool_t pg, svfloat64_t _rad, svfloat64_t _h)
{
  // dW/dr = (dW/dq) * (1/h) * (1/r) * r ~ dW/dq * 1/(r h)
  svfloat64_t _one = svdup_f64(1.0);
  svfloat64_t _zero = svdup_f64(0.0);
  svfloat64_t _eps = svdup_f64(1.0e-30);

  svfloat64_t _den = svmul_x(pg, _rad, _h);   // r h
  svfloat64_t _irh = svdiv_x(pg, _one, _den); // 1/(r h)
  _irh = svsel(svcmpge(pg, _rad, _eps), _irh, _zero);

  svfloat64_t _dwdq = WendlandQuinticC4_dwdq_sve_d(pg, _rad, _h);
  return svmul_x(pg, _dwdq, _irh);
}

static inline void calc_grav_S2_with_sve_i8_j1_with_potdens_d(struct iptcl *ip, const int64_t ni, struct jlist_t *jp,
                                                              const int64_t nj, double sft2_pp, double sft_pm)
{
  int64_t i = 0;
  svbool_t pg = svwhilelt_b64(i, ni);

  const double rsft_pm = 1.0 / sft_pm;
  const double h0 = 0.2 / (double)NPART_1D;

  do {
    svfloat64_t _xacc = svdup_f64(0.0);
    svfloat64_t _yacc = svdup_f64(0.0);
    svfloat64_t _zacc = svdup_f64(0.0);
    svfloat64_t _pot = svdup_f64(0.0);

    svfloat64_t _dens = svdup_f64(0.0);
    svfloat64_t _xdg = svdup_f64(0.0);
    svfloat64_t _ydg = svdup_f64(0.0);
    svfloat64_t _zdg = svdup_f64(0.0);

    svfloat64_t _xi = svld1(pg, ip->xpos + i);
    svfloat64_t _yi = svld1(pg, ip->ypos + i);
    svfloat64_t _zi = svld1(pg, ip->zpos + i);

    svfloat64_t _h = svld1(pg, ip->smooth_l + i);
    _h = svsel(svcmplt(pg, _h, svdup_f64(0.0)), svdup_f64(h0), _h);

    svfloat64_t _eps2 = svdup_f64(sft2_pp);
    svfloat64_t _rsft_pm = svdup_f64(rsft_pm);
    svfloat64_t _tiny = svdup_f64(1.0e-30);

    for(int64_t j = 0; j < nj; j++) {
      const double xj = jp[j].xpos;
      const double yj = jp[j].ypos;
      const double zj = jp[j].zpos;
      const double mj = jp[j].mass;

      svfloat64_t _xj = svdup_f64(xj);
      svfloat64_t _yj = svdup_f64(yj);
      svfloat64_t _zj = svdup_f64(zj);
      svfloat64_t _mj = svdup_f64(mj);

      svfloat64_t _dx = svsub_x(pg, _xj, _xi);
      svfloat64_t _dy = svsub_x(pg, _yj, _yi);
      svfloat64_t _dz = svsub_x(pg, _zj, _zi);

      svfloat64_t _rsq = svmad_x(pg, _dx, _dx, svdup_f64(DBL_MIN));
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);

      svfloat64_t _rad = svmul_x(pg, rsqrt_acc_d(pg, _rsq), _rsq);
      svfloat64_t _gfact = gfactor_acc_sve_d(pg, _rad, _rsft_pm);
      svfloat64_t _rinv = rsqrt_acc_d(pg, svadd_x(pg, _rsq, _eps2));

      svfloat64_t _mrinv3 = svmul_x(pg, svmul_x(pg, _rinv, _rinv), _rinv);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);

      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      // potential: pot = pot - mj * g_pot * (1/(r+eps))
      svfloat64_t _gpot = gfactor_pot_acc_sve_d(pg, _rad, _rsft_pm);
      _gpot = svsel(svcmpge(pg, _rsq, _tiny), _gpot, svdup_f64(0.0));
      _pot = svmsb_x(pg, svmul_x(pg, _mj, _gpot), _rinv, _pot);

      // density + grad
      svfloat64_t _kern = density_kernel_sve_d(pg, _rad, _h);
      _dens = svmad_x(pg, _mj, _kern, _dens);

      svfloat64_t _gkern = density_grad_kernel_sve_d(pg, _rad, _h);
      _gkern = svmul_x(pg, _mj, _gkern);

      _xdg = svmad_x(pg, _gkern, _dx, _xdg);
      _ydg = svmad_x(pg, _gkern, _dy, _ydg);
      _zdg = svmad_x(pg, _gkern, _dz, _zdg);
    }

    svst1(pg, ip->xacc + i, _xacc);
    svst1(pg, ip->yacc + i, _yacc);
    svst1(pg, ip->zacc + i, _zacc);

    svst1(pg, ip->pot + i, _pot);
    svst1(pg, ip->dens + i, _dens);
    svst1(pg, ip->xdg + i, _xdg);
    svst1(pg, ip->ydg + i, _ydg);
    svst1(pg, ip->zdg + i, _zdg);

    i += svcntd();
    pg = svwhilelt_b64(i, ni);
  } while(svptest_any(svptrue_b64(), pg));
}
#endif // OUTPUT_POT

void calc_grav_simd_thread(EPItreepm *iptcl, Result_treepm *ppforce, const int ni, struct jlist_t *jlist, const int nj,
                           double eps2_pp, double eps_pm)
{
  struct iptcl ip;
  const double sft_eps2_pp = eps2_pp;
  const double sft_eps_pm = eps_pm;

  for(int i = 0; i < ni; i += NIMAX) {
    int nn = NIMAX;
    if(i + NIMAX > ni) nn = ni - i;

    for(int ii = 0; ii < nn; ii++) {
      const int iii = i + ii;
      ip.xpos[ii] = iptcl[iii].pos[0];
      ip.ypos[ii] = iptcl[iii].pos[1];
      ip.zpos[ii] = iptcl[iii].pos[2];
    }

    calc_grav_S2_with_sve_i8_j1_d(&ip, nn, jlist, nj, sft_eps2_pp, sft_eps_pm);

    for(int ii = 0; ii < nn; ii++) {
      const int iii = i + ii;
      ppforce[iii].acc[0] += ip.xacc[ii];
      ppforce[iii].acc[1] += ip.yacc[ii];
      ppforce[iii].acc[2] += ip.zacc[ii];
    }
  }
}

#ifdef OUTPUT_POT
void calc_grav_simd_thread_with_pot(EPItreepm *iptcl, Result_treepm *ppforce, const int ni, struct jlist_t *jlist,
                                    const int nj, double eps2_pp, double eps_pm)
{
  struct iptcl ip;
  const double sft_eps2_pp = eps2_pp;
  const double sft_eps_pm = eps_pm;

  for(int i = 0; i < ni; i += NIMAX) {
    int nn = NIMAX;
    if(i + NIMAX > ni) nn = ni - i;

    for(int ii = 0; ii < nn; ii++) {
      const int iii = i + ii;
      ip.xpos[ii] = iptcl[iii].pos[0];
      ip.ypos[ii] = iptcl[iii].pos[1];
      ip.zpos[ii] = iptcl[iii].pos[2];
      ip.smooth_l[ii] = iptcl[iii].smooth_l;
    }

    calc_grav_S2_with_sve_i8_j1_with_potdens_d(&ip, nn, jlist, nj, sft_eps2_pp, sft_eps_pm);

    for(int ii = 0; ii < nn; ii++) {
      const int iii = i + ii;
      ppforce[iii].acc[0] += ip.xacc[ii];
      ppforce[iii].acc[1] += ip.yacc[ii];
      ppforce[iii].acc[2] += ip.zacc[ii];

#ifdef OUTPUT_POT
      ppforce[iii].pot += ip.pot[ii];
      ppforce[iii].dens += ip.dens[ii];
      ppforce[iii].dens_grad[0] -= ip.xdg[ii];
      ppforce[iii].dens_grad[1] -= ip.ydg[ii];
      ppforce[iii].dens_grad[2] -= ip.zdg[ii];
#endif
    }
  }
}
#endif // OUTPUT_POT
