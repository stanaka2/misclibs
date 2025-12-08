// FCCpx -std=c++17 -Nclang -O3  -mcpu=a64fx+sve -march=armv8.2-a+sve main_w_pot.cpp -o test_sve_w_pot
// FCCpx -std=c++17 -Nclang -O3  -mcpu=a64fx+sve -march=armv8.2-a+sve -DPG6 main_w_pot.cpp -o test_sve_w_pot_pd

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

#define OUTPUT_POT

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static constexpr double TINY = 1.0e-30;

namespace PS
{
using F32 = float;
using F64 = double;

struct F64vec {
  F64 x, y, z;
  F64vec() : x(0.0), y(0.0), z(0.0) {}
  F64vec(F64 _x, F64 _y, F64 _z) : x(_x), y(_y), z(_z) {}
  F64 &operator[](int i) { return (i == 0 ? x : (i == 1 ? y : z)); }
  const F64 &operator[](int i) const { return (i == 0 ? x : (i == 1 ? y : z)); }
};

struct F32vec {
  F32 x, y, z;
  F32vec() : x(0.f), y(0.f), z(0.f) {}
  F32vec(F32 _x, F32 _y, F32 _z) : x(_x), y(_y), z(_z) {}
  F32 &operator[](int i) { return (i == 0 ? x : (i == 1 ? y : z)); }
  const F32 &operator[](int i) const { return (i == 0 ? x : (i == 1 ? y : z)); }
};
} // namespace PS

//====================
// Result / EPItreepm
//====================

struct Result_treepm {
  PS::F32vec acc;
  PS::F32 pot;
  PS::F32 dens;
  PS::F32vec dens_grad;

  Result_treepm() : acc(0.f, 0.f, 0.f), pot(0.f), dens(0.f), dens_grad(0.f, 0.f, 0.f) {}
};

struct EPItreepm {
  PS::F64vec pos;
  PS::F64 mass;
  PS::F64 smooth_l;
};

#ifdef PG6
#include "sve_grav_pd.h"
#else
#include "sve_grav.h"
#endif

inline PS::F64 gfactor_S2(const PS::F64 rad, const PS::F64 eps_pm)
{
  PS::F64 R = 2.0 * rad / eps_pm;
  if(R > 2.0) R = 2.0;

  PS::F64 S = R - 1.0;
  if(S < 0.0) S = 0.0;

  PS::F64 R2 = R * R;
  PS::F64 R3 = R2 * R;

  PS::F64 S2 = S * S;
  PS::F64 S3 = S2 * S;
  PS::F64 S6 = S3 * S3;

  PS::F64 term1 = -1.6 + R2 * (1.6 + R * (-0.5 + R * (0.15 * R - 12.0 / 35.0)));
  PS::F64 term2 = (3.0 / 35.0) + R * ((18.0 / 35.0) + 0.2 * R);

  PS::F64 g = 1.0 + R3 * term1 - S6 * term2;
  return g;
}

inline PS::F64 gfactor_pot_S2(PS::F64 rad, const PS::F64 eps_pm)
{
  /* for potential */

  rad = (rad > eps_pm) ? eps_pm : rad;
  PS::F64 R = 2.0 * rad / eps_pm;
  PS::F64 R2 = R * R;

  PS::F64 f = rad / (70 * eps_pm); // R/140
  PS::F64 f2 = 1.0 / 14.0;
  PS::F64 g, g1, g2;

  g1 = -8.0 + 3.0 * R;
  g1 = -1.0 + f2 * R * g1;
  g1 = 4.0 + R * g1;
  g1 = -8.0 + R2 * g1;
  g1 = 208.0 + 14.0 * R2 * g1;

  g2 = R * (8.0 - R);
  g2 = -1.0 + f2 * g2;
  g2 = -4.0 + R * g2;
  g2 = 20.0 + R * g2;
  g2 = -32.0 + R * g2;
  g2 = 16.0 + R * g2;
  g2 = 128.0 + R * 14.0 * g2;
  g2 = (12.0 / R) + g2;

  if(R <= 1.0) g = g1;
  else g = g2;

  g = 1.0 - f * g;

  if(R > 1.999) g = 0.0; // for numerical stability

  return g;
}

inline PS::F64 WendlandQuinticC4(PS::F64 r, const PS::F64 h)
{
  PS::F64 a = 495.0 / (256.0 * M_PI * h * h * h);
  PS::F64 q = r / h;
  PS::F64 w = 0.0;
  if(q <= 2.0) {
    PS::F64 t = 1.0 - 0.5 * q;
    PS::F64 t3 = t * t * t;
    PS::F64 term = (q * q) * (35.0 / 12.0) + 3.0 * q + 1.0;
    w = (t3 * t3) * term;
  }
  return a * w;
}

inline PS::F64 WendlandQuinticC4_dwdq(PS::F64 r, const PS::F64 h)
{
  PS::F64 a = 495.0 / (256.0 * M_PI * h * h * h);
  PS::F64 q = r / h;
  PS::F64 dwdq = 0.0;
  if(q <= 2.0) {
    PS::F64 t = 1.0 - 0.5 * q;
    PS::F64 t2 = t * t;
    PS::F64 t4 = t2 * t2;
    dwdq = t4 * t * (-14.0 / 3.0) * q * (1.0 + 2.5 * q);
  }
  return a * dwdq;
}

inline PS::F64 density_kernel(PS::F64 r, const PS::F64 h) { return WendlandQuinticC4(r, h); }

inline PS::F64 density_grad_kernel(PS::F64 r, const PS::F64 h)
{
  /* Multiply this by the distance in each axis. */
  if(r < TINY) return 0.0;
  PS::F64 irh = 1.0 / (r * h);
  return WendlandQuinticC4_dwdq(r, h) * irh;
}

void calc_pp_force_simd(EPItreepm *iptcl, Result_treepm *ppforce, int ni, double (*xj)[3], double *mj, int nj,
                        double eps2_pp, double eps_pm)
{
  for(int i = 0; i < ni; i++) {
    ppforce[i].acc = PS::F32vec(0.f, 0.f, 0.f);
    ppforce[i].pot = 0.f;
    ppforce[i].dens = 0.f;
    ppforce[i].dens_grad = PS::F32vec(0.f, 0.f, 0.f);
  }

  jlist_t *jlist = new jlist_t[nj];

  for(int j = 0; j < nj; j++) {
    jlist[j].xpos = xj[j][0];
    jlist[j].ypos = xj[j][1];
    jlist[j].zpos = xj[j][2];
    jlist[j].mass = mj[j];
  }

  calc_grav_simd_thread_with_pot(iptcl, ppforce, ni, jlist, nj, eps2_pp, eps_pm);

  delete[] jlist;
}

void calc_grav_scalar(EPItreepm *iptcl, Result_treepm *ppforce, int ni, double (*xj)[3], double *mj, int nj,
                      double eps2_pp, double eps_pm)
{
  for(int i = 0; i < ni; i++) {
    double xi = iptcl[i].pos.x;
    double yi = iptcl[i].pos.y;
    double zi = iptcl[i].pos.z;

    double ax = 0.0;
    double ay = 0.0;
    double az = 0.0;

    double pot = 0.0;
    double dens = 0.0;
    double dgradx = 0.0, dgrady = 0.0, dgradz = 0.0;

    const double smooth_l = iptcl[i].smooth_l;

    for(int j = 0; j < nj; j++) {
      double dx = xj[j][0] - xi;
      double dy = xj[j][1] - yi;
      double dz = xj[j][2] - zi;

      // rsq = |x_j - x_i|^2
      double rsq = dx * dx + dy * dy + dz * dz;
      double rad = std::sqrt(rsq);

      //--- force (S2) ---
      double gfact = gfactor_S2(rad, eps_pm);
      double rinv = 1.0 / std::sqrt(rsq + eps2_pp);
      double rinv2 = rinv * rinv;
      double rinv3 = rinv2 * rinv;

      double mrinv3 = mj[j] * rinv3 * gfact;

      ax += mrinv3 * dx;
      ay += mrinv3 * dy;
      az += mrinv3 * dz;

      //--- potential (S2) ---
      if(rsq > TINY) {
        double gfact_p = gfactor_pot_S2(rad, eps_pm);
        pot -= gfact_p * mj[j] * rinv; // 本番と同じ符号
      }

      double drx = xi - xj[j][0];
      double dry = yi - xj[j][1];
      double drz = zi - xj[j][2];

      double w = density_kernel(rad, smooth_l);
      double dw = density_grad_kernel(rad, smooth_l);

      dens += mj[j] * w;
      dgradx += mj[j] * drx * dw;
      dgrady += mj[j] * dry * dw;
      dgradz += mj[j] * drz * dw;
    }

    ppforce[i].acc.x = static_cast<PS::F32>(ax);
    ppforce[i].acc.y = static_cast<PS::F32>(ay);
    ppforce[i].acc.z = static_cast<PS::F32>(az);

    ppforce[i].pot = static_cast<PS::F32>(pot);
    ppforce[i].dens = static_cast<PS::F32>(dens);
    ppforce[i].dens_grad.x = static_cast<PS::F32>(dgradx);
    ppforce[i].dens_grad.y = static_cast<PS::F32>(dgrady);
    ppforce[i].dens_grad.z = static_cast<PS::F32>(dgradz);
  }
}

int main()
{
  const int ni = 10000;
  const int nj = 10000;

  std::vector<EPItreepm> ip(ni);
  std::vector<Result_treepm> simd(ni);
  std::vector<Result_treepm> scalar(ni);

  double (*xj)[3] = new double[nj][3];
  double *mj = new double[nj];

  std::mt19937_64 rng(123);
  std::uniform_real_distribution<double> posdist(0.0, 1.0);
  std::uniform_real_distribution<double> mdist(0.5, 1.5);

  for(int i = 0; i < ni; i++) {
    ip[i].pos[0] = posdist(rng);
    ip[i].pos[1] = posdist(rng);
    ip[i].pos[2] = posdist(rng);
    ip[i].mass = mdist(rng);
    ip[i].smooth_l = 0.05;
  }

  for(int j = 0; j < nj; j++) {
    xj[j][0] = posdist(rng);
    xj[j][1] = posdist(rng);
    xj[j][2] = posdist(rng);
    mj[j] = mdist(rng);
  }

  const double eps2_pp = 1.0e-4;
  const double eps_pm = 0.05;

  calc_grav_scalar(ip.data(), scalar.data(), ni, xj, mj, nj, eps2_pp, eps_pm);
  calc_pp_force_simd(ip.data(), simd.data(), ni, xj, mj, nj, eps2_pp, eps_pm);

  for(int i = 0; i < ni; i++) {
    std::printf("%6d %.7e %.7e %.7e , %.7e , %.7e ,  %.7e %.7e %.7e\n", i, simd[i].acc.x, simd[i].acc.y, simd[i].acc.z,
                simd[i].pot, simd[i].dens, simd[i].dens_grad.x, simd[i].dens_grad.y, simd[i].dens_grad.z);

    std::printf("%6d %.7e %.7e %.7e , %.7e , %.7e ,  %.7e %.7e %.7e\n", i, scalar[i].acc.x, scalar[i].acc.y,
                scalar[i].acc.z, scalar[i].pot, scalar[i].dens, scalar[i].dens_grad.x, scalar[i].dens_grad.y,
                scalar[i].dens_grad.z);
  }

  delete[] xj;
  delete[] mj;
  return 0;
}
