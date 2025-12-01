
// g++ -DAVX2 -std=c++17 -O3 -mavx2 -mfma main.cpp -o test_avx2
// g++ -DAVX512 -std=c++17 -O3 -mavx2 -mfma -mavx512f -mavx512dq -mavx512vl main.cpp -o test_avx512

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

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

struct Result_treepm {
  PS::F32vec acc;
  Result_treepm() : acc(0.f, 0.f, 0.f) {}
};

struct EPItreepm {
  PS::F64vec pos;
  PS::F64 mass;
};

#ifdef AVX2
#include "avx_grav_avx2.h"
#elif defined AVX512
#include "avx_grav_avx512.h"
#endif

void calc_pp_force_simd(EPItreepm *iptcl, Result_treepm *ppforce, int ni, double (*xj)[3], double *mj, int nj,
                        double eps2_pp, double eps_pm)
{
  for(int i = 0; i < ni; i++) {
    ppforce[i].acc = PS::F32vec(0.f, 0.f, 0.f);
  }
  calc_grav_simd_thread(iptcl, ppforce, ni, xj, mj, nj, eps2_pp, eps_pm);
}

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

    for(int j = 0; j < nj; j++) {
      double dx = xj[j][0] - xi;
      double dy = xj[j][1] - yi;
      double dz = xj[j][2] - zi;

      double rsq = dx * dx + dy * dy + dz * dz;
      double rad = std::sqrt(rsq);

      double gfact = gfactor_S2(rad, eps_pm);
      double rinv = 1.0 / std::sqrt(rsq + eps2_pp);
      double rinv2 = rinv * rinv;
      double rinv3 = rinv2 * rinv;

      double mrinv3 = mj[j] * rinv3 * gfact;

      ax += mrinv3 * dx;
      ay += mrinv3 * dy;
      az += mrinv3 * dz;
    }

    ppforce[i].acc.x = static_cast<PS::F32>(ax);
    ppforce[i].acc.y = static_cast<PS::F32>(ay);
    ppforce[i].acc.z = static_cast<PS::F32>(az);
  }
}

int main()
{
  const int ni = 10000; // I 粒子数
  const int nj = 1000;  // J 粒子数（適当に同数）

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
    std::printf("%6d  %.7e  %.7e  %.7e , %.7e  %.7e  %.7e\n", i, simd[i].acc.x, simd[i].acc.y, simd[i].acc.z,
                scalar[i].acc.x, scalar[i].acc.y, scalar[i].acc.z);
  }

  delete[] xj;
  delete[] mj;
  return 0;
}
