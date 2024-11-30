#pragma once

#include <vector>
#include <algorithm>
#include <cstdint>
#include <cmath>

#include "particle.hpp"

template <typename T>
std::vector<T> ngp(const vecpt &ptcl, const int nside, const int threads = 1, const bool periodic = true)
{
  const uint64_t nmesh = nside * nside * nside;
  uint64_t npart = ptcl.size();
  std::vector<T> mesh(nmesh, 0);

#pragma omp parallel for schedule(auto)
  for(uint64_t p = 0; p < npart; p++) {
    double xpos, ypos, zpos; /* normalized coordinate 0.0 < {x,y,z}pos < 1.0 */
    double xt1, dx1;
    int64_t ix, iy, iz;

    xpos = ptcl[p].xpos;
    ypos = ptcl[p].ypos;
    zpos = ptcl[p].zpos;

    xt1 = xpos * (double)nside;
    ix = (int)(xt1);

    if(!periodic)
      if(ix < 0 || ix > nside - 1) continue;

    xt1 = ypos * (double)nside;
    iy = (int)(xt1);

    if(!periodic)
      if(iy < 0 || iy > nside - 1) continue;

    xt1 = zpos * (double)nside;
    iz = (int)(xt1);

    if(!periodic)
      if(iz < 0 || iz > nside - 1) continue;

    // double mass = ptcl[p].mass;
    double mass = 1.0;
    const int64_t index = iz + nside * (iy + nside * ix);

#pragma omp atomic
    mesh[index] += mass;

  } // iptcl

  return mesh;
}

template <typename T>
std::vector<T> cic(const vecpt &ptcl, const int nside, const int threads = 1, const bool periodic = true)
{
  const uint64_t nmesh = nside * nside * nside;
  uint64_t npart = ptcl.size();
  std::vector<T> mesh(nmesh, 0);

#pragma omp parallel for schedule(auto)
  for(uint64_t p = 0; p < npart; p++) {
    double xpos, ypos, zpos; /* normalized coordinate 0.0 < {x,y,z}pos < 1.0 */
    double xt1, dx1;
    double wix, wix_nbr, wiy, wiy_nbr, wiz, wiz_nbr;
    int64_t iwx, iwx_nbr, iwy, iwy_nbr, iwz, iwz_nbr;

    xpos = ptcl[p].xpos;
    ypos = ptcl[p].ypos;
    zpos = ptcl[p].zpos;

    xt1 = xpos * (double)nside - 0.5;
    iwx = (int)(xt1 + 0.5);                      // 粒子が所属するindex(メッシュ中心ベース)
    dx1 = xt1 - (double)iwx;                     // メッシュ中心からの距離 [-0.5,0.5]。負なら左側。正なら右側
    iwx_nbr = (dx1 < 0) ? (iwx - 1) : (iwx + 1); // 粒子がメッシュの左側なら隣接メッシュはマイナス側。右側なら逆。
    wix_nbr = fabs(dx1);                         // 隣接メッシュの重み
    wix = 1.0 - wix_nbr;                         // 自身の重み

    if(periodic) {
      if(iwx_nbr < 0) iwx_nbr += nside;
      if(iwx_nbr >= nside) iwx_nbr -= nside;
    } else {
      if(iwx < 0 || iwx > nside - 1) continue;
    }

    xt1 = ypos * (double)nside - 0.5;
    iwy = (int)(xt1 + 0.5);
    dx1 = xt1 - (double)iwy;
    iwy_nbr = (dx1 < 0) ? (iwy - 1) : (iwy + 1);
    wiy_nbr = fabs(dx1);
    wiy = 1.0 - wiy_nbr;

    if(periodic) {
      if(iwy_nbr < 0) iwy_nbr += nside;
      if(iwy_nbr >= nside) iwy_nbr -= nside;
    } else {
      if(iwy < 0 || iwy > nside - 1) continue;
    }

    xt1 = zpos * (double)nside - 0.5;
    iwz = (int)(xt1 + 0.5);
    dx1 = xt1 - (double)iwz;
    iwz_nbr = (dx1 < 0) ? (iwz - 1) : (iwz + 1);
    wiz_nbr = fabs(dx1);
    wiz = 1.0 - wiz_nbr;

    if(periodic) {
      if(iwz_nbr < 0) iwz_nbr += nside;
      if(iwz_nbr >= nside) iwz_nbr -= nside;
    } else {
      if(iwz < 0 || iwz > nside - 1) continue;
    }

    // double mass = ptcl[p].mass;
    double mass = 1.0;

    int64_t iwxs[2] = {iwx, iwx_nbr};
    int64_t iwys[2] = {iwy, iwy_nbr};
    int64_t iwzs[2] = {iwz, iwz_nbr};

    double wixs[2] = {wix, wix_nbr};
    double wiys[2] = {wiy, wiy_nbr};
    double wizs[2] = {wiz, wiz_nbr};

    for(int ix = 0; ix < 2; ix++) {
      for(int iy = 0; iy < 2; iy++) {
        for(int iz = 0; iz < 2; iz++) {
          const int64_t index = iwzs[iz] + nside * (iwys[iy] + nside * iwxs[ix]);
          double weight = wixs[ix] * wiys[iy] * wizs[iz];
#pragma omp atomic
          mesh[index] += weight * mass;
        }
      }
    }
  } // iptcl

  return mesh;
}

template <typename T>
std::vector<T> tsc(const vecpt &ptcl, const int nside, const int threads = 1, const bool periodic = true)
{
  const uint64_t nmesh = nside * nside * nside;
  uint64_t npart = ptcl.size();
  std::vector<T> mesh(nmesh, 0);

#pragma omp parallel for schedule(auto)
  for(uint64_t p = 0; p < npart; p++) {
    double xpos, ypos, zpos; /* normalized coordinate 0.0 < {x,y,z}pos < 1.0 */
    double xt1, dx1;
    double wi11, wi21, wi31, wi12, wi22, wi32, wi13, wi23, wi33;
    int64_t iw11, iw21, iw31, iw12, iw22, iw32, iw13, iw23, iw33;

    xpos = ptcl[p].xpos;
    ypos = ptcl[p].ypos;
    zpos = ptcl[p].zpos;

    xt1 = xpos * (double)nside - 0.5;
    iw21 = (int)(xt1 + 0.5);
    dx1 = xt1 - (double)iw21;
    wi11 = 0.5 * (0.5 - dx1) * (0.5 - dx1);
    wi21 = 0.75 - dx1 * dx1;
    wi31 = 0.5 * (0.5 + dx1) * (0.5 + dx1);
    iw11 = iw21 - 1;
    iw31 = iw21 + 1;

    if(periodic) {
      if(iw21 == 0) {
        iw11 = nside - 1;
      } else if(iw21 == nside - 1) {
        iw31 = 0;
      } else if(iw21 == nside) {
        iw21 = 0;
        iw31 = 1;
      }
    } else {
      if(iw21 < 0 || iw21 > nside - 1) continue;
    }

    xt1 = ypos * (double)nside - 0.5;
    iw22 = (int)(xt1 + 0.5);
    dx1 = xt1 - (double)iw22;
    wi12 = 0.5 * (0.5 - dx1) * (0.5 - dx1);
    wi22 = 0.75 - dx1 * dx1;
    wi32 = 0.5 * (0.5 + dx1) * (0.5 + dx1);
    iw12 = iw22 - 1;
    iw32 = iw22 + 1;

    if(periodic) {
      if(iw22 == 0) {
        iw12 = nside - 1;
      } else if(iw22 == nside - 1) {
        iw32 = 0;
      } else if(iw22 == nside) {
        iw22 = 0;
        iw32 = 1;
      }
    } else {
      if(iw22 < 0 || iw22 > nside - 1) continue;
    }

    xt1 = zpos * (double)nside - 0.5;
    iw23 = (int)(xt1 + 0.5);
    dx1 = xt1 - (double)iw23;
    wi13 = 0.5 * (0.5 - dx1) * (0.5 - dx1);
    wi23 = 0.75 - dx1 * dx1;
    wi33 = 0.5 * (0.5 + dx1) * (0.5 + dx1);
    iw13 = iw23 - 1;
    iw33 = iw23 + 1;

    if(periodic) {
      if(iw23 == 0) {
        iw13 = nside - 1;
      } else if(iw23 == nside - 1) {
        iw33 = 0;
      } else if(iw23 == nside) {
        iw23 = 0;
        iw33 = 1;
      }
    } else {
      if(iw23 < 0 || iw23 > nside - 1) continue;
    }

    // double mass = ptcl[p].mass;
    double mass = 1.0;

    int64_t iwxs[3] = {iw11, iw21, iw31};
    int64_t iwys[3] = {iw12, iw22, iw32};
    int64_t iwzs[3] = {iw13, iw23, iw33};

    double wixs[3] = {wi11, wi21, wi31};
    double wiys[3] = {wi12, wi22, wi32};
    double wizs[3] = {wi13, wi23, wi33};

    for(int ix = 0; ix < 3; ix++) {
      for(int iy = 0; iy < 3; iy++) {
        for(int iz = 0; iz < 3; iz++) {
          const int64_t index = iwzs[iz] + nside * (iwys[iy] + nside * iwxs[ix]);
          double weight = wixs[ix] * wiys[iy] * wizs[iz];
#pragma omp atomic
          mesh[index] += weight * mass;
        }
      }
    }
  } // iptcl

  return mesh;
}
