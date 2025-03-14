/*
g++ -O3 -fopenmp nanoflann_sample_peri.cpp
*/

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

#include <omp.h>
#include "nanoflann.hpp"

struct particle {
  float x, y, z;
  int id;
};

struct Point {
  float x, y, z;
  int id;
};

struct PointCloud {
  int64_t norig;
  int64_t npad = 0;
  std::vector<Point> pts;

  inline size_t kdtree_get_point_count() const { return pts.size(); }
  inline float kdtree_get_pt(const size_t idx, const size_t dim) const
  {
    if(dim == 0) return pts[idx].x;
    else if(dim == 1) return pts[idx].y;
    else return pts[idx].z;
  }

  template <class BBOX>
  bool kdtree_get_bbox(BBOX & /* bb */) const
  {
    return false;
  }
};

template <typename T>
void set_ptcl_to_point(PointCloud &cloud, T &ptcls, const double boxsize, const double pad_size,
                       const double range_wp[6], const bool ptcls_mem_release = true, const bool periodic = true)
{
  using point_type = typename std::decay_t<decltype(cloud.pts)>::value_type;

  uint64_t nptcl(ptcls.size());
  cloud.norig = nptcl;
  cloud.pts.resize(nptcl);

  for(uint64_t ip = 0; ip < nptcl; ip++) {
    cloud.pts[ip].x = ptcls[ip].x;
    cloud.pts[ip].y = ptcls[ip].y;
    cloud.pts[ip].z = ptcls[ip].z;
    cloud.pts[ip].id = ptcls[ip].id;
  }

  if(ptcls_mem_release) {
    ptcls.clear();
    ptcls.shrink_to_fit();
  }

  std::cout << "orig nsize=" << cloud.pts.size() << std::endl;
  std::cout << "orig nsize^(1/3)=" << (int)(pow((double)(cloud.pts.size()), 1.0 / 3.0)) << std::endl;

  int64_t npad = 0;

  //  Duplicate ptcls around global boundaries
  if(periodic) {
    const double lpad = pad_size;
    const double rpad = boxsize - pad_size;

    uint64_t nn = nptcl + npad;
    for(uint64_t ip = 0; ip < nn; ip++) {
      point_type tmp = cloud.pts[ip];

      if(tmp.x < lpad) {
        tmp.x += boxsize;
        cloud.pts.push_back(tmp);
        npad++;
      } else if(tmp.x > rpad) {
        tmp.x -= boxsize;
        cloud.pts.push_back(tmp);
        npad++;
      }
    }

    std::cout << "after-x nsize^(1/3)=" << (int)(pow((double)(nptcl + npad), 1.0 / 3.0)) << std::endl;

    nn = nptcl + npad;
    for(uint64_t ip = 0; ip < nn; ip++) {
      point_type tmp = cloud.pts[ip];

      if(tmp.y < lpad) {
        tmp.y += boxsize;
        cloud.pts.push_back(tmp);
        npad++;
      } else if(tmp.y > rpad) {
        tmp.y -= boxsize;
        cloud.pts.push_back(tmp);
        npad++;
      }
    }

    std::cout << "after-xy nsize^(1/3)=" << (int)(pow((double)(nptcl + npad), 1.0 / 3.0)) << std::endl;

    nn = nptcl + npad;
    for(uint64_t ip = 0; ip < nn; ip++) {
      point_type tmp = cloud.pts[ip];

      if(tmp.z < lpad) {
        tmp.z += boxsize;
        cloud.pts.push_back(tmp);
        npad++;
      } else if(tmp.z > rpad) {
        tmp.z -= boxsize;
        cloud.pts.push_back(tmp);
        npad++;
      }
    }
  }

  std::cout << "after-xyz nsize^(1/3)=" << (int)(pow((double)(nptcl + npad), 1.0 / 3.0)) << std::endl;

  cloud.pts.erase(std::remove_if(cloud.pts.begin(), cloud.pts.end(),
                                 [&range_wp](const point_type &p) {
                                   return ((p.x < range_wp[0]) || (p.x > range_wp[1]) || (p.y < range_wp[2]) ||
                                           (p.y > range_wp[3]) || (p.z < range_wp[4]) || (p.z > range_wp[5]));
                                 }),
                  cloud.pts.end());

  nptcl = cloud.pts.size();
  std::cout << "after timing nsize=" << nptcl << std::endl;
  std::cout << "after timing nsize^(1/3)=" << (int)(pow((double)(nptcl), 1.0 / 3.0)) << std::endl;
}

template <typename T>
void set_ptcl_to_point_mp(PointCloud &cloud, T &ptcls, const double boxsize, const double pad_size,
                          const double range_wp[6], const bool ptcls_mem_release = true, const bool periodic = true)
{
  using point_type = typename std::decay_t<decltype(cloud.pts)>::value_type;
  using point_vec = std::vector<point_type>;

  uint64_t nptcl(ptcls.size());
  cloud.norig = nptcl;
  cloud.pts.resize(nptcl);

#pragma omp parallel for
  for(uint64_t ip = 0; ip < nptcl; ip++) {
    cloud.pts[ip].x = ptcls[ip].x;
    cloud.pts[ip].y = ptcls[ip].y;
    cloud.pts[ip].z = ptcls[ip].z;
    cloud.pts[ip].id = ptcls[ip].id;
  }

  if(ptcls_mem_release) {
    ptcls.clear();
    ptcls.shrink_to_fit();
  }

  std::cout << "orig nsize=" << cloud.pts.size() << std::endl;
  std::cout << "orig nsize^(1/3)=" << (int)(pow((double)(cloud.pts.size()), 1.0 / 3.0)) << std::endl;

  int64_t npad = 0;

  //  Duplicate ptcls around global boundaries
  if(periodic) {
    const double lpad = pad_size;
    const double rpad = boxsize - pad_size;

    uint64_t nn = nptcl + npad;

#pragma omp parallel
    {
      point_vec th_point;
      uint64_t th_npad = 0;

#pragma omp for schedule(auto)
      for(uint64_t ip = 0; ip < nn; ip++) {
        point_type tmp = cloud.pts[ip];

        if(tmp.x < lpad) {
          tmp.x += boxsize;
          th_point.push_back(tmp);
          th_npad++;
        } else if(tmp.x > rpad) {
          tmp.x -= boxsize;
          th_point.push_back(tmp);
          th_npad++;
        }
      }

#pragma omp critical
      {
        npad += th_npad;
        cloud.pts.insert(cloud.pts.end(), th_point.begin(), th_point.end());
      }
    } // omp parallel

    std::cout << "after-x nsize^(1/3)=" << (int)(pow((double)(nptcl + npad), 1.0 / 3.0)) << std::endl;

    nn = nptcl + npad;

#pragma omp parallel
    {
      point_vec th_point;
      uint64_t th_npad = 0;

#pragma omp for schedule(auto)
      for(uint64_t ip = 0; ip < nn; ip++) {
        point_type tmp = cloud.pts[ip];

        if(tmp.y < lpad) {
          tmp.y += boxsize;
          th_point.push_back(tmp);
          th_npad++;
        } else if(tmp.y > rpad) {
          tmp.y -= boxsize;
          th_point.push_back(tmp);
          th_npad++;
        }
      }

#pragma omp critical
      {
        npad += th_npad;
        cloud.pts.insert(cloud.pts.end(), th_point.begin(), th_point.end());
      }
    } // omp parallel

    std::cout << "after-xy nsize^(1/3)=" << (int)(pow((double)(nptcl + npad), 1.0 / 3.0)) << std::endl;

    nn = nptcl + npad;

#pragma omp parallel
    {
      point_vec th_point;
      uint64_t th_npad = 0;

#pragma omp for schedule(auto)
      for(uint64_t ip = 0; ip < nn; ip++) {
        point_type tmp = cloud.pts[ip];

        if(tmp.z < lpad) {
          tmp.z += boxsize;
          th_point.push_back(tmp);
          th_npad++;
        } else if(tmp.z > rpad) {
          tmp.z -= boxsize;
          th_point.push_back(tmp);
          th_npad++;
        }
      }

#pragma omp critical
      {
        npad += th_npad;
        cloud.pts.insert(cloud.pts.end(), th_point.begin(), th_point.end());
      }
    } // omp parallel
  } // periodic

  std::cout << "after-xyz nsize^(1/3)=" << (int)(pow((double)(nptcl + npad), 1.0 / 3.0)) << std::endl;

  cloud.pts.erase(std::remove_if(cloud.pts.begin(), cloud.pts.end(),
                                 [&range_wp](const point_type &p) {
                                   return ((p.x < range_wp[0]) || (p.x > range_wp[1]) || (p.y < range_wp[2]) ||
                                           (p.y > range_wp[3]) || (p.z < range_wp[4]) || (p.z > range_wp[5]));
                                 }),
                  cloud.pts.end());

  nptcl = cloud.pts.size();
  std::cout << "after timing nsize=" << nptcl << std::endl;
  std::cout << "after timing nsize^(1/3)=" << (int)(pow((double)(nptcl), 1.0 / 3.0)) << std::endl;
}

template <typename T>
void set_ptcl_to_point_memsave(PointCloud &cloud, T &ptcls, const double boxsize, const double pad_size,
                               const double range_wp[6], const bool ptcls_mem_release = true,
                               const bool periodic = true)
{
  using point_type = typename std::decay_t<decltype(cloud.pts)>::value_type;

  uint64_t nptcl(ptcls.size());
  cloud.norig = nptcl;
  cloud.pts.resize(nptcl);

  for(uint64_t ip = 0; ip < nptcl; ip++) {
    cloud.pts[ip].x = ptcls[ip].x;
    cloud.pts[ip].y = ptcls[ip].y;
    cloud.pts[ip].z = ptcls[ip].z;
    cloud.pts[ip].id = ptcls[ip].id;
  }

  if(ptcls_mem_release) {
    ptcls.clear();
    ptcls.shrink_to_fit();
  }

  std::cout << "orig nsize=" << cloud.pts.size() << std::endl;
  std::cout << "orig nsize^(1/3)=" << (int)(pow((double)(cloud.pts.size()), 1.0 / 3.0)) << std::endl;

  int64_t npad = 0;

  //  Duplicate ptcls around global boundaries
  if(periodic) {
    const double lpad = pad_size;
    const double rpad = boxsize - pad_size;

    // uint64_t nn = nptcl + npad;
    uint64_t nn = cloud.pts.size();

    for(uint64_t ip = 0; ip < nn; ip++) {
      point_type tmp = cloud.pts[ip];

      if(tmp.x < lpad) {
        tmp.x += boxsize;
        cloud.pts.push_back(tmp);
        npad++;
      } else if(tmp.x > rpad) {
        tmp.x -= boxsize;
        cloud.pts.push_back(tmp);
        npad++;
      }
    }

    std::cout << "after-x A nsize^(1/3)=" << (int)(pow((double)(cloud.pts.size()), 1.0 / 3.0)) << std::endl;
    cloud.pts.erase(
        std::remove_if(cloud.pts.begin(), cloud.pts.end(),
                       [&range_wp](const point_type &p) { return ((p.x < range_wp[0]) || (p.x > range_wp[1])); }),
        cloud.pts.end());
    std::cout << "after-x B nsize^(1/3)=" << (int)(pow((double)(cloud.pts.size()), 1.0 / 3.0)) << std::endl;

    // nn = nptcl + npad;
    nn = cloud.pts.size();
    for(uint64_t ip = 0; ip < nn; ip++) {
      point_type tmp = cloud.pts[ip];

      if(tmp.y < lpad) {
        tmp.y += boxsize;
        cloud.pts.push_back(tmp);
        npad++;
      } else if(tmp.y > rpad) {
        tmp.y -= boxsize;
        cloud.pts.push_back(tmp);
        npad++;
      }
    }

    std::cout << "after-xy A nsize^(1/3)=" << (int)(pow((double)(cloud.pts.size()), 1.0 / 3.0)) << std::endl;
    cloud.pts.erase(
        std::remove_if(cloud.pts.begin(), cloud.pts.end(),
                       [&range_wp](const point_type &p) { return ((p.y < range_wp[2]) || (p.y > range_wp[3])); }),
        cloud.pts.end());
    std::cout << "after-xy B nsize^(1/3)=" << (int)(pow((double)(cloud.pts.size()), 1.0 / 3.0)) << std::endl;

    // nn = nptcl + npad;
    nn = cloud.pts.size();
    for(uint64_t ip = 0; ip < nn; ip++) {
      point_type tmp = cloud.pts[ip];

      if(tmp.z < lpad) {
        tmp.z += boxsize;
        cloud.pts.push_back(tmp);
        npad++;
      } else if(tmp.z > rpad) {
        tmp.z -= boxsize;
        cloud.pts.push_back(tmp);
        npad++;
      }
    }

    std::cout << "after-xyz A nsize^(1/3)=" << (int)(pow((double)(cloud.pts.size()), 1.0 / 3.0)) << std::endl;
    cloud.pts.erase(
        std::remove_if(cloud.pts.begin(), cloud.pts.end(),
                       [&range_wp](const point_type &p) { return ((p.z < range_wp[4]) || (p.z > range_wp[5])); }),
        cloud.pts.end());
    std::cout << "after-xyz B nsize^(1/3)=" << (int)(pow((double)(cloud.pts.size()), 1.0 / 3.0)) << std::endl;
  }

  nptcl = cloud.pts.size();
  cloud.npad = nptcl - cloud.norig;

  std::cout << "after-xyz nsize=" << nptcl << std::endl;
  std::cout << "after-xyz nsize^(1/3)=" << (int)(pow((double)(nptcl), 1.0 / 3.0)) << std::endl;
  std::cout << "norig=" << cloud.norig << " " << "npad=" << cloud.npad << std::endl;
}

template <typename T>
void set_ptcl_to_point_memsave_mp(PointCloud &cloud, T &ptcls, const double boxsize, const double pad_size,
                                  const double range_wp[6], const bool ptcls_mem_release = true,
                                  const bool periodic = true)
{
  using point_type = typename std::decay_t<decltype(cloud.pts)>::value_type;
  using point_vec = std::vector<point_type>;

  uint64_t nptcl(ptcls.size());
  cloud.norig = nptcl;
  cloud.pts.resize(nptcl);

#pragma omp parallel for
  for(uint64_t ip = 0; ip < nptcl; ip++) {
    cloud.pts[ip].x = ptcls[ip].x;
    cloud.pts[ip].y = ptcls[ip].y;
    cloud.pts[ip].z = ptcls[ip].z;
    cloud.pts[ip].id = ptcls[ip].id;
  }

  if(ptcls_mem_release) {
    ptcls.clear();
    ptcls.shrink_to_fit();
  }

  std::cout << "orig nsize=" << cloud.pts.size() << std::endl;
  std::cout << "orig nsize^(1/3)=" << (int)(pow((double)(cloud.pts.size()), 1.0 / 3.0)) << std::endl;

  int64_t npad = 0;

  //  Duplicate ptcls around global boundaries
  if(periodic) {
    const double lpad = pad_size;
    const double rpad = boxsize - pad_size;

    // uint64_t nn = nptcl + npad;
    uint64_t nn = cloud.pts.size();

#pragma omp parallel
    {
      point_vec th_point;
      uint64_t th_npad = 0;

#pragma omp for schedule(auto)
      for(uint64_t ip = 0; ip < nn; ip++) {
        point_type tmp = cloud.pts[ip];

        if(tmp.x < lpad) {
          tmp.x += boxsize;
          th_point.push_back(tmp);
          th_npad++;
        } else if(tmp.x > rpad) {
          tmp.x -= boxsize;
          th_point.push_back(tmp);
          th_npad++;
        }
      }

#pragma omp critical
      {
        npad += th_npad;
        cloud.pts.insert(cloud.pts.end(), th_point.begin(), th_point.end());
      }
    } // omp parallel

    std::cout << "after-x A nsize^(1/3)=" << (int)(pow((double)(cloud.pts.size()), 1.0 / 3.0)) << std::endl;
    cloud.pts.erase(
        std::remove_if(cloud.pts.begin(), cloud.pts.end(),
                       [&range_wp](const point_type &p) { return ((p.x < range_wp[0]) || (p.x > range_wp[1])); }),
        cloud.pts.end());
    std::cout << "after-x B nsize^(1/3)=" << (int)(pow((double)(cloud.pts.size()), 1.0 / 3.0)) << std::endl;

    // nn = nptcl + npad;
    nn = cloud.pts.size();

#pragma omp parallel
    {
      point_vec th_point;
      uint64_t th_npad = 0;

#pragma omp for schedule(auto)
      for(uint64_t ip = 0; ip < nn; ip++) {
        point_type tmp = cloud.pts[ip];

        if(tmp.y < lpad) {
          tmp.y += boxsize;
          th_point.push_back(tmp);
          th_npad++;
        } else if(tmp.y > rpad) {
          tmp.y -= boxsize;
          th_point.push_back(tmp);
          th_npad++;
        }
      }

#pragma omp critical
      {
        npad += th_npad;
        cloud.pts.insert(cloud.pts.end(), th_point.begin(), th_point.end());
      }
    } // omp parallel

    std::cout << "after-xy A nsize^(1/3)=" << (int)(pow((double)(cloud.pts.size()), 1.0 / 3.0)) << std::endl;
    cloud.pts.erase(
        std::remove_if(cloud.pts.begin(), cloud.pts.end(),
                       [&range_wp](const point_type &p) { return ((p.y < range_wp[2]) || (p.y > range_wp[3])); }),
        cloud.pts.end());
    std::cout << "after-xy B nsize^(1/3)=" << (int)(pow((double)(cloud.pts.size()), 1.0 / 3.0)) << std::endl;

    // nn = nptcl + npad;
    nn = cloud.pts.size();

#pragma omp parallel
    {
      point_vec th_point;
      uint64_t th_npad = 0;

#pragma omp for schedule(auto)
      for(uint64_t ip = 0; ip < nn; ip++) {
        point_type tmp = cloud.pts[ip];

        if(tmp.z < lpad) {
          tmp.z += boxsize;
          th_point.push_back(tmp);
          th_npad++;
        } else if(tmp.z > rpad) {
          tmp.z -= boxsize;
          th_point.push_back(tmp);
          th_npad++;
        }
      }

#pragma omp critical
      {
        npad += th_npad;
        cloud.pts.insert(cloud.pts.end(), th_point.begin(), th_point.end());
      }
    } // omp parallel

    std::cout << "after-xyz A nsize^(1/3)=" << (int)(pow((double)(cloud.pts.size()), 1.0 / 3.0)) << std::endl;
    cloud.pts.erase(
        std::remove_if(cloud.pts.begin(), cloud.pts.end(),
                       [&range_wp](const point_type &p) { return ((p.z < range_wp[4]) || (p.z > range_wp[5])); }),
        cloud.pts.end());
    std::cout << "after-xyz B nsize^(1/3)=" << (int)(pow((double)(cloud.pts.size()), 1.0 / 3.0)) << std::endl;
  } // periodic

  nptcl = cloud.pts.size();
  cloud.npad = nptcl - cloud.norig;

  std::cout << "after-xyz nsize=" << nptcl << std::endl;
  std::cout << "after-xyz nsize^(1/3)=" << (int)(pow((double)(nptcl), 1.0 / 3.0)) << std::endl;
  std::cout << "norig=" << cloud.norig << " " << "npad=" << cloud.npad << std::endl;
}

template <typename T>
void set_ptcl_to_point_new(PointCloud &cloud, T &ptcls, const double boxsize, const double pad_size,
                           const double range_wp[6], const bool ptcls_mem_release = true, const bool periodic = true)
{
  using point_type = typename std::decay_t<decltype(cloud.pts)>::value_type;

  uint64_t nptcl = ptcls.size();
  cloud.norig = nptcl;
  cloud.pts.clear();
  cloud.pts.reserve(nptcl * 1.05);

  std::vector<double> shiftsX, shiftsY, shiftsZ;
  shiftsX.reserve(3);
  shiftsY.reserve(3);
  shiftsZ.reserve(3);

  for(uint64_t ip = 0; ip < nptcl; ++ip) {
    point_type orig;
    orig.x = ptcls[ip].x;
    orig.y = ptcls[ip].y;
    orig.z = ptcls[ip].z;
    orig.id = ptcls[ip].id;

    if(((range_wp[0] <= orig.x) && (orig.x <= range_wp[1])) && ((range_wp[2] <= orig.y) && (orig.y <= range_wp[3])) &&
       ((range_wp[4] <= orig.z) && (orig.z <= range_wp[5]))) {
      cloud.pts.push_back(orig);
    }

    if(periodic) {
      bool bound = (orig.x < pad_size) || (orig.x > boxsize - pad_size);
      bound |= (orig.y < pad_size) || (orig.y > boxsize - pad_size);
      bound |= (orig.z < pad_size) || (orig.z > boxsize - pad_size);

      if(bound) {
        shiftsX.clear();
        shiftsY.clear();
        shiftsZ.clear();
        shiftsX.push_back(0.0);
        shiftsY.push_back(0.0);
        shiftsZ.push_back(0.0);

        if(orig.x < pad_size) shiftsX.push_back(+boxsize);
        if(orig.x > boxsize - pad_size) shiftsX.push_back(-boxsize);
        if(orig.y < pad_size) shiftsY.push_back(+boxsize);
        if(orig.y > boxsize - pad_size) shiftsY.push_back(-boxsize);
        if(orig.z < pad_size) shiftsZ.push_back(+boxsize);
        if(orig.z > boxsize - pad_size) shiftsZ.push_back(-boxsize);

        for(auto dx : shiftsX) {
          for(auto dy : shiftsY) {
            for(auto dz : shiftsZ) {
              if(dx == 0.0 && dy == 0.0 && dz == 0.0) continue;
              point_type ghost;
              ghost.x = orig.x + dx;
              ghost.y = orig.y + dy;
              ghost.z = orig.z + dz;
              ghost.id = orig.id;
              cloud.pts.push_back(ghost);
            }
          }
        }
      } // bound
    } // periodic
  } // ipart

  if(ptcls_mem_release) {
    ptcls.clear();
    ptcls.shrink_to_fit();
  }

#if 0
  cloud.pts.erase(std::remove_if(cloud.pts.begin(), cloud.pts.end(),
                                 [&range_wp](const point_type &p) {
                                   return ((p.x < range_wp[0]) || (p.x > range_wp[1]) || (p.y < range_wp[2]) ||
                                           (p.y > range_wp[3]) || (p.z < range_wp[4]) || (p.z > range_wp[5]));
                                 }),
                  cloud.pts.end());
#endif

  uint64_t nnew = cloud.pts.size();
  cloud.npad = nnew - cloud.norig;

  std::cout << "after-xyz nsize=" << nnew << std::endl;
  std::cout << "norig=" << cloud.norig << " npad=" << cloud.npad << std::endl;
}

template <typename T>
void set_ptcl_to_point_new_mp(PointCloud &cloud, T &ptcls, const double boxsize, const double pad_size,
                              const double range_wp[6], const bool ptcls_mem_release = true, const bool periodic = true)
{
  using point_type = typename std::decay_t<decltype(cloud.pts)>::value_type;
  using point_vec = std::vector<point_type>;

  uint64_t nptcl = ptcls.size();
  cloud.norig = nptcl;
  cloud.pts.clear();

#pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    point_vec th_point;
    th_point.reserve(1.05 * nptcl / nthreads);

    std::vector<double> shiftsX, shiftsY, shiftsZ;
    shiftsX.reserve(3);
    shiftsY.reserve(3);
    shiftsZ.reserve(3);

#pragma omp for schedule(auto)
    for(uint64_t ip = 0; ip < nptcl; ++ip) {
      point_type orig;
      orig.x = ptcls[ip].x;
      orig.y = ptcls[ip].y;
      orig.z = ptcls[ip].z;
      orig.id = ptcls[ip].id;

      if(((range_wp[0] <= orig.x) && (orig.x <= range_wp[1])) && ((range_wp[2] <= orig.y) && (orig.y <= range_wp[3])) &&
         ((range_wp[4] <= orig.z) && (orig.z <= range_wp[5]))) {
        th_point.push_back(orig);
      }

      if(periodic) {
        bool bound = (orig.x < pad_size) || (orig.x > boxsize - pad_size);
        bound |= (orig.y < pad_size) || (orig.y > boxsize - pad_size);
        bound |= (orig.z < pad_size) || (orig.z > boxsize - pad_size);

        if(bound) {
          shiftsX.clear();
          shiftsY.clear();
          shiftsZ.clear();
          shiftsX.push_back(0.0);
          shiftsY.push_back(0.0);
          shiftsZ.push_back(0.0);

          if(orig.x < pad_size) shiftsX.push_back(+boxsize);
          if(orig.x > boxsize - pad_size) shiftsX.push_back(-boxsize);
          if(orig.y < pad_size) shiftsY.push_back(+boxsize);
          if(orig.y > boxsize - pad_size) shiftsY.push_back(-boxsize);
          if(orig.z < pad_size) shiftsZ.push_back(+boxsize);
          if(orig.z > boxsize - pad_size) shiftsZ.push_back(-boxsize);

          for(auto dx : shiftsX) {
            for(auto dy : shiftsY) {
              for(auto dz : shiftsZ) {
                if(dx == 0.0 && dy == 0.0 && dz == 0.0) continue;
                point_type ghost;
                ghost.x = orig.x + dx;
                ghost.y = orig.y + dy;
                ghost.z = orig.z + dz;
                ghost.id = orig.id;
                th_point.push_back(ghost);
              }
            }
          }
        } // bound
      } // periodic
    } // ipart

#pragma omp critical
    {
      cloud.pts.insert(cloud.pts.end(), th_point.begin(), th_point.end());
    }
  } // parallel

  if(ptcls_mem_release) {
    ptcls.clear();
    ptcls.shrink_to_fit();
  }

#if 0
  cloud.pts.erase(std::remove_if(cloud.pts.begin(), cloud.pts.end(),
                                 [&range_wp](const point_type &p) {
                                   return ((p.x < range_wp[0]) || (p.x > range_wp[1]) || (p.y < range_wp[2]) ||
                                           (p.y > range_wp[3]) || (p.z < range_wp[4]) || (p.z > range_wp[5]));
                                 }),
                  cloud.pts.end());
#endif

  uint64_t nnew = cloud.pts.size();
  cloud.npad = nnew - cloud.norig;

  std::cout << "after-xyz nsize=" << nnew << std::endl;
  std::cout << "norig=" << cloud.norig << " npad=" << cloud.npad << std::endl;
}

void triming_ptcl_to_point(PointCloud &cloud, const double range_wp[6])
{
  using point_type = typename std::decay_t<decltype(cloud.pts)>::value_type;

  uint64_t nptcl(cloud.pts.size());

  std::cout << "befor triming nsize=" << nptcl << std::endl;
  std::cout << "before nsize^(1/3)=" << (int)(pow((double)(nptcl), 1.0 / 3.0)) << std::endl;

  cloud.pts.erase(std::remove_if(cloud.pts.begin(), cloud.pts.end(),
                                 [&range_wp](const point_type &p) {
                                   return ((p.x < range_wp[0]) || (p.x > range_wp[1]) || (p.y < range_wp[2]) ||
                                           (p.y > range_wp[3]) || (p.z < range_wp[4]) || (p.z > range_wp[5]));
                                 }),
                  cloud.pts.end());

  nptcl = cloud.pts.size();
  std::cout << "after triming nsize=" << nptcl << std::endl;
  std::cout << "after nsize^(1/3)=" << (int)(pow((double)(nptcl), 1.0 / 3.0)) << std::endl;
}

int main(int argc, char **argv)
{
  constexpr int ndim = 3;
  constexpr int max_leaf = 50;
  constexpr int build_threads = 0;

  // int64_t npart = 1024 * 1024 * 64;
  int64_t npart = 1024 * 1024 * 4;
  std::vector<particle> ptcls(npart);

  double boxsize = 1.0;
  double pad_size = 0.05;

  int seed = 100;
  std::mt19937 mt(seed);
  std::uniform_real_distribution<float> dist(0.0, boxsize);

#if 0
  for(size_t i = 0; i < ptcls.size(); i++) {
    ptcls[i].x = dist(mt);
    ptcls[i].y = dist(mt);
    ptcls[i].z = dist(mt);
    ptcls[i].id = static_cast<int>(i);
  }
#elif 0
  std::uniform_real_distribution<float> dist_x_first(0.0, 0.1);
  std::uniform_real_distribution<float> dist_x_second(0.9, 1.0);
  ptcls.resize(npart + npart);

  for(size_t i = 0; i < npart; ++i) {
    ptcls[i].x = dist_x_first(mt);
    ptcls[i].y = dist(mt);
    ptcls[i].z = dist(mt);
    ptcls[i].id = static_cast<int>(i);
  }

  for(size_t i = 0; i < npart; ++i) {
    ptcls[npart + i].x = dist_x_second(mt);
    ptcls[npart + i].y = dist(mt);
    ptcls[npart + i].z = dist(mt);
    ptcls[npart + i].id = static_cast<int>(npart + i);
  }
#else
  std::uniform_real_distribution<float> dist_x_first(0.0, 0.1);
  std::uniform_real_distribution<float> dist_x_second(0.9, 1.0);
  ptcls.resize(npart + npart);

  for(size_t i = 0; i < npart; ++i) {
    ptcls[i].x = dist_x_first(mt);
    ptcls[i].y = dist_x_first(mt);
    ptcls[i].z = dist_x_first(mt);
    ptcls[i].id = static_cast<int>(i);
  }

  for(size_t i = 0; i < npart; ++i) {
    ptcls[npart + i].x = dist_x_second(mt);
    ptcls[npart + i].y = dist_x_second(mt);
    ptcls[npart + i].z = dist_x_second(mt);
    ptcls[npart + i].id = static_cast<int>(npart + i);
  }
#endif

  std::cout << "before nsize=" << ptcls.size() << " nsize^(1/3)=" << (int)(pow((double)(ptcls.size()), 1.0 / 3.0))
            << std::endl;

  PointCloud cloud;

  const double range[6] = {0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
  const double range_wp[6] = {range[0] - pad_size, range[1] + pad_size, range[2] - pad_size,
                              range[3] + pad_size, range[4] - pad_size, range[5] + pad_size};

  auto t_start = std::chrono::high_resolution_clock::now();

#if 0
  // set_ptcl_to_point(cloud, ptcls, boxsize, pad_size, range_wp, true);
  set_ptcl_to_point_mp(cloud, ptcls, boxsize, pad_size, range_wp, true);
  //  triming_ptcl_to_point(cloud, range_wp);
#elif 0
  // set_ptcl_to_point_memsave(cloud, ptcls, boxsize, pad_size, range_wp, true);
  set_ptcl_to_point_memsave_mp(cloud, ptcls, boxsize, pad_size, range_wp, true);
#else
  // set_ptcl_to_point_new(cloud, ptcls, boxsize, pad_size, range_wp, true);
  set_ptcl_to_point_new_mp(cloud, ptcls, boxsize, pad_size, range_wp, true);
#endif

  auto t_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> t_elapsed = t_end - t_start;
  std::cout << "set periodic point colud time: " << t_elapsed.count() << " [s]\n";

  std::cout << "after nsize=" << cloud.pts.size()
            << " nsize^(1/3)=" << (int)(pow((double)(cloud.pts.size()), 1.0 / 3.0)) << std::endl;

  // construct a kd-tree index:
  using my_kd_tree_t =
      nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud>, PointCloud, ndim>;

  t_start = std::chrono::high_resolution_clock::now();

  nanoflann::KDTreeSingleIndexAdaptorParams tree_params;
  tree_params.leaf_max_size = max_leaf;
  tree_params.n_thread_build = build_threads;

  // build tree index
  my_kd_tree_t index(ndim, cloud, tree_params);

  t_end = std::chrono::high_resolution_clock::now();
  t_elapsed = t_end - t_start;
  std::cout << "KD-Tree built with " << ptcls.size() << " particles. Elapsed time: " << t_elapsed.count() << " [s]\n";

  std::vector<nanoflann::ResultItem<uint32_t, float>> matches;
  nanoflann::SearchParameters search_params;
  //  search_params.sorted = false;

  // float query_pt[3] = {0.5f, 0.5f, 0.5f};
  // float query_pt[3] = {0.03, 0.02, 0.01};
  float query_pt[3] = {0.99, 0.99, 0.98};

  // radiusSearch
  // Ensure that the input is the squared distance, not the actual distance.
  float radius = 0.05f;

  t_start = std::chrono::high_resolution_clock::now();
  const uint32_t num_found = index.radiusSearch(query_pt, radius * radius, matches, search_params);
  std::cout << "Found " << num_found << " particles within radius " << radius << ":\n";

  t_end = std::chrono::high_resolution_clock::now();
  t_elapsed = t_end - t_start;
  std::cout << "Elapsed time: " << t_elapsed.count() << " [s]\n";

#if 1
  for(const auto &m : matches) {
    std::cout << "Index: " << cloud.pts[m.first].id << ", Distance^2: " << m.second << " pos: " << cloud.pts[m.first].x
              << ", " << cloud.pts[m.first].y << ", " << cloud.pts[m.first].z << '\n';
  }
#endif

  uint64_t check_idx1 = 0;
  for(const auto &m : matches) {
    check_idx1 += cloud.pts[m.first].id;
  }

  // knnSearch
  const size_t k = 20;
  uint32_t out_indices[k];
  float out_distances_sq[k];

  t_start = std::chrono::high_resolution_clock::now();
  const size_t nMatches = index.knnSearch(query_pt, k, out_indices, out_distances_sq);
  std::cout << "Found " << nMatches << " nearest neighbors:\n";

  t_end = std::chrono::high_resolution_clock::now();
  t_elapsed = t_end - t_start;
  std::cout << "Elapsed time: " << t_elapsed.count() << " [s]\n";

  for(size_t i = 0; i < nMatches; i++) {
    std::cout << "Index: " << cloud.pts[out_indices[i]].id << ", Distance^2: " << out_distances_sq[i]
              << " pos: " << cloud.pts[out_indices[i]].x << ", " << cloud.pts[out_indices[i]].y << ", "
              << cloud.pts[out_indices[i]].z << "\n";
  }

  uint64_t check_idx2 = 0;
  for(size_t i = 0; i < nMatches; i++) {
    check_idx2 += cloud.pts[out_indices[i]].id;
  }

  std::cout << "\n";
  std::cout << "check index sum: " << check_idx1 << " " << check_idx2 << "\n";

  return 0;
}
