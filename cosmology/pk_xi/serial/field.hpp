#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cassert>

#include "util.hpp"

static inline int assign_axis(const double pos, const int nmesh, const int scheme, int idx[3], double w[3])
{
  int nassign = 0;
  double x = pos * nmesh;
  if(scheme == 1) {
    // NGP
    int i = static_cast<int>(x + 0.5);
    i = (i + nmesh) % nmesh;
    idx[0] = i;
    w[0] = 1.0;
    nassign = 1;
  } else if(scheme == 2) {
    // CIC
    int i0 = static_cast<int>(std::floor(x));
    double f = x - i0;
    int i1 = i0 + 1;
    i0 = (i0 + nmesh) % nmesh;
    i1 = (i1 + nmesh) % nmesh;
    idx[0] = i0;
    idx[1] = i1;
    w[0] = 1.0 - f;
    w[1] = f;
    nassign = 2;
  } else if(scheme == 3) {
    // TSC
    double xt = x - 0.5;
    int i1 = static_cast<int>(xt + 0.5);
    double dx = xt - i1;
    double w_left = 0.5 * (0.5 - dx) * (0.5 - dx);
    double w_center = 0.75 - dx * dx;
    double w_right = 0.5 * (0.5 + dx) * (0.5 + dx);
    int i_left = i1 - 1;
    int i_right = i1 + 1;
    i_left = (i_left + nmesh) % nmesh;
    i1 = (i1 + nmesh) % nmesh;
    i_right = (i_right + nmesh) % nmesh;
    idx[0] = i_left;
    idx[1] = i1;
    idx[2] = i_right;
    w[0] = w_left;
    w[1] = w_center;
    w[2] = w_right;
    nassign = 3;
  }

  return nassign;
}

/*
  Function that assigns all particles to the mesh
  with openmp
*/
template <typename T, typename U>
void ptcl_assign_mesh_omp(T &ptcls, U &mesh, const int nmesh, const double lbox, const double ptcl_mass,
                          const int valtype, const int scheme)
{
  // scheme: 1: NGP, 2: CIC, 3: TSC
  int64_t nmesh_z = nmesh + 2;
  int64_t nmesh_tot = mesh.size();
  int64_t npart = ptcls.size();

  constexpr double mem_size = 100.0; // GB

  size_t element_size = sizeof(typename U::value_type);
  double mesh_gb = nmesh_tot * element_size * 1e-9;
  int nthread = std::max(1, (int)(mem_size / mesh_gb));
  nthread = std::min(nthread, omp_get_max_threads());

  std::cout << "# mesh size " << nmesh_tot << " , " << mesh_gb << " GB :: preset node memory size " << mem_size << " GB"
            << std::endl;
  std::cout << "# calc mesh assign for " << nthread << " threads" << std::endl;

#pragma omp parallel for reduction(vec_float_plus : mesh) num_threads(nthread)
  for(uint64_t p = 0; p < npart; p++) {
    /* rescale the coordinates of the particles */
    auto xpos = ptcls[p].pos[0] / lbox;
    auto ypos = ptcls[p].pos[1] / lbox;
    auto zpos = ptcls[p].pos[2] / lbox;

    double _val;
    if(valtype == 0) _val = ptcl_mass;
    else if(valtype == 1) _val = ptcls[p].pot;

    assert(xpos >= 0.0 && xpos <= 1.0);
    assert(ypos >= 0.0 && ypos <= 1.0);
    assert(zpos >= 0.0 && zpos <= 1.0);

    int idx_x[3], idx_y[3], idx_z[3];
    double w_x[3], w_y[3], w_z[3];
    int nx = assign_axis(xpos, nmesh, scheme, idx_x, w_x);
    int ny = assign_axis(ypos, nmesh, scheme, idx_y, w_y);
    int nz = assign_axis(zpos, nmesh, scheme, idx_z, w_z);

    for(int ix = 0; ix < nx; ix++) {
      for(int iy = 0; iy < ny; iy++) {
        for(int iz = 0; iz < nz; iz++) {
          int64_t iwx = idx_x[ix];
          int64_t iwy = idx_y[iy];
          int64_t iwz = idx_z[iz];
          double weight = w_x[ix] * w_y[iy] * w_z[iz];
          int64_t idx = iwz + nmesh_z * (iwy + nmesh * iwx);
          mesh[idx] += weight * _val;
        }
      }
    }
  } // p loop

  std::cout << "# done mesh assign (scheme=" << scheme << ")" << std::endl;
}

/*
  Function that reads file input and assigns particles to the mesh
*/
template <typename T, typename U>
void ptcl_assign_mesh(T &ptcls, U &mesh, const int nmesh, const double lbox, const double ptcl_mass, const int valtype,
                      const int scheme)
{
  int64_t nmesh_z = nmesh + 2;
  int64_t npart = ptcls.size();

  for(uint64_t p = 0; p < npart; p++) {
    /* rescale the coordinates of the particles */
    auto xpos = ptcls[p].pos[0] / lbox;
    auto ypos = ptcls[p].pos[1] / lbox;
    auto zpos = ptcls[p].pos[2] / lbox;

    double _val;
    if(valtype == 0) _val = ptcl_mass;
    else if(valtype == 1) _val = ptcls[p].pot;

    assert(xpos >= 0.0 && xpos <= 1.0);
    assert(ypos >= 0.0 && ypos <= 1.0);
    assert(zpos >= 0.0 && zpos <= 1.0);

    int idx_x[3], idx_y[3], idx_z[3];
    double w_x[3], w_y[3], w_z[3];
    int nx = assign_axis(xpos, nmesh, scheme, idx_x, w_x);
    int ny = assign_axis(ypos, nmesh, scheme, idx_y, w_y);
    int nz = assign_axis(zpos, nmesh, scheme, idx_z, w_z);

    for(int ix = 0; ix < nx; ix++) {
      for(int iy = 0; iy < ny; iy++) {
        for(int iz = 0; iz < nz; iz++) {
          int64_t iwx = idx_x[ix];
          int64_t iwy = idx_y[iy];
          int64_t iwz = idx_z[iz];
          double weight = w_x[ix] * w_y[iy] * w_z[iz];
          int64_t idx = iwz + nmesh_z * (iwy + nmesh * iwx);
          mesh[idx] += weight * _val;
        }
      }
    }
  } // p loop
}

template <typename T>
void halo_assign_mesh(T &pos, T &val, T &mesh, const int nmesh, const double lbox, const int scheme)
{
  int64_t nmesh_z = nmesh + 2;
  int64_t npart = pos.size() / 3;

  for(uint64_t p = 0; p < npart; p++) {
    /* rescale the coordinates of the particles */
    auto xpos = pos[3 * p + 0] / lbox;
    auto ypos = pos[3 * p + 1] / lbox;
    auto zpos = pos[3 * p + 2] / lbox;
    auto _val = val[p];

    assert(xpos >= 0.0 && xpos <= 1.0);
    assert(ypos >= 0.0 && ypos <= 1.0);
    assert(zpos >= 0.0 && zpos <= 1.0);

    int idx_x[3], idx_y[3], idx_z[3];
    double w_x[3], w_y[3], w_z[3];
    int nx = assign_axis(xpos, nmesh, scheme, idx_x, w_x);
    int ny = assign_axis(ypos, nmesh, scheme, idx_y, w_y);
    int nz = assign_axis(zpos, nmesh, scheme, idx_z, w_z);

    for(int ix = 0; ix < nx; ix++) {
      for(int iy = 0; iy < ny; iy++) {
        for(int iz = 0; iz < nz; iz++) {
          int64_t iwx = idx_x[ix];
          int64_t iwy = idx_y[iy];
          int64_t iwz = idx_z[iz];
          double weight = w_x[ix] * w_y[iy] * w_z[iz];
          int64_t idx = iwz + nmesh_z * (iwy + nmesh * iwx);
          mesh[idx] += weight * _val;
        }
      }
    }
  }
}

template <typename G, typename T>
void group_assign_mesh(G &grp, T &mesh, const int nmesh, const int scheme)
{
  int64_t nmesh_z = nmesh + 2;
  int64_t npart = grp.size();

  for(uint64_t p = 0; p < npart; p++) {
    auto xpos = grp[p].xpos;
    auto ypos = grp[p].ypos;
    auto zpos = grp[p].zpos;
    // auto _val = grp[p].mass;
    auto _val = 1;

    assert(xpos >= 0.0 && xpos <= 1.0);
    assert(ypos >= 0.0 && ypos <= 1.0);
    assert(zpos >= 0.0 && zpos <= 1.0);

    int idx_x[3], idx_y[3], idx_z[3];
    double w_x[3], w_y[3], w_z[3];
    int nx = assign_axis(xpos, nmesh, scheme, idx_x, w_x);
    int ny = assign_axis(ypos, nmesh, scheme, idx_y, w_y);
    int nz = assign_axis(zpos, nmesh, scheme, idx_z, w_z);

    for(int ix = 0; ix < nx; ix++) {
      for(int iy = 0; iy < ny; iy++) {
        for(int iz = 0; iz < nz; iz++) {
          int64_t iwx = idx_x[ix];
          int64_t iwy = idx_y[iy];
          int64_t iwz = idx_z[iz];
          double weight = w_x[ix] * w_y[iy] * w_z[iz];
          int64_t idx = iwz + nmesh_z * (iwy + nmesh * iwx);
          mesh[idx] += weight * _val;
        }
      }
    }
  }
}

template <typename T>
void output_field(T &mesh, const int nmesh, const double lbox, std::string filename)
{
  const int64_t nmesh_tot = (int64_t)nmesh * (int64_t)nmesh * (int64_t)nmesh;
  int64_t nmesh_z = nmesh + 2;

  size_t Tsize = sizeof(typename T::value_type);
  std::ofstream fout(filename);
  fout.write((char *)&nmesh, sizeof(int));
  fout.write((char *)&nmesh_tot, sizeof(int64_t));
  fout.write((char *)&lbox, sizeof(double));
  for(int ix = 0; ix < nmesh; ix++) {
    for(int iy = 0; iy < nmesh; iy++) {
      // int iz = 0;
      const int64_t istart = nmesh_z * (iy + (nmesh * ix));
      fout.write((char *)&mesh[istart], Tsize * nmesh);
    }
  }
  fout.close();
}

template <typename T>
void output_fft_field(T &mesh, const int nmesh, const double lbox, std::string filename)
{
  int64_t nmesh_tot = (int64_t)nmesh * (int64_t)nmesh * (int64_t)(nmesh / 2 + 1);
  nmesh_tot *= 2; // for complex

  size_t Tsize = sizeof(typename T::value_type);
  std::ofstream fout(filename);
  fout.write((char *)&nmesh, sizeof(int));
  fout.write((char *)&nmesh_tot, sizeof(int64_t));
  fout.write((char *)&lbox, sizeof(double));
  fout.write((char *)mesh.data(), Tsize * nmesh_tot);
  fout.close();
}
