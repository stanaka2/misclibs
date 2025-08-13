#pragma once

#include <vector>
#include <cmath>

#include "run_param.hpp"

template <typename T>
class moments
{
public:
  using vec1d = std::vector<T>;
  using vec2d = std::vector<std::vector<T>>;
  using vecpt = std::vector<particle_str>;

  int scheme = 2; // 1: NGP, 2: CIC, 3: TSC, 4: PCS
  bool p2_flag = false;
  double lbox = 1.0;

  int64_t nmesh_x, nmesh_y, nmesh_z;

  bool use_deltaf = false;
  double Mnu_bg_box = 0.0;

  moments(int64_t nmesh, double lbox, const std::string &_scheme_type = "TSC", const bool _deltaf = false)
      : nmesh_x(nmesh), nmesh_y(nmesh), nmesh_z(nmesh), lbox(lbox), use_deltaf(_deltaf)
  {
    if(_scheme_type == "NGP") scheme = 1;
    else if(_scheme_type == "CIC") scheme = 2;
    else if(_scheme_type == "TSC") scheme = 3;
    else if(_scheme_type == "PCS") scheme = 4;
  }

  inline int64_t nmesh_total(int64_t nx, int64_t ny, int64_t nz) { return nx * ny * nz; }

  inline int64_t mesh_index(int ix, int iy, int iz, int64_t nx, int64_t ny, int64_t nz)
  {
    return int64_t(iz) + nz * (int64_t(iy) + ny * int64_t(ix));
  }

  inline int assign_axis(const double pos, const int64_t nmesh, int idx[4], double w[4])
  {
    double x = pos * nmesh;
    if(x >= nmesh) x -= nmesh;
    if(x < 0.0) x += nmesh;

    int n;

    if(scheme == 1) {
      // NGP
      int i = (int)std::floor(x + 0.5);
      i = (i % nmesh + nmesh) % nmesh;

      n = 0;
      idx[n] = i;
      w[n] = 1.0;
      n++;

    } else if(scheme == 2) {
      // CIC
      int i0 = (int)std::floor(x);
      double f = x - i0; // [0,1)
      int i1 = i0 + 1;
      i0 = (i0 % nmesh + nmesh) % nmesh;
      i1 = (i1 % nmesh + nmesh) % nmesh;

      n = 0;
      idx[n] = i0;
      w[n] = 1.0 - f;
      n++;
      idx[n] = i1;
      w[n] = f;
      n++;

    } else if(scheme == 3) {
      // TSC
      double xt = x - 0.5;
      int ic = (int)std::floor(xt + 0.5);
      double dx = xt - ic;
      double wl = 0.5 * (0.5 - dx) * (0.5 - dx);
      double wc = 0.75 - dx * dx;
      double wr = 0.5 * (0.5 + dx) * (0.5 + dx);
      int il = ic - 1, ir = ic + 1;
      il = (il % nmesh + nmesh) % nmesh;
      ic = (ic % nmesh + nmesh) % nmesh;
      ir = (ir % nmesh + nmesh) % nmesh;

      n = 0;
      idx[n] = il;
      w[n] = wl;
      n++;
      idx[n] = ic;
      w[n] = wc;
      n++;
      idx[n] = ir;
      w[n] = wr;
      n++;

    } else if(scheme == 4) {
      // PCS
      int i0 = (int)std::floor(x) - 1;
      n = 0;
      for(int k = 0; k < 4; ++k) {
        int j = i0 + k;
        double r = std::fabs(x - (double)j);
        double r2 = r * r;
        double r3 = r2 * r;
        double wk = 0.0;
        if(r < 1.0) {
          wk = (4.0 - 6.0 * r2 + 3.0 * r3) / 6.0;
        } else if(r < 2.0) {
          double t = (2.0 - r);
          wk = (t * t * t) / 6.0;
        } else {
          wk = 0.0;
        }
        if(wk > 0.0) {
          idx[n] = (j % nmesh + nmesh) % nmesh;
          w[n] = wk;
          ++n;
        }
      }
    }
    return n;
  }

  // return rho
  vec1d calc_dens(const vecpt &ptcl)
  {
    const int64_t nzlen = p2_flag ? (nmesh_z + 2) : nmesh_z;
    const int64_t ntot = nmesh_total(nmesh_x, nmesh_y, nzlen);
    const int64_t nptcl = static_cast<int64_t>(ptcl.size());

    vec1d dens(ntot, T(0.0));

#pragma omp parallel for schedule(auto)
    for(int64_t p = 0; p < nptcl; p++) {
      double x = ptcl[p].xpos / lbox;
      double y = ptcl[p].ypos / lbox;
      double z = ptcl[p].zpos / lbox;
      double m = ptcl[p].mass;

      int ix[4], iy[4], iz[4];
      double wx[4], wy[4], wz[4];

      int nwx = assign_axis(x, nmesh_x, ix, wx);
      int nwy = assign_axis(y, nmesh_y, iy, wy);
      int nwz = assign_axis(z, nmesh_z, iz, wz);

#pragma unroll
      for(int a = 0; a < nwx; ++a) {
        for(int b = 0; b < nwy; ++b) {
          for(int c = 0; c < nwz; ++c) {
            const auto w = wx[a] * wy[b] * wz[c];
            int64_t idx = mesh_index(ix[a], iy[b], iz[c], nmesh_x, nmesh_y, nzlen);
#pragma omp atomic
            dens[idx] += m * w;
          }
        }
      } // a,b,c
    } // particle

    return dens;
  }

  // return rho*vel
  vec2d calc_velc(const vecpt &ptcl)
  {
    const int64_t nzlen = p2_flag ? (nmesh_z + 2) : nmesh_z;
    const int64_t ntot = nmesh_total(nmesh_x, nmesh_y, nzlen);
    const int64_t nptcl = static_cast<int64_t>(ptcl.size());

    vec2d velc(3);
    velc[0].assign(ntot, T(0));
    velc[1].assign(ntot, T(0));
    velc[2].assign(ntot, T(0));

#pragma omp parallel for schedule(auto)
    for(int64_t p = 0; p < nptcl; p++) {
      const double x = ptcl[p].xpos / lbox;
      const double y = ptcl[p].ypos / lbox;
      const double z = ptcl[p].zpos / lbox;

      const double vx = ptcl[p].xvel;
      const double vy = ptcl[p].yvel;
      const double vz = ptcl[p].zvel;

      const double m = ptcl[p].mass;

      int ix[4], iy[4], iz[4];
      double wx[4], wy[4], wz[4];
      const int nwx = assign_axis(x, nmesh_x, ix, wx);
      const int nwy = assign_axis(y, nmesh_y, iy, wy);
      const int nwz = assign_axis(z, nmesh_z, iz, wz);

      for(int a = 0; a < nwx; ++a) {
        for(int b = 0; b < nwy; ++b) {
          for(int c = 0; c < nwz; ++c) {
            const auto w = wx[a] * wy[b] * wz[c];
            const int64_t idx = mesh_index(ix[a], iy[b], iz[c], nmesh_x, nmesh_y, nzlen);
            const T mw = T(m * w);
#pragma omp atomic
            velc[0][idx] += mw * T(vx);
#pragma omp atomic
            velc[1][idx] += mw * T(vy);
#pragma omp atomic
            velc[2][idx] += mw * T(vz);
          }
        }
      } // a,b,c
    } // particle

    return velc;
  }

  // return rho * sigma
  vec2d calc_sigma(const vecpt &ptcl, const vec2d &mean_velc)
  {
    const int64_t nzlen = p2_flag ? (nmesh_z + 2) : nmesh_z;
    const int64_t ntot = nmesh_total(nmesh_x, nmesh_y, nzlen);
    const int64_t nptcl = static_cast<int64_t>(ptcl.size());

    vec2d sig(6);
    for(int k = 0; k < 6; k++) sig[k].assign(ntot, T(0));

#pragma omp parallel for schedule(auto)
    for(int64_t p = 0; p < nptcl; p++) {
      const double x = ptcl[p].xpos / lbox;
      const double y = ptcl[p].ypos / lbox;
      const double z = ptcl[p].zpos / lbox;

      const double vx = ptcl[p].xvel;
      const double vy = ptcl[p].yvel;
      const double vz = ptcl[p].zvel;

      const double m = ptcl[p].mass;

      int ix[4], iy[4], iz[4];
      double wx[4], wy[4], wz[4];
      const int nwx = assign_axis(x, nmesh_x, ix, wx);
      const int nwy = assign_axis(y, nmesh_y, iy, wy);
      const int nwz = assign_axis(z, nmesh_z, iz, wz);

      for(int a = 0; a < nwx; ++a) {
        for(int b = 0; b < nwy; ++b) {
          for(int c = 0; c < nwz; ++c) {
            const auto w = wx[a] * wy[b] * wz[c];
            const int64_t idx = mesh_index(ix[a], iy[b], iz[c], nmesh_x, nmesh_y, nzlen);

            const auto ux = mean_velc[0][idx];
            const auto uy = mean_velc[1][idx];
            const auto uz = mean_velc[2][idx];

            const auto dvx = vx - ux;
            const auto dvy = vy - uy;
            const auto dvz = vz - uz;

            const T mw = T(m * w);

#pragma omp atomic
            sig[0][idx] += mw * T(dvx * dvx); // xx
#pragma omp atomic
            sig[1][idx] += mw * T(dvx * dvy); // xy
#pragma omp atomic
            sig[2][idx] += mw * T(dvx * dvz); // xz
#pragma omp atomic
            sig[3][idx] += mw * T(dvy * dvy); // yy
#pragma omp atomic
            sig[4][idx] += mw * T(dvy * dvz); // yz
#pragma omp atomic
            sig[5][idx] += mw * T(dvz * dvz); // zz
          }
        }
      } // a,b,c
    } // particle

    return sig;
  }

  vec1d calc_dens_field(const vecpt &ptcl)
  {
    std::cout << "Calculating density field ..." << std::flush;
    auto dens = calc_dens(ptcl);

    if(use_deltaf) {
      // Mnu_bg_box = rho_nu_bar * Vcell
      assert(Mnu_bg_box > 1.0);
      auto factor = Mnu_bg_box / ((double)nmesh_x * nmesh_y * nmesh_z);
      factor = 1.0 / factor;

      const int64_t ntot = static_cast<int64_t>(dens.size());
      for(int64_t i = 0; i < ntot; i++) {
        dens[i] = 1.0 + dens[i] * factor;
      }
    }

    std::cout << " done." << std::endl;
    return dens;
  }

  vec2d calc_velc_field(const vecpt &ptcl, const double vunit = 1.0)
  {
    std::cout << "Calculating velocity field ..." << std::flush;
    const auto rho = calc_dens_field(ptcl);
    auto velc = calc_velc(ptcl);
    const int64_t ntot = static_cast<int64_t>(rho.size());

#pragma omp parallel for schedule(auto)
    for(int64_t i = 0; i < ntot; i++) {
      if(rho[i] > 0.0) {
        const auto f = vunit / rho[i];
        for(int iv = 0; iv < 3; iv++) velc[iv][i] *= f;
      } else {
        for(int iv = 0; iv < 3; iv++) velc[iv][i] = 0.0;
      }
    }

    std::cout << " done." << std::endl;
    return velc;
  }

  vec2d calc_velc_field(const vecpt &ptcl, const vec1d &rho, const double vunit = 1.0)
  {
    std::cout << "Calculating velocity field ..." << std::flush;
    auto velc = calc_velc(ptcl);
    const int64_t ntot = static_cast<int64_t>(rho.size());

#pragma omp parallel for schedule(auto)
    for(int64_t i = 0; i < ntot; i++) {
      if(rho[i] > 0.0) {
        const auto f = vunit / rho[i];
        for(int iv = 0; iv < 3; iv++) velc[iv][i] *= f;
      } else {
        for(int iv = 0; iv < 3; iv++) velc[iv][i] = 0.0;
      }
    }
    return velc;
  }

  vec2d calc_sigma_field(const vecpt &ptcl, const double vunit = 1.0)
  {
    std::cout << "Calculating velocity dispersion field ..." << std::flush;
    const auto rho = calc_dens_field(ptcl);
    const auto mean_velc = calc_velc_field(ptcl, rho, 1.0);
    auto sigma = calc_sigma(ptcl, mean_velc);
    const int64_t ntot = static_cast<int64_t>(rho.size());

#pragma omp parallel for schedule(static)
    for(int64_t i = 0; i < ntot; i++) {
      if(rho[i] > 0.0) {
        const auto f = vunit * vunit / rho[i];
        // xx, xy, xz, yy, yz, zz
        for(int isig = 0; isig < 6; isig++) sigma[isig][i] *= f;
      } else {
        for(int isig = 0; isig < 6; isig++) sigma[isig][i] = 0.0;
      }
    }

    std::cout << " done." << std::endl;
    return sigma;
  }

  void output_header_base(run_param &tr, std::ofstream &fout)
  {
    fout.write((char *)&(tr.cosmology_flag), sizeof(int));
    fout.write((char *)&(tr.canonical_flag), sizeof(int));

    /* dimensions of domain decomposition */
    fout.write((char *)&(tr.nnode_x), sizeof(int));
    fout.write((char *)&(tr.nnode_y), sizeof(int));
    fout.write((char *)&(tr.nnode_z), sizeof(int));

    /* MPI ranks */
    fout.write((char *)&(tr.mpi_nproc), sizeof(int));
    fout.write((char *)&(tr.mpi_rank), sizeof(int));
    fout.write((char *)&(tr.rank_x), sizeof(int));
    fout.write((char *)&(tr.rank_y), sizeof(int));
    fout.write((char *)&(tr.rank_z), sizeof(int));

    /* global domain inforamtion */
    fout.write((char *)&(tr.xmin), sizeof(float));
    fout.write((char *)&(tr.xmax), sizeof(float));
    fout.write((char *)&(tr.ymin), sizeof(float));
    fout.write((char *)&(tr.ymax), sizeof(float));
    fout.write((char *)&(tr.zmin), sizeof(float));
    fout.write((char *)&(tr.zmax), sizeof(float));

    /* local domain inforamtion */
    fout.write((char *)&(tr.xmin_local), sizeof(float));
    fout.write((char *)&(tr.xmax_local), sizeof(float));
    fout.write((char *)&(tr.ymin_local), sizeof(float));
    fout.write((char *)&(tr.ymax_local), sizeof(float));
    fout.write((char *)&(tr.zmin_local), sizeof(float));
    fout.write((char *)&(tr.zmax_local), sizeof(float));

    /* conversion units */
    fout.write((char *)&(tr.lunit), sizeof(double));
    fout.write((char *)&(tr.munit), sizeof(double));
    fout.write((char *)&(tr.tunit), sizeof(double));

    /* physical params */
    fout.write((char *)&(tr.tnow), sizeof(float));
    fout.write((char *)&(tr.dtime), sizeof(float));
    fout.write((char *)&(tr.global_step), sizeof(int));

    /* cosmological parameters */
    fout.write((char *)&(tr.cosm), sizeof(cosmology));
  }

  void output_moment_header(run_param &tr, std::ofstream &fout)
  {
    struct meshes _mesh;
    _mesh.x_total = _mesh.x_local = _mesh.x_local_end = nmesh_x;
    _mesh.y_total = _mesh.y_local = _mesh.y_local_end = nmesh_y;
    _mesh.z_total = _mesh.z_local = _mesh.z_local_end = nmesh_z;
    _mesh.x_local_start = 0;
    _mesh.y_local_start = 0;
    _mesh.z_local_start = 0;
    _mesh.local_size = nmesh_x * nmesh_y * nmesh_z;

    output_header_base(tr, fout);
    fout.write((char *)&(_mesh), sizeof(meshes));
  }

  void output_moment_field(const vec1d &mesh, run_param &tr, const std::string &filename)
  {
    std::ofstream fout;
    fout.open(filename, std::ios_base::binary);
    output_moment_header(tr, fout);

    // float for output
    if constexpr(std::is_same<T, float>::value) {
      fout.write(reinterpret_cast<const char *>(mesh.data()), sizeof(T) * mesh.size());
    } else {
      std::vector<float> fmesh(mesh.begin(), mesh.end());
      fout.write(reinterpret_cast<const char *>(fmesh.data()), sizeof(float) * fmesh.size());
    }

    fout.close();
    std::cerr << "output to " << filename << std::endl;
  }

  void output_moment_field(const vec2d &mesh, run_param &tr, const std::string &filename)
  {
    std::ofstream fout;
    fout.open(filename, std::ios_base::binary);
    output_moment_header(tr, fout);

    for(const auto &imesh : mesh) {
      if constexpr(std::is_same<T, float>::value) {
        fout.write(reinterpret_cast<const char *>(imesh.data()), sizeof(T) * imesh.size());
      } else {
        std::vector<float> fmesh(imesh.begin(), imesh.end());
        fout.write(reinterpret_cast<const char *>(fmesh.data()), sizeof(float) * fmesh.size());
      }

    } // ielem loop

    fout.close();
    std::cerr << "output to " << filename << std::endl;
  }
};
