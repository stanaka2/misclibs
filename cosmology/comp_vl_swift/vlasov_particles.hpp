#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <fstream>
#include <cassert>
#include <iostream>

#include "run_param.hpp"
#include "cosmology.hpp"

struct particle_str {
  uint64_t id;
  float mass;
  float xpos, ypos, zpos;
  float xvel, yvel, zvel;
};

#pragma pack(push, 1)
struct particle_f32 {
  uint64_t indx;
  float mass;
  float xpos, ypos, zpos;
  float xvel, yvel, zvel;
};
#pragma pack(pop)

class vlasov_particles
{
public:
  using vecpt = std::vector<particle_str>;
  vecpt ptcls;

  uint64_t npart_total, npart_local;
  std::string prefix, suffix;

  vlasov_particles(const std::string &prefix, const std::string &suffix) : prefix(prefix), suffix(suffix) {}
  ~vlasov_particles() {};

  std::string set_filepath(int ix, int iy, int iz) const
  {
    char buf[64];
    // std::sprintf(buf, "x%03d_y%03d_z%03d", ix, iy, iz);
    // std::sprintf(buf, "x%03d_y%03d/z%03d", ix, iy, iz);
    std::sprintf(buf, "/x%03d/y%03d/z%03d", ix, iy, iz);
    std::string filepath = prefix + buf + suffix;
    return filepath;
  }

  size_t input_ptcl_header(run_param &tr, std::ifstream &fin)
  {
    size_t head_size = 0;
    /* flag */
    head_size += fin.read((char *)&(tr.cosmology_flag), sizeof(int)).gcount();
    head_size += fin.read((char *)&(tr.canonical_flag), sizeof(int)).gcount();

    /* dimensions of domain decomposition */
    head_size += fin.read((char *)&(tr.nnode_x), sizeof(int)).gcount();
    head_size += fin.read((char *)&(tr.nnode_y), sizeof(int)).gcount();
    head_size += fin.read((char *)&(tr.nnode_z), sizeof(int)).gcount();

    // head_size += fin.read((char *)&(tr.mpi_nproc), sizeof(int)).gcount();
    // head_size += fin.read((char *)&(tr.mpi_rank), sizeof(int)).gcount();

    int tmp;
    head_size += fin.read((char *)&(tmp), sizeof(int)).gcount();
    head_size += fin.read((char *)&(tmp), sizeof(int)).gcount();

    head_size += fin.read((char *)&(tr.rank_x), sizeof(int)).gcount();
    head_size += fin.read((char *)&(tr.rank_y), sizeof(int)).gcount();
    head_size += fin.read((char *)&(tr.rank_z), sizeof(int)).gcount();

    /* global domain inforamtion */
    head_size += fin.read((char *)&(tr.xmin), sizeof(float)).gcount();
    head_size += fin.read((char *)&(tr.xmax), sizeof(float)).gcount();
    head_size += fin.read((char *)&(tr.ymin), sizeof(float)).gcount();
    head_size += fin.read((char *)&(tr.ymax), sizeof(float)).gcount();
    head_size += fin.read((char *)&(tr.zmin), sizeof(float)).gcount();
    head_size += fin.read((char *)&(tr.zmax), sizeof(float)).gcount();

    /* local domain inforamtion */
    head_size += fin.read((char *)&(tr.xmin_local), sizeof(float)).gcount();
    head_size += fin.read((char *)&(tr.xmax_local), sizeof(float)).gcount();
    head_size += fin.read((char *)&(tr.ymin_local), sizeof(float)).gcount();
    head_size += fin.read((char *)&(tr.ymax_local), sizeof(float)).gcount();
    head_size += fin.read((char *)&(tr.zmin_local), sizeof(float)).gcount();
    head_size += fin.read((char *)&(tr.zmax_local), sizeof(float)).gcount();

    /* conversion of units */
    head_size += fin.read((char *)&(tr.lunit), sizeof(double)).gcount();
    head_size += fin.read((char *)&(tr.munit), sizeof(double)).gcount();
    head_size += fin.read((char *)&(tr.tunit), sizeof(double)).gcount();

    /* physical params */
    head_size += fin.read((char *)&(tr.tnow), sizeof(float)).gcount();
    head_size += fin.read((char *)&(tr.dtime), sizeof(float)).gcount();
    head_size += fin.read((char *)&(tr.global_step), sizeof(int)).gcount();

    /* cosmological parameters */
    head_size += fin.read((char *)&(tr.cosm), sizeof(cosmology)).gcount();
    tr.znow = timetoz(tr.tnow, tr.cosm);
    tr.anow = 1.0 / (1.0 + tr.znow);

    double om = tr.cosm.omega_m;
    double ov = tr.cosm.omega_v;
    double _or = tr.cosm.omega_r; // batting `or ||`
    double anow = tr.anow;
    tr.hnow = sqrt(1.0 + om * (1.0 / anow - 1.0) + ov * (SQR(anow) - 1.0) + _or * (1.0 / SQR(anow) - 1.0)) / anow;

    head_size += fin.read((char *)&(tr.npart), sizeof(uint64_t)).gcount();
    head_size += fin.read((char *)&(tr.npart_total), sizeof(uint64_t)).gcount();

    double vunit;
    constexpr double pc = 3.0856775814913673e+13;  // [km]
    constexpr double mpc = 3.0856775814913673e+19; // [km]

    if(tr.cosmology_flag) {
      auto lbox = tr.lunit * tr.cosm.hubble / mpc;
      /* L / T to cm / s unit to km / s */
      vunit = 1.0e-5 * (tr.lunit / tr.tunit);
      vunit *= tr.anow;
    } else {
      auto lbox = 1.0;
      vunit = (tr.lunit / tr.tunit);
    }

    tr.vunit = vunit;

    return head_size;
  }

  void load_ptcls(run_param &tr)
  {
    auto base_file = set_filepath(0, 0, 0);
    std::cout << __func__ << " : " << base_file << std::endl;

    std::ifstream fin;
    fin.open(base_file, std::ios_base::binary);
    input_ptcl_header(tr, fin);
    fin.close();

    using vecpt_f32 = std::vector<particle_f32>;

    /* connection file rank ID and running exec rank */
    const int nfile_x = tr.nnode_x;
    const int nfile_y = tr.nnode_y;
    const int nfile_z = tr.nnode_z;

    int64_t nptcl_loc_tot = 0;

    std::cout << "nfile(x,y,z)" << nfile_x << " " << nfile_y << " " << nfile_z << std::endl;

    /* read header */
    for(int ix = 0; ix < nfile_x; ix++) {
      for(int iy = 0; iy < nfile_y; iy++) {
        for(int iz = 0; iz < nfile_z; iz++) {
          std::ifstream fin;
          auto input_filename = set_filepath(ix, iy, iz);
          fin.open(input_filename, std::ios_base::binary);

          run_param tmp_run;
          input_ptcl_header(tmp_run, fin);
          fin.close();

          assert(tmp_run.rank_x == ix);
          assert(tmp_run.rank_y == iy);
          assert(tmp_run.rank_z == iz);

          nptcl_loc_tot += (int64_t)tmp_run.npart;
          std::cout << input_filename << " " << tmp_run.npart << std::endl;
        }
      }
    }

    ptcls.reserve(nptcl_loc_tot);

    /* read body */
    for(int ix = 0; ix < nfile_x; ix++) {
      for(int iy = 0; iy < nfile_y; iy++) {
        for(int iz = 0; iz < nfile_z; iz++) {
          std::ifstream fin;
          auto input_filename = set_filepath(ix, iy, iz);
          fin.open(input_filename, std::ios_base::binary);

          run_param tmp_run;
          input_ptcl_header(tmp_run, fin);

          const int64_t loc_size = (int64_t)tmp_run.npart;
          vecpt_f32 io_ptcls_f(loc_size);
          fin.read((char *)io_ptcls_f.data(), sizeof(particle_f32) * loc_size);
          fin.close();

          vecpt io_ptcls(loc_size);
#pragma omp parallel for
          for(int64_t i = 0; i < loc_size; i++) {
            io_ptcls[i].id = io_ptcls_f[i].indx;
            io_ptcls[i].mass = io_ptcls_f[i].mass;

            io_ptcls[i].xpos = io_ptcls_f[i].xpos;
            io_ptcls[i].ypos = io_ptcls_f[i].ypos;
            io_ptcls[i].zpos = io_ptcls_f[i].zpos;

            // to km/s
            io_ptcls[i].xvel = io_ptcls_f[i].xvel * tr.vunit;
            io_ptcls[i].yvel = io_ptcls_f[i].yvel * tr.vunit;
            io_ptcls[i].zvel = io_ptcls_f[i].zvel * tr.vunit;
          }

          ptcls.insert(ptcls.end(), io_ptcls.begin(), io_ptcls.end());

          io_ptcls.clear();
          io_ptcls.shrink_to_fit();

          std::cout << input_filename << " " << tmp_run.npart << std::endl;
        }
      }
    } // rank_x,y,z loop

    npart_total = tr.npart_total;
    npart_local = nptcl_loc_tot;
  }

  //  void load_swift_ptcls() {}

  void free_ptcls()
  {
    ptcls.clear();
    ptcls.shrink_to_fit();
  }
};
