#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <fstream>
#include <cassert>
#include <iostream>

#include <hdf5.h>

#include "run_param.hpp"
#include "cosmology.hpp"

struct particle_str {
  uint64_t id;
  float mass;
  float xpos, ypos, zpos;
  float xvel, yvel, zvel;
};

class swift_particles
{
public:
  using vecpt = std::vector<particle_str>;
  vecpt ptcls;

  std::string input_file;
  std::string comp_type = "nu";
  uint64_t npart_total, npart_local;
  double boxsize;

  swift_particles(std::string _input_file, std::string _comp_type = "nu")
      : input_file(_input_file), comp_type(_comp_type) {};
  ~swift_particles() {};

  size_t input_ptcl_header(run_param &tr)
  {
    hid_t file_id = H5Fopen(input_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    tr.cosmology_flag = 1;
    tr.canonical_flag = 0;

    tr.nnode_x = tr.nnode_y = tr.nnode_z = 1;
    tr.rank_x = tr.rank_y = tr.rank_z = 0;

    tr.xmin = tr.ymin = tr.zmin = 0.0;
    tr.xmax = tr.ymax = tr.zmax = 1.0;

    tr.xmin_local = tr.ymin_local = tr.zmin_local = 0.0;
    tr.xmax_local = tr.ymax_local = tr.zmax_local = 1.0;
    tr.lunit = tr.munit = tr.tunit = 1.0;

    {
      hid_t group_id = H5Gopen(file_id, "/Header", H5P_DEFAULT);
      hid_t attr_id = H5Aopen(group_id, "BoxSize", H5P_DEFAULT);
      hid_t space_id = H5Aget_space(attr_id);
      hsize_t dims[3];
      int ndims = H5Sget_simple_extent_dims(space_id, dims, nullptr);

      std::vector<double> lsize(dims[0]);
      H5Aread(attr_id, H5T_NATIVE_DOUBLE, lsize.data());
      boxsize = lsize[0];
      std::cout << "BoxSize = " << boxsize << "\n";
      H5Sclose(space_id);
      H5Aclose(attr_id);
      H5Gclose(group_id);
    }

    {
      hid_t group_id = H5Gopen(file_id, "/Header", H5P_DEFAULT);
      hid_t attr_id = H5Aopen(group_id, "NumPart_ThisFile", H5P_DEFAULT);
      hsize_t dims[1];
      H5Sget_simple_extent_dims(H5Aget_space(attr_id), dims, nullptr);
      std::vector<uint64_t> np_this(dims[0]);
      H5Aread(attr_id, H5T_NATIVE_UINT64, np_this.data());
      H5Aclose(attr_id);

      // NumPart_Total
      hid_t attr_tot = H5Aopen(group_id, "NumPart_Total", H5P_DEFAULT);
      std::vector<uint32_t> np_tot_l(dims[0]);
      H5Aread(attr_tot, H5T_NATIVE_UINT32, np_tot_l.data());
      H5Aclose(attr_tot);

      // NumPart_Total_HighWord
      hid_t attr_high = H5Aopen(group_id, "NumPart_Total_HighWord", H5P_DEFAULT);
      std::vector<uint32_t> np_tot_h(dims[0]);
      H5Aread(attr_high, H5T_NATIVE_UINT32, np_tot_h.data());
      H5Aclose(attr_high);

      int idx = (comp_type == "cdm") ? 1 : 6;
      uint64_t this_file = np_this[idx];
      uint64_t total = (uint64_t)np_tot_l[idx] + ((uint64_t)np_tot_h[idx] << 32);

      std::cout << comp_type << ": this_file=" << this_file << " total=" << total << "\n";

      npart_total = total;
      npart_local = this_file;
    }

    {
      hid_t group_id = H5Gopen(file_id, "/Cosmology", H5P_DEFAULT);
      hid_t attr_id = H5Aopen(group_id, "h", H5P_DEFAULT);
      double tmp;
      H5Aread(attr_id, H5T_NATIVE_DOUBLE, &tmp);
      std::cout << "h = " << tmp << "\n";
      tr.cosm.hubble = tmp;
      H5Aclose(attr_id);

      // this Omega_m is Omega_cdm + Omega_b
      // this Omega_m is not Omega_cdm + Omega_b + Omega_nu
      attr_id = H5Aopen(group_id, "Omega_m", H5P_DEFAULT);
      H5Aread(attr_id, H5T_NATIVE_DOUBLE, &tmp);
      std::cout << "Omega_m = " << tmp << "\n";
      tr.cosm.omega_m = tmp;
      H5Aclose(attr_id);

      attr_id = H5Aopen(group_id, "Omega_b", H5P_DEFAULT);
      H5Aread(attr_id, H5T_NATIVE_DOUBLE, &tmp);
      std::cout << "Omega_b = " << tmp << "\n";
      tr.cosm.omega_b = tmp;
      H5Aclose(attr_id);

      attr_id = H5Aopen(group_id, "Omega_lambda", H5P_DEFAULT);
      H5Aread(attr_id, H5T_NATIVE_DOUBLE, &tmp);
      std::cout << "Omega_v = " << tmp << "\n";
      tr.cosm.omega_v = tmp;
      H5Aclose(attr_id);

      attr_id = H5Aopen(group_id, "Omega_nu_0", H5P_DEFAULT);
      H5Aread(attr_id, H5T_NATIVE_DOUBLE, &tmp);
      std::cout << "Omega_nu = " << tmp << "\n";
      tr.cosm.omega_nu = tmp;
      H5Aclose(attr_id);

      // this Omega_m is Omega_cdm + Omega_b + Omega_nu
      tr.cosm.omega_m += tr.cosm.omega_nu;

      attr_id = H5Aopen(group_id, "Omega_r", H5P_DEFAULT);
      H5Aread(attr_id, H5T_NATIVE_DOUBLE, &tmp);
      std::cout << "Omega_r = " << tmp << "\n";
      tr.cosm.omega_r = tmp;
      H5Aclose(attr_id);

      attr_id = H5Aopen(group_id, "Redshift", H5P_DEFAULT);
      H5Aread(attr_id, H5T_NATIVE_DOUBLE, &tmp);
      std::cout << "Redshift = " << tmp << "\n";
      tr.znow = tmp;
      tr.anow = 1.0 / (1.0 + tr.znow);
      H5Aclose(attr_id);

      H5Gclose(group_id);
    }

    {
      hid_t group_id = H5Gopen(file_id, "/Cosmology", H5P_DEFAULT);

      // M_nu_eV -> mass[3], sum_mass
      if(H5Aexists(group_id, "M_nu_eV") > 0) {
        hid_t attr_id = H5Aopen(group_id, "M_nu_eV", H5P_DEFAULT);
        hid_t space_id = H5Aget_space(attr_id);
        hsize_t dims[1] = {0};
        H5Sget_simple_extent_dims(space_id, dims, nullptr);
        std::vector<double> m(std::max<hsize_t>(dims[0], 3));
        H5Aread(attr_id, H5T_NATIVE_DOUBLE, m.data());
        H5Sclose(space_id);
        H5Aclose(attr_id);

        tr.cosm.nu.sum_mass = 0.0f;
        for(int i = 0; i < 3; ++i) {
          tr.cosm.nu.mass[i] = (i < (int)dims[0]) ? (float)m[i] : 0.0f;
          tr.cosm.nu.sum_mass += tr.cosm.nu.mass[i];
        }
      }

      // deg_nu -> deg[3]
      if(H5Aexists(group_id, "deg_nu") > 0) {
        hid_t attr_id = H5Aopen(group_id, "deg_nu", H5P_DEFAULT);
        hid_t space_id = H5Aget_space(attr_id);
        hsize_t dims[1] = {0};
        H5Sget_simple_extent_dims(space_id, dims, nullptr);
        std::vector<double> d(std::max<hsize_t>(dims[0], 3), 0.0);
        H5Aread(attr_id, H5T_NATIVE_DOUBLE, d.data());
        H5Sclose(space_id);
        H5Aclose(attr_id);

        for(int i = 0; i < 3; ++i) tr.cosm.nu.deg[i] = (int)std::lround(d[i]);
      }

      // deg_nu_tot -> mass_nu_num
      if(H5Aexists(group_id, "deg_nu_tot") > 0) {
        hid_t attr_id = H5Aopen(group_id, "deg_nu_tot", H5P_DEFAULT);
        double v = 0.0;
        H5Aread(attr_id, H5T_NATIVE_DOUBLE, &v);
        H5Aclose(attr_id);
        tr.cosm.nu.mass_nu_num = (int)std::lround(v);
      }

      double rho_c0 = 0.0;
      {
        hid_t attr_id = H5Aopen(group_id, "Critical density at redshift zero [internal units]", H5P_DEFAULT);
        H5Aread(attr_id, H5T_NATIVE_DOUBLE, &rho_c0);
        H5Aclose(attr_id);
      }
      H5Gclose(group_id);

      for(int i = 0; i < 3; ++i) tr.cosm.nu.frac[i] = 1.0;

      std::cout << "nu_mass_sum(eV)=" << tr.cosm.nu.sum_mass << std::endl;
      std::cout << "nu mass(eV)=[" << tr.cosm.nu.mass[0] << "," << tr.cosm.nu.mass[1] << "," << tr.cosm.nu.mass[2]
                << "]" << std::endl;
      std::cout << "mass_nu_num=" << tr.cosm.nu.mass_nu_num << "\n";
      std::cout << "nu_deg=[" << tr.cosm.nu.deg[0] << "," << tr.cosm.nu.deg[1] << "," << tr.cosm.nu.deg[2] << "]"
                << std::endl;

      const auto volume = boxsize * boxsize * boxsize;
      tr.Mnu_bg_box = tr.cosm.omega_nu * rho_c0 * volume;
    }

    double om = tr.cosm.omega_m;
    double ov = tr.cosm.omega_v;
    double _or = tr.cosm.omega_r; // batting `or ||`
    double anow = tr.anow;
    tr.hnow = sqrt(1.0 + om * (1.0 / anow - 1.0) + ov * (SQR(anow) - 1.0) + _or * (1.0 / SQR(anow) - 1.0)) / anow;
    tr.tnow = atotime(anow, tr.cosm);

    tr.vunit = 1.0 / sqrt(tr.anow);

    std::cout << "read header from " << input_file << std::endl;

    return 0;
  }

  void load_ptcls(run_param &tr)
  {
    auto hsize = input_ptcl_header(tr);

    const std::string group_name = (comp_type == "cdm") ? "/DMParticles" : "/NeutrinoParticles";

    hid_t file_id = H5Fopen(input_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t group_id = H5Gopen(file_id, group_name.c_str(), H5P_DEFAULT);

    hid_t ds_ids = H5Dopen(group_id, "ParticleIDs", H5P_DEFAULT);
    hid_t sp_ids = H5Dget_space(ds_ids);
    hsize_t dims_ids[2] = {0, 0};
    int nd_ids = H5Sget_simple_extent_dims(sp_ids, dims_ids, nullptr);
    const size_t N = (nd_ids >= 1) ? (size_t)dims_ids[0] : 0;

    std::cout << "read particle IDs" << std::endl;

    std::vector<uint64_t> ids(N);
    H5Dread(ds_ids, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, ids.data());
    H5Sclose(sp_ids);
    H5Dclose(ds_ids);

    hid_t ds_x = H5Dopen(group_id, "Coordinates", H5P_DEFAULT);
    hid_t sp_x = H5Dget_space(ds_x);
    hsize_t dims_x[2] = {0, 0};
    H5Sget_simple_extent_dims(sp_x, dims_x, nullptr);
    std::vector<double> X(3 * N);
    H5Dread(ds_x, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, X.data());
    H5Sclose(sp_x);
    H5Dclose(ds_x);

    std::cout << "read particle position" << std::endl;

    hid_t ds_v = H5Dopen(group_id, "Velocities", H5P_DEFAULT);
    hid_t sp_v = H5Dget_space(ds_v);
    hsize_t dims_v[2] = {0, 0};
    H5Sget_simple_extent_dims(sp_v, dims_v, nullptr);
    std::vector<double> V(3 * N);
    H5Dread(ds_v, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, V.data());
    H5Sclose(sp_v);
    H5Dclose(ds_v);

    std::cout << "read particle velocity" << std::endl;

    std::vector<double> M(N, 0.0);
    const int idx = (comp_type == "cdm") ? 1 : 6;
    hid_t ds_m = H5Dopen(group_id, "Masses", H5P_DEFAULT);
    hid_t sp_m = H5Dget_space(ds_m);
    hsize_t dims_m[1] = {0};
    H5Sget_simple_extent_dims(sp_m, dims_m, nullptr);
    H5Dread(ds_m, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, M.data());
    H5Sclose(sp_m);
    H5Dclose(ds_m);

    std::cout << "read particle mass" << std::endl;

    if(comp_type == "nu") {
      std::cout << "Here the neutrino mass used for deposition is `mass x weight`, hence negative values are possible."
                << std::endl;
    }

    ptcls.clear();
    ptcls.resize(N);

#pragma omp parallel for
    for(int64_t i = 0; i < (int64_t)N; i++) {
      particle_str p;
      p.id = ids[i];
      p.mass = M[i];

      // [0, 1]
      p.xpos = X[3 * i + 0] / boxsize;
      p.ypos = X[3 * i + 1] / boxsize;
      p.zpos = X[3 * i + 2] / boxsize;

      p.xvel = V[3 * i + 0] * tr.vunit;
      p.yvel = V[3 * i + 1] * tr.vunit;
      p.zvel = V[3 * i + 2] * tr.vunit;
      ptcls[i] = p;
    }

    npart_local = N;
    H5Gclose(group_id);
    H5Fclose(file_id);
  }

  void free_ptcls()
  {
    ptcls.clear();
    ptcls.shrink_to_fit();
  }
};
