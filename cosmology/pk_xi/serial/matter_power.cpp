#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cassert>

#include "load_ptcl.hpp"
#include "powerspec.hpp"
#include "base_opts.hpp"

int main(int argc, char **argv)
{
  BaseOptions opt(argc, argv);

  int nk = opt.nk;
  float kmin = opt.krange[0];
  float kmax = opt.krange[1];
  bool log_bin = opt.log_bin;
  auto nmesh = opt.nmesh;

  std::cout << "# input prefix " << opt.input_prefix << std::endl;
  std::cout << "# output filename " << opt.output_filename << std::endl;
  std::cout << "# kmin, kmax, Nk " << kmin << ", " << kmax << ", " << nk << std::endl;
  std::cout << "# log_bin " << std::boolalpha << opt.log_bin << std::endl;
  std::cout << "# FFT mesh " << nmesh << "^3" << std::endl;
  std::cout << std::endl;

  load_ptcl<particle_pot_str> snap;
  snap.nmesh = nmesh;
  snap.scheme = opt.p_assign;
  snap.read_header(opt.input_prefix);

  double lbox(snap.h.BoxSize);
  double ptcl_mass(snap.ptcl_mass);
  int64_t nmesh_tot((int64_t)nmesh * (int64_t)nmesh * (int64_t)nmesh);
  int64_t nfft_tot((int64_t)nmesh * (int64_t)nmesh * (int64_t)(nmesh + 2));

  /* for density fluctuations */
  int type = 0;
  std::vector<float> dens_mesh(nfft_tot);
  std::fill(dens_mesh.begin(), dens_mesh.end(), 0.0);
  snap.load_gdt_and_assing(opt.input_prefix, dens_mesh, type);

  double dens_mean = 0.0;
  for(int64_t i = 0; i < nfft_tot; i++) {
    dens_mean += dens_mesh[i];
  }
  dens_mean /= (double(nmesh) * double(nmesh) * double(nmesh));

#pragma omp parallel for
  for(int64_t i = 0; i < nfft_tot; i++) {
    dens_mesh[i] = dens_mesh[i] / dens_mean - 1.0;
  }

  // output_field(dens_mesh, nmesh, lbox, "dens_mesh");

  powerspec power;
  power.p = snap.scheme;
  power.lbox = lbox;
  power.nmesh = nmesh;

  power.set_kbin(kmin, kmax, nk, log_bin);
  //   power.check_kbin();

  power.shotnoise_corr = !opt.no_shotnoise;
  power.set_shotnoise(snap.npart_tot);

#if 1
  std::vector<float> power_dens;
  std::vector<float> weight_dens;
  power.calc_power_spec(dens_mesh, power_dens, weight_dens);
  power.output_pk(power_dens, weight_dens, opt.output_filename);
#else
  std::vector<std::vector<float>> power_dens_ell;
  std::vector<float> weight_dens;
  power.calc_power_spec_ell(dens_mesh, power_dens_ell, weight_dens);
  power.output_pk_ell(power_dens_ell, weight_dens, opt.output_filename);
#endif

  std::exit(EXIT_SUCCESS);
}
