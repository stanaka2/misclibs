#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cassert>

#include "load_ptcl.hpp"
#include "powerspec.hpp"

int main(int argc, char **argv)
{
  if(argc < 3) {
    std::cerr << "Usage:: " << argv[0] << " gdt_snap_prefix nmesh" << std::endl;
    std::exit(EXIT_SUCCESS);
  }

  double kmin = 1e-2;
  double kmax = 50;

  std::string input_prefix = std::string(argv[1]);
  int nmesh = std::atol(argv[2]);

  std::cout << "# input prefix " << input_prefix << std::endl;

  load_ptcl<particle_pot_str> snap;
  snap.nmesh = nmesh;
  snap.scheme = 3;
  snap.read_header(input_prefix);

  double lbox(snap.h.BoxSize);
  double ptcl_mass(snap.ptcl_mass);
  int64_t nmesh_tot((int64_t)nmesh * (int64_t)nmesh * (int64_t)nmesh);
  int64_t nfft_tot((int64_t)nmesh * (int64_t)nmesh * (int64_t)(nmesh + 2));

  /* for density fluctuations */
  int type = 0;
  std::vector<float> dens_mesh(nfft_tot);
  std::fill(dens_mesh.begin(), dens_mesh.end(), 0.0);
  snap.load_gdt_and_assing(input_prefix, dens_mesh, type);

  double dens_mean = 0.0;
  for(int64_t i = 0; i < nfft_tot; i++) {
    dens_mean += dens_mesh[i];
  }
  dens_mean /= (double(nmesh) * double(nmesh) * double(nmesh));

#pragma omp parallel for
  for(int64_t i = 0; i < nfft_tot; i++) {
    dens_mesh[i] /= dens_mean;
  }

  // output_field(dens_mesh, nmesh, lbox, "dens_mesh");

  powerspec power;
  power.p = snap.scheme;
  power.lbox = lbox;
  power.nmesh = nmesh;

  power.set_kbin(kmin, kmax, 250, true);
  // power.set_kbin(kmin, kmax, nmesh, false);
  //   power.check_kbin();

#if 1
  std::vector<float> power_dens;
  std::vector<float> weight_dens;
  power.calc_power_spec(dens_mesh, power_dens, weight_dens);
  power.output_pk(power_dens, weight_dens, "pk_matter.dat");
#else
  std::vector<std::vector<float>> power_dens_ell;
  std::vector<float> weight_dens;
  power.calc_power_spec_ell(dens_mesh, power_dens_ell, weight_dens);
  power.output_pk_ell(power_dens_ell, weight_dens, "pk_matter_ell.dat");
#endif

  std::exit(EXIT_SUCCESS);
}
