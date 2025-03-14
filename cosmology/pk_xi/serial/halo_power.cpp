#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cassert>

#include "field.hpp"
#include "load_halo.hpp"
#include "powerspec.hpp"

const std::string suffix = ".h5";
// const std::string suffix = ".hdf5";

int main(int argc, char **argv)
{
  if(argc < 3) {
    std::cerr << "Usage:: " << argv[0] << "hdf5_halo_prefix nmesh" << std::endl;
    std::exit(EXIT_SUCCESS);
  }

  double mmin = 0;  // 10^mmin
  double mmax = 20; // 10^mmax

  double kmin = 1e-2;
  double kmax = 50;

  std::string input_prefix = std::string(argv[1]);
  std::string base_file = input_prefix + ".0" + suffix;

  int nmesh = std::atol(argv[2]);

  std::cout << "# input prefix " << input_prefix << std::endl;
  std::cout << "# base file " << base_file << std::endl;

  load_halos halos;
  halos.read_header(base_file);
  halos.print_header();
  halos.scheme = 3;

  std::vector<float> pos;
  std::vector<float> mvir;

  halos.load_halo_pm(pos, mvir, input_prefix, suffix);

  int64_t nmesh_tot((int64_t)nmesh * (int64_t)nmesh * (int64_t)nmesh);
  int64_t nfft_tot((int64_t)nmesh * (int64_t)nmesh * (int64_t)(nmesh + 2));
  double lbox(halos.box_size);

  std::vector<float> dens_mesh(nfft_tot);
  std::fill(dens_mesh.begin(), dens_mesh.end(), 0.0);

#if 0
/* halo mass density filed */
halo_assign_mesh(pos, mvir, dens_mesh, nmesh, lbox, halos.scheme);

#else
  /* halo number density filed */
  std::vector<float> ones(mvir.size(), 1.0);
  halo_assign_mesh(pos, ones, dens_mesh, nmesh, lbox, halos.scheme);

  double dens_mean = 0.0;
  for(int64_t i = 0; i < nfft_tot; i++) {
    dens_mean += dens_mesh[i];
  }
  dens_mean /= (double(nmesh) * double(nmesh) * double(nmesh));

  for(int64_t i = 0; i < nfft_tot; i++) {
    dens_mesh[i] /= dens_mean;
  }
#endif

  powerspec power;
  power.p = halos.scheme;
  power.lbox = lbox;
  power.nmesh = nmesh;

  power.set_kbin(kmin, kmax, 250, true);
  // power.set_kbin(kmin, kmax, nmesh, false);
  //   power.check_kbin();

#if 1
  std::vector<float> power_dens;
  std::vector<float> weight_dens;
  power.calc_power_spec(dens_mesh, power_dens, weight_dens);
  power.output_pk(power_dens, weight_dens, "pk_halo.dat");
#else
  std::vector<std::vector<float>> power_dens_ell;
  std::vector<float> weight_dens;
  power.calc_power_spec_ell(dens_mesh, power_dens_ell, weight_dens);
  power.output_pk_ell(power_dens_ell, weight_dens, "pk_halo_ell.dat");
#endif

  std::exit(EXIT_SUCCESS);
}
