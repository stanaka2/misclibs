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

const bool log_bin = false;

int main(int argc, char **argv)
{
  if(argc < 5) {
    std::cerr << "Usage:: " << argv[0] << " mvir_min mvir_max hdf5_halo_prefix nmesh (output_filename)" << std::endl;
    std::cerr << "mvir_min, mvir_max:: log10 Msun/h scale (ex. 12.0 15.0)" << std::endl;
    std::cerr << "hdf5_halo_prefix:: HDF5 halo prefix. (ex. ./halo_props/S003/halos)" << std::endl;
    std::cerr << "nmesh:: FFT mesh of 1D (ex. 1024)" << std::endl;
    std::cerr << "(output_filename):: opition. output filename" << std::endl;
    std::cerr << std::endl;
    std::cerr << argv[0] << " 12.0 15.0 ./halo_props/S003/halos 1024 output.dat" << std::endl;
    std::exit(EXIT_SUCCESS);
  }

  float mvir_min, mvir_max;
  mvir_min = pow(10.0, atof(argv[1]));
  mvir_max = pow(10.0, atof(argv[2]));

  double kmin = 1e-2;
  double kmax = 50;
  int nk = 250;

  std::string input_prefix = std::string(argv[3]);
  std::string base_file = input_prefix + ".0" + suffix;

  int nmesh = std::atol(argv[4]);

  std::string output_filename = "pk_halo.dat";
  if(argc == 6) output_filename = std::string(argv[5]);

  std::cout << "# input prefix " << input_prefix << std::endl;
  std::cout << "# base file " << base_file << std::endl;
  std::cout << "# output filename " << output_filename << std::endl;
  std::cout << "# Mmin, Mmax " << mvir_min << ", " << mvir_max << std::endl;
  std::cout << "# kmin, kmax, Nk " << kmin << ", " << kmax << ", " << nk << std::endl;
  std::cout << "# log_bin " << std::boolalpha << log_bin << std::endl;
  std::cout << "# FFT mesh " << nmesh << "^3" << std::endl;

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
  /* halo selection */
  std::vector<float> ones(mvir.size(), 0.0);
  for(size_t i = 0; i < mvir.size(); i++) {
    if(mvir[i] > mvir_min && mvir[i] < mvir_max) ones[i] = 1.0;
  }

  halo_assign_mesh(pos, ones, dens_mesh, nmesh, lbox, halos.scheme);

  double dens_mean = 0.0;
  for(int64_t i = 0; i < nfft_tot; i++) {
    dens_mean += dens_mesh[i];
  }
  dens_mean /= (double(nmesh) * double(nmesh) * double(nmesh));

  for(int64_t i = 0; i < nfft_tot; i++) {
    dens_mesh[i] = dens_mesh[i] / dens_mean - 1.0;
  }
#endif

  powerspec power;
  power.p = halos.scheme;
  power.lbox = lbox;
  power.nmesh = nmesh;

  power.set_kbin(kmin, kmax, nk, log_bin);
  //   power.check_kbin();

#if 1
  std::vector<float> power_dens;
  std::vector<float> weight_dens;
  power.calc_power_spec(dens_mesh, power_dens, weight_dens);
  power.output_pk(power_dens, weight_dens, output_filename);
#else
  std::vector<std::vector<float>> power_dens_ell;
  std::vector<float> weight_dens;
  power.calc_power_spec_ell(dens_mesh, power_dens_ell, weight_dens);
  power.output_pk_ell(power_dens_ell, weight_dens, output_filename);
#endif

  std::exit(EXIT_SUCCESS);
}
