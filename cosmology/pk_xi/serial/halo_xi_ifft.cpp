#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <omp.h>

#include "field.hpp"
#include "load_halo.hpp"
#include "correlationfunc.hpp"

const std::string suffix = ".h5";
// const std::string suffix = ".hdf5";

const bool log_bin = false;

int main(int argc, char **argv)
{
  if(argc < 5) {
    std::cerr << "Usage:: " << argv[0] << " mvir_min mvir_max hdf5_halo_prefix nmesh" << std::endl;
    std::cerr << "mvir_min, mvir_max:: log10 Msun/h scale (ex. 12.0 15.0)" << std::endl;
    std::cerr << "hdf5_halo_prefix:: HDF5 halo prefix. (ex. ./halo_props/S003/halos)" << std::endl;
    std::cerr << "nmesh:: FFT mesh of 1D (ex. 1024)" << std::endl;
    std::cerr << std::endl;
    std::cerr << argv[0] << " 12.0 15.0 ./halo_props/S003/halos 1024" << std::endl;
    std::exit(EXIT_SUCCESS);
  }

  float mvir_min, mvir_max;
  mvir_min = pow(10.0, atof(argv[1]));
  mvir_max = pow(10.0, atof(argv[2]));

  int nr = 100;
  float rmin, rmax; // [Mpc/h]
  rmin = 0.1;
  rmax = 150.0;

  std::string input_prefix = std::string(argv[3]);
  std::string base_file = input_prefix + ".0" + suffix;
  int nmesh = std::atol(argv[4]);

  std::cout << "# input prefix " << input_prefix << std::endl;
  std::cout << "# base file " << base_file << std::endl;

  load_halos halos;
  halos.read_header(base_file);
  halos.print_header();
  halos.scheme = 3;

  double lbox(halos.box_size);
  std::vector<float> pos;
  std::vector<float> mvir;

  halos.load_halo_pm(pos, mvir, input_prefix, suffix);

  int64_t nmesh_tot((int64_t)nmesh * (int64_t)nmesh * (int64_t)nmesh);
  int64_t nfft_tot((int64_t)nmesh * (int64_t)nmesh * (int64_t)(nmesh + 2));

  std::vector<float> dens_mesh(nfft_tot);
  std::fill(dens_mesh.begin(), dens_mesh.end(), 0.0);

  /* halo number density filed */
  /* halo selection */
  std::vector<float> ones(mvir.size(), 0.0);
  for(size_t i = 0; i < mvir.size(); i++) {
    if(mvir[i] > mvir_min && mvir[i] < mvir_max) ones[i] = 1.0;
  }
  halo_assign_mesh(pos, ones, dens_mesh, nmesh, lbox, halos.scheme);

  double dens_mean = 0.0;
  for(int64_t i = 0; i < nfft_tot; i++) dens_mean += dens_mesh[i];
  dens_mean /= (double(nmesh) * double(nmesh) * double(nmesh));
  for(int64_t i = 0; i < nfft_tot; i++) dens_mesh[i] = dens_mesh[i] / dens_mean - 1.0;

  correlation cor;
  cor.p = halos.scheme;
  cor.lbox = lbox;
  cor.nmesh = nmesh;
  cor.mmin = mvir_min;
  cor.mmax = mvir_max;

  cor.set_rbin(rmin, rmax, nr, lbox, log_bin);

  std::vector<float> weight;
  cor.calc_xi_ifft(dens_mesh, weight);

  if(log_bin) cor.output_xi_ifft("xi_halo_ifft_log.dat", weight);
  else cor.output_xi_ifft("xi_halo_ifft_lin.dat", weight);

  return EXIT_SUCCESS;
}
