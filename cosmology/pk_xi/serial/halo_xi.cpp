#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <omp.h>

#include "load_halo.hpp"
#include "correlationfunc.hpp"

const std::string suffix = ".h5";
// const std::string suffix = ".hdf5";

const int jk_block = 8;
const bool log_bin = false;

int main(int argc, char **argv)
{
  if(argc < 4) {
    std::cerr << "Usage:: " << argv[0] << " mvir_min mvir_max hdf5_halo_prefix" << std::endl;
    std::cerr << "mvir_min, mvir_max:: log10 Msun/h scale (ex. 12.0 15.0)" << std::endl;
    std::cerr << "hdf5_halo_prefix:: HDF5 halo prefix. (ex. ./halo_props/S003/halos)" << std::endl;
    std::cerr << std::endl;
    std::cerr << argv[0] << " 12.0 15.0 ./halo_props/S003/halos" << std::endl;
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

  std::cout << "# input prefix " << input_prefix << std::endl;
  std::cout << "# base file " << base_file << std::endl;

  load_halos halos;
  halos.read_header(base_file);
  halos.print_header();

  double lbox(halos.box_size);

  if(0.5 * lbox < rmax) {
    std::cerr << "\n###############" << std::endl;
    std::cerr << "Rmax=" << rmax << " Mpc/h is too large for boxsize=" << lbox << " Mpc/h." << std::endl;
    std::cerr << "Forced Rmax=" << 0.5 * lbox << " Mpc/h." << std::endl;
    std::cerr << "###############\n" << std::endl;
    rmax = 0.5 * lbox;
  }

  std::vector<float> pos;
  std::vector<float> mvir;

  halos.load_halo_pm(pos, mvir, input_prefix, suffix);

  correlation cor;
  cor.set_rbin(rmin, rmax, nr, lbox, log_bin);
  cor.mmin = mvir_min;
  cor.mmax = mvir_max;
  cor.jk_block = jk_block;

  if(cor.jk_block <= 1) {
#if 1
  cor.set_halo_pm_group(pos, mvir);
  cor.calc_xi();
  if(log_bin) cor.output_xi("xi_halo_log.dat");
  else cor.output_xi("xi_halo_lin.dat");
#else
  cor.set_halo_pm_group(pos, mvir);
  cor.calc_xi_LS();
    if(log_bin) cor.output_xi("xi_ls_halo_log.dat");
    else cor.output_xi("xi_ls_halo_lin.dat");
#endif

  } else {
#if 0
#if 1
  cor.set_halo_pm_group(pos, mvir);
  cor.calc_xi_jk();
    if(log_bin) cor.output_xi("xi_jk_halo_log.dat");
    else cor.output_xi("xi_jk_halo_lin.dat");
#else
  cor.set_halo_pm_group(pos, mvir);
  cor.calc_xi_jk_LS();
    if(log_bin) cor.output_xi("xi_jk_ls_halo_log.dat");
    else cor.output_xi("xi_jk_ls_halo_lin.dat");
#endif
#endif
    ;
  }

  return EXIT_SUCCESS;
}
