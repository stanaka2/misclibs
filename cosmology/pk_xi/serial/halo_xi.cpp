#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <omp.h>

#include "load_halo.hpp"
#include "correlationfunc.hpp"

const std::string suffix = ".h5";
// const std::string suffix = ".hdf5";

int main(int argc, char **argv)
{
  if(argc < 4) {
    std::cerr << "Usage:: " << argv[0] << " mvir_min mvir_max input_prefix" << std::endl;
    // std::cerr << "rmin, rmax:: Mpc/h scale" << std::endl;
    std::cerr << "mvir_min, mvir_max:: log10 Msun/h scale" << std::endl;
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
  std::vector<float> pos;
  std::vector<float> mvir;

  halos.load_halo_pm(pos, mvir, input_prefix, suffix);

  bool log_bin = false;

  correlation cor;
  cor.set_rbin(rmin, rmax, nr, lbox, log_bin);
  cor.mmin = mvir_min;
  cor.mmax = mvir_max;

#if 1
  cor.set_halo_pm_group(pos, mvir);
  cor.calc_xi();
  if(log_bin) cor.output_xi("xi_halo_log.dat");
  else cor.output_xi("xi_halo_lin.dat");

#elif 0
  cor.set_halo_pm_group(pos, mvir);
  cor.calc_xi_LS();
  cor.output_xi_LS("xi_ls_halo.dat");
#elif 0
  cor.set_halo_pm_group(pos, mvir);
  cor.calc_xi_jk();
  cor.output_xi_jk("xi_jk_halo.dat");
#elif 0
  cor.set_halo_pm_group(pos, mvir);
  cor.calc_xi_jk_LS();
  cor.output_xi_jk_LS("xi_jk_ls_halo.dat");
#endif

  return EXIT_SUCCESS;
}
