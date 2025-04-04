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

constexpr bool log_bin = false;
constexpr bool use_Landy_Szalay = false;

int main(int argc, char **argv)
{
  if(argc < 5) {
    std::cerr << "Usage:: " << argv[0] << " mvir_min mvir_max hdf5_halo_prefix jk_block (output_filename)" << std::endl;
    std::cerr << "mvir_min, mvir_max:: log10 Msun/h scale (ex. 12.0 15.0)" << std::endl;
    std::cerr << "hdf5_halo_prefix:: HDF5 halo prefix. (ex. ./halo_props/S003/halos)" << std::endl;
    std::cerr << "jk_block:: level of jackknife block. block number is jk_block^3" << std::endl;
    std::cerr << "(output_filename):: opition. output filename" << std::endl;
    std::cerr << std::endl;
    std::cerr << argv[0] << " 12.0 15.0 ./halo_props/S003/halos 1 output.dat" << std::endl;
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

  int jk_level = std::atol(argv[4]);
  if(jk_level < 1) jk_level = 1;
  const int jk_block = jk_level * jk_level * jk_level;

  std::string output_filename = "xi_halo.dat";
  if(argc == 6) output_filename = std::string(argv[5]);

  std::cout << "# input prefix " << input_prefix << std::endl;
  std::cout << "# base file " << base_file << std::endl;
  std::cout << "# output filename " << output_filename << std::endl;
  std::cout << "# Mmin, Mmax " << mvir_min << ", " << mvir_max << std::endl;
  std::cout << "# Rmin, Rmax, NR " << rmin << ", " << rmax << ", " << nr << std::endl;
  std::cout << "# log_bin " << std::boolalpha << log_bin << std::endl;
  std::cout << "# jackknife block" << jk_block << std::endl;
  // std::cout << "# FFT mesh " << nmesh << "^3" << std::endl;

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

  cor.set_halo_pm_group(pos, mvir);

  if(jk_block <= 1) {
    if constexpr(use_Landy_Szalay) cor.calc_xi_LS();
    else cor.calc_xi();
    cor.output_xi(output_filename);
  } else {
    if constexpr(use_Landy_Szalay) cor.calc_xi_jk_LS();
    else cor.calc_xi_jk();
    cor.output_xi_jk(output_filename);
  }

  return EXIT_SUCCESS;
}
