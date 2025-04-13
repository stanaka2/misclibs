#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <omp.h>

#include "load_halo.hpp"
#include "correlationfunc.hpp"
#include "base_opts.hpp"

class ProgOptions : public BaseOptions
{
public:
  /* default arguments */
  int jk_level = 1;
  int jk_type = 0;
  bool use_LS = false;
  int nrand_factor = 5;
  /* end arguments */

  ProgOptions() = default;
  ProgOptions(int argc, char **argv)
  {
    app.description(std::string(argv[0]) + " description");
    add_to_base_app(app);
    add_to_app(app);

    try {
      app.parse(argc, argv);
    } catch(const CLI::ParseError &e) {
      std::exit(app.exit(e));
    }
  }

protected:
  template <typename T>
  void add_to_app(T &app)
{
    app.add_flag("--use_LS", use_LS, "use Landy Szalay estimator")->capture_default_str();
    app.add_option("--jk_level", jk_level, "JK level")->capture_default_str();
    app.add_option("--jk_type", jk_type, "JK type (0: spaced, 1: random)")
        ->check(CLI::IsMember({0, 1}))
        ->capture_default_str();
    app.add_option("--nrand_factor", nrand_factor, "factor of nrand to ngrp")->capture_default_str();
  }
};

int main(int argc, char **argv)
{
  ProgOptions opt(argc, argv);

  std::string base_file = opt.input_prefix + ".0" + opt.h5_suffix;

  float mvir_min, mvir_max;
  mvir_min = pow(10.0, opt.mrange[0]);
  mvir_max = pow(10.0, opt.mrange[1]);

  int nr = opt.nr;
  float rmin = opt.rrange[0];
  float rmax = opt.rrange[1];
  bool log_bin = opt.log_bin;

  int jk_level = opt.jk_level;
  if(jk_level < 1) jk_level = 1;
  const int jk_block = jk_level * jk_level * jk_level;

  auto nmesh = opt.nmesh;

  std::cout << "# input prefix " << opt.input_prefix << std::endl;
  std::cout << "# base file " << base_file << std::endl;
  std::cout << "# output filename " << opt.output_filename << std::endl;
  std::cout << "# Mmin, Mmax " << mvir_min << ", " << mvir_max << std::endl;
  std::cout << "# Rmin, Rmax, NR " << rmin << ", " << rmax << ", " << nr << std::endl;
  std::cout << "# log_bin " << std::boolalpha << log_bin << std::endl;
  std::cout << "# FFT mesh " << nmesh << "^3" << std::endl;
  std::cout << "# jackknife block" << jk_block << std::endl;
  std::cout << "# Landy Szalay estimator" << std::boolalpha << opt.use_LS << std::endl;
  if(opt.use_LS) std::cout << "# nrand factor" << std::boolalpha << opt.nrand_factor << std::endl;
  std::cout << std::endl;

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

  halos.load_halo_pm(pos, mvir, opt.input_prefix, opt.h5_suffix);

  correlation cor;
  cor.set_rbin(rmin, rmax, nr, lbox, log_bin);
  cor.mmin = mvir_min;
  cor.mmax = mvir_max;
  cor.jk_block = jk_block;
  cor.jk_level = jk_level;
  cor.use_LS = opt.use_LS;
  cor.jk_type = opt.jk_type;
  cor.nrand_factor = opt.nrand_factor;

  auto grp = cor.set_halo_pm_group(pos, mvir);

  cor.calc_xi(grp);
  cor.output_xi(opt.output_filename);

  return EXIT_SUCCESS;
}
