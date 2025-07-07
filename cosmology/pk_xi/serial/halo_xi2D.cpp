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
  std::string mode = "spsp";
  bool half_angle = false;
  int jk_level = 1;
  int jk_type = 0;
  bool use_LS = false;
  int nrand_factor = 1;
  /* end arguments */

  bool mode_spsp = true; // default = spsp

  ProgOptions() = default;
  ProgOptions(int argc, char **argv)
  {
    app.description(std::string(argv[0]) + " description");
    add_to_base_app(app);
    add_to_app(app);

    try {
      app.parse(argc, argv);
      mode_spsp = (mode == "spsp");
    } catch(const CLI::ParseError &e) {
      std::exit(app.exit(e));
    }
  }

protected:
  template <typename T>
  void add_to_app(T &app)
  {
    app.add_option("--mode", mode, "pair-counting mode")
        ->check(CLI::IsMember({"spsp", "smu"}))
        ->default_val("spsp")
        ->capture_default_str();
    app.add_flag("--half_angle", half_angle,
                 "If set, calculate only the upper half-range (0<mu<1 or 0<s_para<Rmax); "
                 "if not set, calculate the full range (-1<mu<1 or -Rmax<s_para<Rmax)")
        ->capture_default_str();
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
  // bool log_bin = opt.log_bin;
  bool log_bin = false;

  auto mode_spsp = opt.mode_spsp;
  auto half_angle = opt.half_angle;

  int nr1, r1min, r1max;
  int nr2, r2min, r2max;

  if(opt.mode_spsp) {
    // spsp mode
    if(!half_angle) {
      // s_perp
      nr1 = nr;
      r1min = rmin;
      r1max = rmax;
      // s_perp
      nr2 = 2 * nr;
      r2min = -rmax;
      r2max = rmax;

    } else {
      // s_perp
      nr1 = nr;
      r1min = rmin;
      r1max = rmax;
      // s_para
      nr2 = nr;
      r2min = 0.0;
      r2max = rmax;
    }

  } else {
    // smu mode
    if(!half_angle) {
      // s
      nr1 = nr;
      r1min = rmin;
      r1max = rmax;
      // mu
      nr2 = 2 * nr;
      r2min = -1.0;
      r2max = 1.0;

    } else {
      // s
      nr1 = nr;
      r1min = rmin;
      r1max = rmax;
      // mu
      nr2 = nr;
      r2min = 0.0;
      r2max = 1.0;
    }
  }

  int jk_level = opt.jk_level;
  if(jk_level < 1) jk_level = 1;
  const int jk_block = jk_level * jk_level * jk_level;

  auto nmesh = opt.nmesh;

  std::cout << "# input prefix " << opt.input_prefix << std::endl;
  std::cout << "# base file " << base_file << std::endl;
  std::cout << "# output filename " << opt.output_filename << std::endl;
  std::cout << "# Mmin, Mmax " << mvir_min << ", " << mvir_max << std::endl;
  std::cout << "# Rmin, Rmax, NR " << rmin << ", " << rmax << ", " << nr << std::endl;
  std::cout << "# mode_spsp " << mode_spsp << std::endl;
  std::cout << "# half_angle " << half_angle << std::endl;
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
  std::vector<int> clevel;
  halos.load_halo_pml(pos, mvir, clevel, opt.input_prefix, opt.h5_suffix);

  correlation cor;

  if(mode_spsp) {
    cor.set_spspbin(r1min, r1max, nr1, r2min, r2max, nr2, lbox);
  } else {
    cor.set_smubin(r1min, r1max, nr1, r2min, r2max, nr2, lbox);
  }

  cor.mmin = mvir_min;
  cor.mmax = mvir_max;
  cor.jk_block = jk_block;
  cor.jk_level = jk_level;
  cor.use_LS = opt.use_LS;
  cor.jk_type = opt.jk_type;
  cor.nrand_factor = opt.nrand_factor;

  auto grp = cor.set_halo_pml_group(pos, mvir, clevel, opt.clevel[0], opt.clevel[1]);

  if(mode_spsp) {
    cor.calc_xi_spsp(grp);
    cor.output_xi_spsp(opt.output_filename);
  } else {
    cor.calc_xi_smu(grp);
    cor.output_xi_smu(opt.output_filename);
  }
  return EXIT_SUCCESS;
}
