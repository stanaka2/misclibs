#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cassert>

#include "load_ptcl.hpp"
#include "correlationfunc.hpp"
#include "base_opts.hpp"

class ProgOptions : public BaseOptions
{
public:
  /* default arguments */
  int jk_level = 1;
  int jk_type = 0;
  bool use_LS = false;
  int nrand_factor = 1;
  double sampling_rate = 0.005;
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

  int nr = opt.nr;
  float rmin = opt.rrange[0];
  float rmax = opt.rrange[1];
  bool log_bin = opt.log_bin;

  int jk_level = opt.jk_level;
  if(jk_level < 1) jk_level = 1;
  const int jk_block = jk_level * jk_level * jk_level;

  auto nmesh = opt.nmesh;
  auto sampling_rate = opt.sampling_rate;

  std::cout << "# input prefix " << opt.input_prefix << std::endl;
  std::cout << "# output filename " << opt.output_filename << std::endl;
  std::cout << "# Rmin, Rmax, NR " << rmin << ", " << rmax << ", " << nr << std::endl;
  std::cout << "# log_bin " << std::boolalpha << log_bin << std::endl;
  std::cout << "# FFT mesh " << nmesh << "^3" << std::endl;
  std::cout << "# jackknife block " << jk_block << std::endl;
  std::cout << "# Landy Szalay estimator" << std::boolalpha << opt.use_LS << std::endl;
  if(opt.use_LS) std::cout << "# nrand factor" << std::boolalpha << opt.nrand_factor << std::endl;
  std::cout << "# sampling_rate " << sampling_rate << std::endl;
  std::cout << std::endl;

  load_ptcl<particle_pot_str> snap;
  snap.read_header(opt.input_prefix);

  double lbox(snap.h.BoxSize);
  if(0.5 * lbox < rmax) {
    std::cerr << "\n###############" << std::endl;
    std::cerr << "Rmax=" << rmax << " Mpc/h is too large for boxsize=" << lbox << " Mpc/h." << std::endl;
    std::cerr << "Forced Rmax=" << 0.5 * lbox << " Mpc/h." << std::endl;
    std::cerr << "###############\n" << std::endl;
    rmax = 0.5 * lbox;
  }

  /* for density fluctuations */
  snap.load_gdt_ptcl_pos(opt.input_prefix);

  correlation cor;
  cor.p = snap.scheme;
  cor.lbox = lbox;
  cor.nmesh = nmesh;
  cor.jk_block = jk_block;
  cor.jk_level = jk_level;

  cor.set_rbin(rmin, rmax, nr, lbox, log_bin);

  auto grp = cor.set_ptcl_pos_group(snap.pdata, sampling_rate);
  cor.calc_xi(grp);
  cor.output_xi(opt.output_filename);

  std::exit(EXIT_SUCCESS);
}
