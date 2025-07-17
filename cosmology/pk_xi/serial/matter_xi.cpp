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
    app.add_option("--jk_level", jk_level, "JK level")->capture_default_str();
    app.add_option("--jk_type", jk_type, "JK type (0: spaced, 1: random)")
        ->check(CLI::IsMember({0, 1}))
        ->capture_default_str();
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

  auto sampling_rate = opt.sampling_rate;

  opt.print_args();

  load_ptcl<particle_str> snap;
  snap.sampling_rate = sampling_rate;
  snap.do_RSD = opt.do_RSD;
  snap.do_Gred = opt.do_Gred;
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
  if(opt.do_RSD || opt.do_Gred) {
    snap.load_gdt_and_shift_ptcl(opt.input_prefix);
  } else {
    snap.load_gdt_ptcl(opt.input_prefix);
  }

  groupcatalog groups;
  groups.lbox = lbox;

  // Note:
  // Particle sampling is already performed during particle loading.
  // Alternatively, sampling can be done during group creation.
  // Make sure that sampling is not applied twice.
  auto grp = groups.set_base_grp(snap.pdata);

  correlation cor;
  cor.jk_block = jk_block;
  cor.jk_level = jk_level;
  cor.jk_type = opt.jk_type;
  cor.estimator = opt.estimator;
  cor.nrand_factor = opt.nrand_factor;

  cor.set_rbin(rmin, rmax, nr, lbox, log_bin);
  cor.calc_xi(grp);
  cor.output_xi(opt.output_filename);

  std::exit(EXIT_SUCCESS);
}
