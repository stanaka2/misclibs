#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <omp.h>

#include "load_halo.hpp"
#include "group.hpp"
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

  opt.print_args();

  load_halos halos;
  halos.read_header(base_file);
  halos.print_header();

  double lbox(halos.box_size);
  float ascale(halos.a);

  if(0.5 * lbox < rmax) {
    std::cerr << "\n###############" << std::endl;
    std::cerr << "Rmax=" << rmax << " Mpc/h is too large for boxsize=" << lbox << " Mpc/h." << std::endl;
    std::cerr << "Forced Rmax=" << 0.5 * lbox << " Mpc/h." << std::endl;
    std::cerr << "###############\n" << std::endl;
    rmax = 0.5 * lbox;
  }

  groupcatalog groups;
  groups.lbox = lbox;
  groups.Om = halos.Om;
  groups.Ol = halos.Ol;

  /* set selection halo index */
  auto mvir = halos.load_halo_field<float>(opt.input_prefix, opt.h5_suffix, "Mvir");
  auto clevel = halos.load_halo_field<int>(opt.input_prefix, opt.h5_suffix, "child_level");
  groups.select_range(mvir, mvir_min, mvir_max);
  groups.select_range(clevel, opt.clevel[0], opt.clevel[1]);

  auto pos = halos.load_halo_field<float>(opt.input_prefix, opt.h5_suffix, "pos");
  auto grp = groups.set_base_grp(pos);

  if(opt.do_RSD) {
    auto vel = halos.load_halo_field<float>(opt.input_prefix, opt.h5_suffix, "vel");
    groups.apply_RSD_shift(vel, ascale, opt.los_axis, grp);
  }

  if(opt.do_Gred) {
    auto pot = halos.load_halo_field<float>(opt.input_prefix, opt.h5_suffix, "pot_total");
    groups.apply_Gred_shift(pot, ascale, opt.los_axis, grp);
  }

  correlation cor;
  cor.set_rbin(rmin, rmax, nr, lbox, log_bin);
  cor.jk_block = jk_block;
  cor.jk_level = jk_level;
  cor.jk_type = opt.jk_type;
  cor.estimator = opt.estimator;
  cor.nrand_factor = opt.nrand_factor;

  cor.calc_xi(grp);
  cor.output_xi(opt.output_filename);

  return EXIT_SUCCESS;
}
