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
  std::vector<float> mrange2 = {1.0f, 20.0f};
  std::vector<int> clevel2 = {0, 100};
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
    app.add_option("--mrange2", mrange2, "halo2 mass range (log10)")->expected(2)->capture_default_str();
    app.add_option("--clevel2", clevel2, "halo2 child level")->expected(2)->capture_default_str();
  }
};

int main(int argc, char **argv)
{
  ProgOptions opt(argc, argv);

  std::string base_file = opt.input_prefix + ".0" + opt.h5_suffix;

  float mvir_min, mvir_max;
  mvir_min = pow(10.0, opt.mrange[0]);
  mvir_max = pow(10.0, opt.mrange[1]);

  float mvir_min2, mvir_max2;
  mvir_min2 = pow(10.0, opt.mrange2[0]);
  mvir_max2 = pow(10.0, opt.mrange2[1]);

  int nr = opt.nr;
  float rmin = opt.rrange[0];
  float rmax = opt.rrange[1];
  bool log_bin = opt.log_bin;

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

  groupcatalog groups2;
  groups2.lbox = lbox;
  groups2.Om = halos.Om;
  groups2.Ol = halos.Ol;

  /* set selection halo index */
  auto mvir = halos.load_halo_field<float>(opt.input_prefix, opt.h5_suffix, "Mvir");
  auto clevel = halos.load_halo_field<int>(opt.input_prefix, opt.h5_suffix, cl_label);
  groups.select_range(mvir, mvir_min, mvir_max);
  groups.select_range(clevel, opt.clevel[0], opt.clevel[1]);

  groups2.select_range(mvir, mvir_min2, mvir_max2);
  groups2.select_range(clevel, opt.clevel2[0], opt.clevel2[1]);

  auto pos = halos.load_halo_field<float>(opt.input_prefix, opt.h5_suffix, pos_label);
  auto grp = groups.set_base_grp(pos);
  auto grp2 = groups2.set_base_grp(pos);

  if(opt.do_RSD) {
    auto vel = halos.load_halo_field<float>(opt.input_prefix, opt.h5_suffix, vel_label);
    groups.apply_RSD_shift(vel, ascale, opt.los_axis, grp);
    groups2.apply_RSD_shift(vel, ascale, opt.los_axis, grp2);
  }

  if(opt.do_Gred) {
    auto pot = halos.load_halo_field<float>(opt.input_prefix, opt.h5_suffix, pot_label);
    groups.apply_Gred_shift(pot, ascale, opt.los_axis, grp);
    groups2.apply_Gred_shift(pot, ascale, opt.los_axis, grp2);
  }

  correlation cor;
  cor.set_rbin(rmin, rmax, nr, lbox, log_bin);
  cor.los = (opt.los_axis == "x") ? 0 : (opt.los_axis == "y") ? 1 : 2;

  cor.set_cor_estimator(opt.estimator);
  cor.set_cor_mode("r");
  cor.nrand_factor = opt.nrand_factor;

  cor.calc_xi(grp, grp2);
  cor.output_xi(opt.output_filename);

  return EXIT_SUCCESS;
}
