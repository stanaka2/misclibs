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
  std::string mode = "smu";
  bool half_angle = false;
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
    app.add_option("--mode", mode, "pair-counting mode")->check(CLI::IsMember({"spsp", "smu"}))->capture_default_str();
    app.add_flag("--half_angle", half_angle,
                 "If set, calculate only the upper half-range (0<mu<1 or 0<s_para<Rmax); "
                 "if not set, calculate the full range (-1<mu<1 or -Rmax<s_para<Rmax)")
        ->capture_default_str();

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
  // bool log_bin = opt.log_bin;
  bool log_bin = false;

  int nr1, r1min, r1max;
  int nr2, r2min, r2max;

  if(opt.mode == "spsp") {
    // spsp mode
    if(!opt.half_angle) {
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
    if(!opt.half_angle) {
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
  cor.nrand_factor = opt.nrand_factor;
  cor.los = (opt.los_axis == "x") ? 0 : (opt.los_axis == "y") ? 1 : 2;

  cor.set_cor_estimator(opt.estimator);
  cor.set_cor_mode(opt.mode);

  cor.set_rbin2D(r1min, r1max, nr1, r2min, r2max, nr2, lbox);
  cor.calc_xi(grp, grp2);
  cor.output_xi2D(opt.output_filename);

  return EXIT_SUCCESS;
}
