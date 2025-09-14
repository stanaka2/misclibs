#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <omp.h>

#include "field.hpp"
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
  halos.scheme = opt.p_assign;
  halos.read_header(base_file);
  halos.print_header();

  double lbox(halos.box_size);
  float ascale(halos.a);

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

  int64_t nfft_tot = (int64_t)nmesh * (int64_t)nmesh * (int64_t)(nmesh + 2);
  std::vector<float> dens_mesh(nfft_tot, 0.0f);
  group_assign_mesh(grp, dens_mesh, nmesh, halos.scheme);

  normalize_mesh(dens_mesh, nmesh);
  // output_field(dens_mesh, nmesh, lbox, "halo_dens_mesh");

  auto nhalo_select = grp.size();

  correlation cor;
  cor.p = halos.scheme;
  cor.lbox = lbox;
  cor.nmesh = nmesh;
  cor.njk = jk_block;
  cor.jk_level = jk_level;
  cor.los = (opt.los_axis == "x") ? 0 : (opt.los_axis == "y") ? 1 : 2;

  cor.set_rbin(rmin, rmax, nr, lbox, log_bin);

  cor.shotnoise_corr = !opt.no_shotnoise_corr;
  cor.set_shotnoise(nhalo_select);

  std::vector<float> weight;
  cor.calc_xi_ifft(dens_mesh, weight);
  cor.output_xi(opt.output_filename, weight);

  return EXIT_SUCCESS;
}
