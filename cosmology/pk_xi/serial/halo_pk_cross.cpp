#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cassert>

#include "field.hpp"
#include "load_halo.hpp"
#include "group.hpp"
#include "powerspec.hpp"
#include "base_opts.hpp"

class ProgOptions : public BaseOptions
{
public:
  /* default arguments */
  std::vector<float> mrange2 = {1.0f, 20.0f};
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
    app.add_option("--mrange2", mrange2, "minimum halo mass (log10)")->expected(2)->capture_default_str();
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

  int nk = opt.nk;
  float kmin = opt.krange[0];
  float kmax = opt.krange[1];
  bool log_bin = opt.log_bin;

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
  groups2.select_range(clevel, opt.clevel[0], opt.clevel[1]);

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

  int64_t nfft_tot = (int64_t)nmesh * (int64_t)nmesh * (int64_t)(nmesh + 2);
  std::vector<float> dens_mesh1(nfft_tot, 0.0f);
  std::vector<float> dens_mesh2(nfft_tot, 0.0f);
  group_assign_mesh(grp, dens_mesh1, nmesh, halos.scheme);
  group_assign_mesh(grp2, dens_mesh2, nmesh, halos.scheme);

  normalize_mesh(dens_mesh1, nmesh);
  normalize_mesh(dens_mesh2, nmesh);

  // output_field(dens_mesh, nmesh, lbox, "halo_dens_mesh");

  //  auto nhalo_select = grp.size();

  powerspec power;
  power.p = halos.scheme;
  power.lbox = lbox;
  power.nmesh = nmesh;
  power.shotnoise_corr = !opt.no_shotnoise_corr;

  power.set_kbin(kmin, kmax, nk, opt.log_bin);
  //   power.check_kbin();
  // power.set_shotnoise(nhalo_select);

  std::vector<float> power_dens;
  std::vector<float> weight_dens;
  power.calc_power_spec(dens_mesh1, dens_mesh2, power_dens, weight_dens);
  power.output_pk(power_dens, weight_dens, opt.output_filename);

  std::exit(EXIT_SUCCESS);
}
