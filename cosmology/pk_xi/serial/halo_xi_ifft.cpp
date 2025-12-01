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

int main(int argc, char **argv)
{
  BaseOptions opt(argc, argv);

  std::string base_file = opt.input_prefix + ".0" + opt.h5_suffix;

  float mvir_min, mvir_max;
  mvir_min = pow(10.0, opt.mrange[0]);
  mvir_max = pow(10.0, opt.mrange[1]);

  int nr = opt.nr;
  float rmin = opt.rrange[0];
  float rmax = opt.rrange[1];
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

  /* set selection halo index */
  auto mvir = halos.load_halo_field<float>(opt.input_prefix, opt.h5_suffix, "Mvir");
  auto clevel = halos.load_halo_field<int>(opt.input_prefix, opt.h5_suffix, cb_label);
  groups.select_range(mvir, mvir_min, mvir_max);
  groups.select_range(clevel, opt.clevel[0], opt.clevel[1]);

  auto pos = halos.load_halo_field<float>(opt.input_prefix, opt.h5_suffix, pos_label);
  auto grp = groups.set_base_grp(pos);

  if(opt.do_RSD) {
    auto vel = halos.load_halo_field<float>(opt.input_prefix, opt.h5_suffix, vel_label);
    groups.apply_RSD_shift(vel, ascale, opt.los_axis, grp);
  }

  if(opt.do_Gred) {
    auto pot = halos.load_halo_field<float>(opt.input_prefix, opt.h5_suffix, pot_label);
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
  cor.los = (opt.los_axis == "x") ? 0 : (opt.los_axis == "y") ? 1 : 2;

  cor.set_rbin(rmin, rmax, nr, lbox, log_bin);

  cor.shotnoise_corr = !opt.no_shotnoise_corr;
  cor.set_shotnoise(nhalo_select);

  std::vector<float> weight;
  cor.calc_xi_ifft(dens_mesh, weight);
  cor.output_xi(opt.output_filename, weight);

  return EXIT_SUCCESS;
}
