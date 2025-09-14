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
  auto clevel = halos.load_halo_field<int>(opt.input_prefix, opt.h5_suffix, cl_label);
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

  correlation cor;
  cor.set_rbin(rmin, rmax, nr, lbox, log_bin);
  cor.los = (opt.los_axis == "x") ? 0 : (opt.los_axis == "y") ? 1 : 2;

  cor.set_cor_estimator(opt.estimator);
  cor.set_cor_mode("r");
  cor.nrand_factor = opt.nrand_factor;

  cor.calc_xi(grp);
  cor.output_xi(opt.output_filename);

  return EXIT_SUCCESS;
}
