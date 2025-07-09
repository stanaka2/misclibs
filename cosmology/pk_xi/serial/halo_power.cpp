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

int main(int argc, char **argv)
{
  BaseOptions opt(argc, argv);
  std::string base_file = opt.input_prefix + ".0" + opt.h5_suffix;

  float mvir_min, mvir_max;
  mvir_min = pow(10.0, opt.mrange[0]);
  mvir_max = pow(10.0, opt.mrange[1]);

  int nk = opt.nk;
  float kmin = opt.krange[0];
  float kmax = opt.krange[1];
  bool log_bin = opt.log_bin;
  auto nmesh = opt.nmesh;

  std::cout << "# input prefix " << opt.input_prefix << std::endl;
  std::cout << "# base file " << base_file << std::endl;
  std::cout << "# output filename " << opt.output_filename << std::endl;
  std::cout << "# Mmin, Mmax " << mvir_min << ", " << mvir_max << std::endl;
  std::cout << "# kmin, kmax, Nk " << kmin << ", " << kmax << ", " << nk << std::endl;
  std::cout << "# log_bin " << std::boolalpha << opt.log_bin << std::endl;
  std::cout << "# FFT mesh " << nmesh << "^3" << std::endl;
  std::cout << std::endl;

  load_halos halos;
  halos.read_header(base_file);
  halos.print_header();
  halos.scheme = opt.p_assign;

  double lbox = halos.box_size;

  auto pos = halos.load_halo_field<float>(opt.input_prefix, opt.h5_suffix, "pos");
  auto mvir = halos.load_halo_field<float>(opt.input_prefix, opt.h5_suffix, "Mvir");
  auto clevel = halos.load_halo_field<int>(opt.input_prefix, opt.h5_suffix, "child_level");

  groupcatalog groups;
  groups.lbox = lbox;
  groups.Om = halos.Om;
  groups.Ol = halos.Ol;

  groups.select_range(mvir, mvir_min, mvir_max);
  groups.select_range(clevel, opt.clevel[0], opt.clevel[1]);

  auto grp = groups.set_base_grp(pos);

  int64_t nfft_tot = (int64_t)nmesh * (int64_t)nmesh * (int64_t)(nmesh + 2);
  std::vector<float> dens_mesh(nfft_tot, 0.0f);

  group_assign_mesh(grp, dens_mesh, nmesh, 1.0, halos.scheme);

  double dens_mean = 0.0;
#pragma omp parallel for reduction(+ : dens_mean)
  for(int64_t i = 0; i < nfft_tot; i++) dens_mean += dens_mesh[i];

  dens_mean /= (double(nmesh) * double(nmesh) * double(nmesh));

#pragma omp parallel for
  for(int64_t i = 0; i < nfft_tot; i++) dens_mesh[i] = dens_mesh[i] / dens_mean - 1.0;

  powerspec power;
  power.p = halos.scheme;
  power.lbox = lbox;
  power.nmesh = nmesh;

  power.set_kbin(kmin, kmax, nk, opt.log_bin);
  //   power.check_kbin();

  power.shotnoise_corr = !opt.no_shotnoise;
  power.set_shotnoise(nhalo_select);

#if 1
  std::vector<float> power_dens;
  std::vector<float> weight_dens;
  power.calc_power_spec(dens_mesh, power_dens, weight_dens);
  power.output_pk(power_dens, weight_dens, opt.output_filename);
#else
  std::vector<std::vector<float>> power_dens_ell;
  std::vector<float> weight_dens;
  power.calc_power_spec_ell(dens_mesh, power_dens_ell, weight_dens);
  power.output_pk_ell(power_dens_ell, weight_dens, opt.output_filename);
#endif

  std::exit(EXIT_SUCCESS);
}
