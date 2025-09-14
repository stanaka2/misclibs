#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cassert>

#include "load_ptcl.hpp"
#include "powerspec.hpp"
#include "base_opts.hpp"

int main(int argc, char **argv)
{
  BaseOptions opt(argc, argv);

  int nk = opt.nk;
  float kmin = opt.krange[0];
  float kmax = opt.krange[1];
  bool log_bin = opt.log_bin;
  auto nmesh = opt.nmesh;

  opt.print_args();

  load_ptcl<particle_str> snap;
  snap.nmesh = nmesh;
  snap.scheme = opt.p_assign;
  snap.do_RSD = opt.do_RSD;
  snap.do_Gred = opt.do_Gred;
  snap.read_header(opt.input_prefix);

  int64_t np_tot(snap.npart_tot);
  double lbox(snap.h.BoxSize);
  int64_t nmesh_tot((int64_t)nmesh * (int64_t)nmesh * (int64_t)nmesh);
  int64_t nfft_tot((int64_t)nmesh * (int64_t)nmesh * (int64_t)(nmesh + 2));

  /* for density fluctuations */
  std::vector<float> dens_mesh(nfft_tot);
  std::fill(dens_mesh.begin(), dens_mesh.end(), 0.0);

  /* for density fluctuations */
  if(opt.do_RSD || opt.do_Gred) {
    snap.load_gdt_and_shift_assing(opt.input_prefix, dens_mesh);
  } else {
    snap.load_gdt_and_assing(opt.input_prefix, dens_mesh);
  }

  snap.free_pdata();

  normalize_mesh(dens_mesh, nmesh);
  // output_field(dens_mesh, nmesh, lbox, "dens_mesh");

  powerspec power;
  power.p = snap.scheme;
  power.lbox = lbox;
  power.nmesh = nmesh;
  power.shotnoise_corr = !opt.no_shotnoise_corr;

  //   power.check_kbin();
  power.set_kbin(kmin, kmax, nk, log_bin);
  power.set_shotnoise(np_tot);

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
