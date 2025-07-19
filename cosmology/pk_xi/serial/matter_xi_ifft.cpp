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

  correlation cor;
  cor.p = snap.scheme;
  cor.lbox = lbox;
  cor.nmesh = nmesh;
  cor.jk_block = jk_block;
  cor.jk_level = jk_level;
  cor.shotnoise_corr = !opt.no_shotnoise_corr;

  cor.set_rbin(rmin, rmax, nr, lbox, log_bin);
  cor.set_shotnoise(np_tot);

  std::vector<float> weight;
  cor.calc_xi_ifft(dens_mesh, weight);
  cor.output_xi(opt.output_filename, weight);

  std::exit(EXIT_SUCCESS);
}
