#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <omp.h>

#include "field.hpp"
#include "load_halo.hpp"
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

  std::cout << "# input prefix " << opt.input_prefix << std::endl;
  std::cout << "# base file " << base_file << std::endl;
  std::cout << "# output filename " << opt.output_filename << std::endl;
  std::cout << "# Mmin, Mmax " << mvir_min << ", " << mvir_max << std::endl;
  std::cout << "# Rmin, Rmax, NR " << rmin << ", " << rmax << ", " << nr << std::endl;
  std::cout << "# log_bin " << std::boolalpha << log_bin << std::endl;
  std::cout << "# FFT mesh " << nmesh << "^3" << std::endl;
  std::cout << "# jackknife block " << jk_block << std::endl;
  std::cout << std::endl;

  load_halos halos;
  halos.read_header(base_file);
  halos.print_header();
  halos.scheme = opt.p_assign;

  double lbox(halos.box_size);
  std::vector<float> pos;
  std::vector<float> mvir;
  std::vector<int> clevel;
  halos.load_halo_pml(pos, mvir, clevel, opt.input_prefix, opt.h5_suffix);

  int64_t nmesh_tot((int64_t)nmesh * (int64_t)nmesh * (int64_t)nmesh);
  int64_t nfft_tot((int64_t)nmesh * (int64_t)nmesh * (int64_t)(nmesh + 2));

  std::vector<float> dens_mesh(nfft_tot);
  std::fill(dens_mesh.begin(), dens_mesh.end(), 0.0);

  /* halo number density filed */
  /* halo selection */
  std::vector<float> ones(mvir.size(), 0.0);
  for(size_t i = 0; i < mvir.size(); i++) {
    if(opt.clevel[0] <= clevel[i] && clevel[i] <= opt.clevel[1]) {
    if(mvir[i] > mvir_min && mvir[i] < mvir_max) ones[i] = 1.0;
    }
  }

  int64_t nhalo_select = 0;
  for(size_t i = 0; i < ones.size(); i++) {
    nhalo_select += ones[i];
  }

  halo_assign_mesh(pos, ones, dens_mesh, nmesh, lbox, halos.scheme);

  double dens_mean = 0.0;
  for(int64_t i = 0; i < nfft_tot; i++) dens_mean += dens_mesh[i];
  dens_mean /= (double(nmesh) * double(nmesh) * double(nmesh));
  for(int64_t i = 0; i < nfft_tot; i++) dens_mesh[i] = dens_mesh[i] / dens_mean - 1.0;

  correlation cor;
  cor.p = halos.scheme;
  cor.lbox = lbox;
  cor.nmesh = nmesh;
  cor.mmin = mvir_min;
  cor.mmax = mvir_max;
  cor.jk_block = jk_block;
  cor.jk_level = jk_level;

  cor.set_rbin(rmin, rmax, nr, lbox, log_bin);

  cor.shotnoise_corr = !opt.no_shotnoise;
  cor.set_shotnoise(nhalo_select);

  std::vector<float> weight;
  cor.calc_xi_ifft(dens_mesh, weight);
  cor.output_xi(opt.output_filename, weight);

  return EXIT_SUCCESS;
}
