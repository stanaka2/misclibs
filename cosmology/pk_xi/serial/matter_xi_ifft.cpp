#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cassert>

#include "load_ptcl.hpp"
#include "correlationfunc.hpp"

const bool log_bin = false;

int main(int argc, char **argv)
{
  if(argc < 4) {
    std::cerr << "Usage:: " << argv[0] << " gdt_snap_prefix nmesh jk_level (output_filename)" << std::endl;
    std::exit(EXIT_SUCCESS);
  }

  int nr = 100;
  float rmin, rmax; // [Mpc/h]
  rmin = 0.1;
  rmax = 150.0;

  std::string input_prefix = std::string(argv[1]);
  int nmesh = std::atol(argv[2]);

  int jk_level = std::atol(argv[3]);
  if(jk_level < 1) jk_level = 1;
  const int jk_block = jk_level * jk_level * jk_level;

  std::string output_filename = "xi_matter_ifft.dat";
  if(argc == 5) output_filename = std::string(argv[4]);

  std::cout << "# input prefix " << input_prefix << std::endl;
  std::cout << "# output filename " << output_filename << std::endl;
  std::cout << "# Rmin, Rmax, NR " << rmin << ", " << rmax << ", " << nr << std::endl;
  std::cout << "# log_bin " << std::boolalpha << log_bin << std::endl;
  std::cout << "# FFT mesh " << nmesh << "^3" << std::endl;
  std::cout << "# jackknife block " << jk_block << std::endl;

  load_ptcl<particle_pot_str> snap;
  snap.nmesh = nmesh;
  snap.scheme = 3;
  snap.read_header(input_prefix);

  double lbox(snap.h.BoxSize);
  double ptcl_mass(snap.ptcl_mass);
  int64_t nmesh_tot((int64_t)nmesh * (int64_t)nmesh * (int64_t)nmesh);
  int64_t nfft_tot((int64_t)nmesh * (int64_t)nmesh * (int64_t)(nmesh + 2));

  /* for density fluctuations */
  int type = 0;
  std::vector<float> dens_mesh(nfft_tot);
  std::fill(dens_mesh.begin(), dens_mesh.end(), 0.0);
  snap.load_gdt_and_assing(input_prefix, dens_mesh, type);

  double dens_mean = 0.0;
  for(int64_t i = 0; i < nfft_tot; i++) {
    dens_mean += dens_mesh[i];
  }
  dens_mean /= (double(nmesh) * double(nmesh) * double(nmesh));

#pragma omp parallel for
  for(int64_t i = 0; i < nfft_tot; i++) {
    dens_mesh[i] = dens_mesh[i] / dens_mean - 1.0;
  }

  // output_field(dens_mesh, nmesh, lbox, "dens_mesh");

  correlation cor;
  cor.p = snap.scheme;
  cor.lbox = lbox;
  cor.nmesh = nmesh;
  cor.jk_block = jk_block;
  cor.jk_level = jk_level;

  cor.set_rbin(rmin, rmax, nr, lbox, log_bin);

  std::vector<float> weight;
  cor.calc_xi_ifft(dens_mesh, weight);
  cor.output_xi(output_filename, weight);

  std::exit(EXIT_SUCCESS);
}
