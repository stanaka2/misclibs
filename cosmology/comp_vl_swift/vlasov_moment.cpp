/*
g++ -std=c++17 -O3 -fopenmp vlasov_moment.cpp -o vlasov_moment
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "run_param.hpp"
#include "vlasov_particles.hpp"
#include "moments.hpp"

int main(int argc, char **argv)
{
  if(argc != 7) {
    std::cerr << "Usage :: " << argv[0] << " <input_prefix> <suffix> <nmesh> <scheme> <type> <output_filename>"
              << std::endl;
    std::cerr << "prefix :: test-10/test-10_nbody_ptcl, test-10_nu_nbody_ptcl" << std::endl;
    std::cerr << "suffix :: _nbody, _nu_nbody" << std::endl;
    std::cerr << "nmesh :: number of mesh" << std::endl;
    std::cerr << "scheme :: NGP, CIC, TSC, PCS" << std::endl;
    std::cerr << "type :: dens, velc, sigma" << std::endl;
    std::cerr << "output_filename :: output_filename" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  std::string prefix = argv[1];
  std::string suffix = argv[2];
  int64_t nmesh = std::stol(argv[3]);
  std::string scheme = argv[4];
  std::string type = argv[5];
  std::string output_filename = argv[6];

  run_param tr;

  vlasov_particles ptcls(prefix, suffix);
  ptcls.load_ptcls(tr);

  double lbox = 1.0;
  moments<float> moments(nmesh, lbox, scheme);

  if(type == "dens") {
    auto dens = moments.calc_dens_field(ptcls.ptcls);
    moments.output_moment_field(dens, tr, output_filename);
  } else if(type == "velc") {
    auto velc = moments.calc_velc_field(ptcls.ptcls);
    moments.output_moment_field(velc, tr, output_filename);
  } else if(type == "sigma") {
    auto sigma = moments.calc_sigma_field(ptcls.ptcls);
    moments.output_moment_field(sigma, tr, output_filename);
  }

  return EXIT_SUCCESS;
}
