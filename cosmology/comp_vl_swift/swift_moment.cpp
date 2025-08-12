/*
g++ -std=c++17 -O3 -fopenmp swift_moment.cpp  -I/home/stanaka/software/hdf5/include -L/home/stanaka/software/hdf5/lib
-lhdf5 -lz -o swift_moment
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "run_param.hpp"
#include "swift_particles.hpp"
#include "moments.hpp"

int main(int argc, char **argv)
{
  if(argc != 7) {
    std::cerr << "Usage :: " << argv[0] << " <input_file> <matter_type> <nmesh> <scheme> <type> <output_filename>"
              << std::endl;
    std::cerr
        << "input :: /import/cage1/wzzhang/vlasov_test/n768_nu04_l200_vlasov_same_mass/output_neu768/snapshot_0006.hdf5"
        << std::endl;
    std::cerr << "matter_type :: cdm, nu" << std::endl;
    std::cerr << "nmesh :: number of mesh" << std::endl;
    std::cerr << "scheme :: NGP, CIC, TSC, PCS" << std::endl;
    std::cerr << "type :: dens, velc, sigma" << std::endl;
    std::cerr << "output_filename :: output_filename" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  std::string input_file = argv[1];
  std::string matter_type = argv[2];
  int64_t nmesh = std::stol(argv[3]);
  std::string scheme = argv[4];
  std::string type = argv[5];
  std::string output_filename = argv[6];

  run_param tr;

  swift_particles ptcls(input_file, matter_type);
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
