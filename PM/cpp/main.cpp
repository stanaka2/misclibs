#include <iostream>
#include <fstream>
#include <sstream>

#include "mass_assign.hpp"
#include "particle.hpp"

int main(int argc, char **argv)
{

  std::string filename = argv[1];
  const int nside = std::atol(argv[2]);
  vecpt ptcl;

  std::ifstream infile(filename);
  std::string line;

  while(std::getline(infile, line)) {
    std::istringstream iss(line);
    particle p;
    if(iss >> p.xpos >> p.ypos >> p.zpos) {
      p.xpos /= 20.0;
      p.ypos /= 20.0;
      p.zpos /= 20.0;

      ptcl.push_back(p);
    } else {
      std::cerr << "Error: Invalid line in input file: " << line << std::endl;
    }
  }

  // auto mesh = ngp<float>(ptcl, nside);
  // auto mesh = cic<float>(ptcl, nside);
  auto mesh = tsc<float>(ptcl, nside);

  for(int i = 0; i < 10; i++) {
    std::cout << mesh[i] << std::endl;
  }
  std::cout << std::endl;
  for(int i = 0; i < 10; i++) {
    std::cout << mesh[nside * nside * nside - 10 + i] << std::endl;
  }

  std::cout << std::endl;
  std::cout << "min " << *std::min_element(mesh.begin(), mesh.end()) << std::endl;
  std::cout << "max " << *std::max_element(mesh.begin(), mesh.end()) << std::endl;

  return 0;
}
