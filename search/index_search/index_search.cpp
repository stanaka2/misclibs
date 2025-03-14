#include <cstdlib>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cassert>

struct particle {
  float x, y, z;
  int id;
};

int main(int argc, char **argv)
{
  uint64_t npart = 1024 * 1024 * 32;
  std::vector<particle> ptcls(npart);

  int seed = 100;
  std::mt19937 mt(seed);
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  for(size_t i = 0; i < ptcls.size(); i++) {
    ptcls[i].x = dist(mt);
    ptcls[i].y = dist(mt);
    ptcls[i].z = dist(mt);
    ptcls[i].id = static_cast<int>(i);
  }

  return 0;
}
