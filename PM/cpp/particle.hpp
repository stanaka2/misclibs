#pragma once

struct particle {
  uint64_t id;
  float mass;
  float xpos, ypos, zpos;
  float xvel, yvel, zvel;
  bool operator<(const particle &right) const
  {
    return (xpos == right.xpos) ? ((ypos == right.ypos) ? (zpos < right.zpos) : (ypos < right.ypos))
                                : (xpos < right.xpos);
  }
};

using vecpt = std::vector<particle>;
