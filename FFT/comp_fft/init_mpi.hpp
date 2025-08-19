#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
#include <mpi.h>

class mympi
{
public:
  int thistask; // thistask_z+ntasks_z*(thistask_y+ntasks_y*thistask_x)
  int thistask_x, thistask_y, thistask_z;

  int ntasks; // ntasks_x*ntasks_y*ntasks_z
  int ntasks_x, ntasks_y, ntasks_z;

  int decomp = -1;

  mympi() {}
  ~mympi() {}

  void set_rank_decomp(int _decomp)
  {

    decomp = _decomp;
    find_closest_ntasks();
    thistask_x = thistask / (ntasks_y * ntasks_z);
    thistask_y = (thistask - thistask_x * ntasks_y * ntasks_z) / ntasks_z;
    thistask_z = thistask - thistask_x * ntasks_y * ntasks_z - thistask_y * ntasks_z;

    assert(thistask == get_task(thistask_x, thistask_y, thistask_z));
  }

  int get_task_x(int tx) const
  {
    if(tx >= ntasks_x) tx -= ntasks_x;
    if(tx < 0) tx += ntasks_x;
    assert((0 <= tx) && (tx < ntasks_x));
    return tx;
  }

  int get_task_y(int ty) const
  {
    if(ty >= ntasks_y) ty -= ntasks_y;
    if(ty < 0) ty += ntasks_y;
    assert((0 <= ty) && (ty < ntasks_y));
    return ty;
  }

  int get_task_z(int tz) const
  {
    if(tz >= ntasks_z) tz -= ntasks_z;
    if(tz < 0) tz += ntasks_z;
    assert((0 <= tz) && (tz < ntasks_z));
    return tz;
  }

  int get_task(int tx, int ty, int tz) const
  {
    tx = get_task_x(tx);
    ty = get_task_y(ty);
    tz = get_task_z(tz);

    int t = -1;
    if(tx >= 0 && ty >= 0 && tz >= 0) t = tz + ntasks_z * (ty + ntasks_y * tx);
    return t;
  }

  void find_closest_ntasks()
  {
    assert(decomp >= 1 && decomp <= 3);

    if(decomp == 1) {
      ntasks_x = ntasks;
      ntasks_y = 1;
      ntasks_z = 1;

    } else if(decomp == 2) {
      int x, y;
      x = sqrt(ntasks);
      while(1) {
        y = ntasks / x;
        if(x * y == ntasks) break;
        x++;
      }

      if(y > x) {
        int tmp = x;
        x = y;
        y = tmp;
      }

      ntasks_x = x;
      ntasks_y = y;
      ntasks_z = 1;

    } else if(decomp == 3) {
      int x, y, z;
      int tmp1, tmp2;
      tmp1 = cbrt(ntasks + 0.1);

      while(ntasks % tmp1) tmp1--;

      x = tmp1;
      tmp2 = ntasks / tmp1;
      tmp1 = sqrt(tmp2 + 0.1);

      while(tmp2 % tmp1) tmp1++;

      y = tmp1;
      z = tmp2 / tmp1;

      if(z > y) {
        int tmp = z;
        z = y;
        y = tmp;
      }

      if(y > x) {
        int tmp = x;
        x = y;
        y = tmp;
      }

      if(z > y) {
        int tmp = z;
        z = y;
        y = tmp;
      }

      ntasks_x = x;
      ntasks_y = y;
      ntasks_z = z;
    }

    if(thistask == 0) {
      std::cerr << "MPI decomp : " << decomp << std::endl;
      std::cerr << "ntasks_x,y,z : " << ntasks_x << " " << ntasks_y << " " << ntasks_z << std::endl;
    }
  }
};
