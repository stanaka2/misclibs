#pragma once

#include <iomanip>
#include <cassert>
#include <stdint.h>

typedef int64_t INT;
typedef uint64_t UINT;

typedef float nb_float;
// typedef double nb_float;

#include "cosmology.hpp"

struct meshes {
  int x_total, y_total, z_total;
  int x_local, y_local, z_local;
  int x_local_start, x_local_end;
  int y_local_start, y_local_end;
  int z_local_start, z_local_end;
  int64_t local_size;
};

struct run_param {
  int cosmology_flag;
  int canonical_flag; /* 0:peculiar velocity, 1:canonical momentum */

  int mpi_rank, mpi_nproc;
  int rank_x, rank_y, rank_z;
  int nnode_x, nnode_y, nnode_z;

  /* Min. and Max. of the entire and local physical space */
  float xmin, xmax, xmin_local, xmax_local, delta_x;
  float ymin, ymax, ymin_local, ymax_local, delta_y;
  float zmin, zmax, zmin_local, zmax_local, delta_z;

  struct meshes nmesh;
  double lunit, munit, tunit;

  int num_df, idf;

  float tnow;
  float tend;
  float dtime;
  int step, global_step;

  UINT npart, npart_skirt, npart_total, npart_max;
  struct cosmology cosm;

  double anow, znow, hnow;
  double lbox;
  double vunit;
  int nfiles;
};

#define SQR(x) ((x) * (x))
