// 2D pencil decomposition

/*
$ mpic++ -O3 -fopenmp pfft_pencil.cpp -lm -lpfftf -lfftw3f_mpi -lfftw3f_threads -lfftw3f -o pfft_pencil
$ mpic++ -O3 -fopenmp -DDOUBLEPRECISION_FFTW pfft_pencil.cpp -lm -lpfft -lfftw3_mpi -lfftw3_threads -lfftw3 \
-o pfft_pencil
*/

#include <iostream>

#define FFT_DECOMP (2)
#define __USE_PFFT__

#include "init_mpi.hpp"
#include "fft.hpp"
#include "common.hpp"

int main(int argc, char **argv)
{
  const int mpi_decomp = FFT_DECOMP;
  mympi p;
  int provided;
  int required = MPI_THREAD_FUNNELED;

  MPI_Init_thread(&argc, &argv, required, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &p.thistask);
  MPI_Comm_size(MPI_COMM_WORLD, &p.ntasks);

  if(argc < 2) {
    if(p.thistask == 0) {
      std::cerr << "Usage:: " << argv[0] << " nmesh_1d" << std::endl;
    }
    MPI_Finalize();
    std::exit(EXIT_SUCCESS);
  }

  p.set_rank_decomp(mpi_decomp);

  int64_t nmesh = std::atol(argv[1]);

  fft_param tf;
  tf.p = p;
  tf.nx_tot = nmesh;
  tf.ny_tot = nmesh;
  tf.nz_tot = nmesh;
  // tf.nz_tot_p2 = tf.nz_tot + 2;
  tf.nz_tot_p2 = (tf.nz_tot / 2 + 1) * 2;

  int dim = 2;
  int node[3] = {tf.p.ntasks_x, tf.p.ntasks_y, 1};

  MPI_Comm fft_2dd_comm;
  pfft_create_procmesh(dim, MPI_COMM_WORLD, node, &fft_2dd_comm);

  int rank_dim = 3;
  ptrdiff_t howmany = 1;
  ptrdiff_t total_mesh[3], ni[3], no[3];
  ptrdiff_t rblock[3], cblock[3];
  ptrdiff_t local_rlength[3], local_rstart[3];
  ptrdiff_t local_clength[3], local_cstart[3];
  ptrdiff_t local_real_size, local_comp_size;

  total_mesh[0] = ni[0] = no[0] = tf.nx_tot;
  total_mesh[1] = ni[1] = no[1] = tf.ny_tot;
  total_mesh[2] = ni[2] = no[2] = tf.nz_tot;

  rblock[0] = rblock[1] = rblock[2] = PFFT_DEFAULT_BLOCK;
  cblock[0] = cblock[1] = cblock[2] = PFFT_DEFAULT_BLOCK;

  local_comp_size =
      pfft_local_size_many_dft_r2c(rank_dim, total_mesh, ni, no, howmany, rblock, cblock, fft_2dd_comm, PFFT_PADDED_R2C,
                                   local_rlength, local_rstart, local_clength, local_cstart);

  local_real_size = 2 * local_comp_size;

  tf.nx_loc = local_rlength[0];
  tf.nx_loc_start = local_rstart[0];
  tf.nx_loc_end = tf.nx_loc_start + tf.nx_loc;

  tf.ny_loc = local_rlength[1];
  tf.ny_loc_start = local_rstart[1];
  tf.ny_loc_end = tf.ny_loc_start + tf.ny_loc;

  tf.nz_loc_p2 = local_rlength[2];
  tf.nz_loc = tf.nz_tot;
  tf.nz_loc_start = local_rstart[2];
  tf.nz_loc_end = tf.nz_loc_start + tf.nz_loc;

  tf.loc_size = local_real_size;

  tf.c_nx_loc = local_clength[0];
  tf.c_nx_loc_start = local_cstart[0];
  tf.c_nx_loc_end = tf.c_nx_loc_start + tf.c_nx_loc;

  tf.c_ny_loc = local_clength[1];
  tf.c_ny_loc_start = local_cstart[1];
  tf.c_ny_loc_end = tf.c_ny_loc_start + tf.c_ny_loc;

  tf.c_nz_loc = local_clength[2];
  tf.c_nz_loc_start = local_cstart[2];
  tf.c_nz_loc_end = tf.c_nz_loc_start + tf.c_nz_loc;

  print_fft_param(tf);

  std::vector<fft_real> mesh(tf.loc_size);
  init_dens(mesh, tf, "gaussian");
  // init_dens(mesh, tf, "tophat");
  mean_dens(mesh, tf);

  //  output_dens(mesh, tf, "fftw_slab_dens");

  /* ------ FFT R2C ------ */
  using clock = std::chrono::steady_clock;
  auto t0 = clock::now();

  pfft_complex *mesh_hat;
  mesh_hat = (pfft_complex *)mesh.data();

  /*
  total_mesh[0] = ni[0] = no[0] = tf.nx_tot;
  total_mesh[1] = ni[1] = no[1] = tf.ny_tot;
  total_mesh[2] = ni[2] = no[2] = tf.nz_tot;
  rblock[0] = rblock[1] = rblock[2] = PFFT_DEFAULT_BLOCK;
  cblock[0] = cblock[1] = cblock[2] = PFFT_DEFAULT_BLOCK;
  rblock[0] = rblock[1] = rblock[2] = PFFT_DEFAULT_BLOCK;
  cblock[0] = cblock[1] = cblock[2] = PFFT_DEFAULT_BLOCK;
  */

  int forward_sign = -1;
  int backward_sign = 1;

  pfft_init();
  pfft_plan_with_nthreads(omp_get_max_threads());

  pfft_plan pf;
  pf = pfft_plan_many_dft_r2c(rank_dim, total_mesh, ni, no, howmany, rblock, cblock, mesh.data(),
                              (pfft_complex *)mesh_hat, fft_2dd_comm, forward_sign, PFFT_ESTIMATE | PFFT_PADDED_R2C);

  pfft_execute(pf);
  pfft_destroy_plan(pf);

  auto t1 = clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cerr << "elapsed FFT forward: " << ms << " ms\n";
  /* ------ FFT R2C ------ */

  int nk = nmesh;
  std::vector<double> power, weight;
  calc_pk(mesh_hat, power, weight, nk, tf);

  if(p.thistask == 0) {
    output_pk(power, weight, nk, "pfft_pencil_pk.dat");
  }

  MPI_Finalize();
  return EXIT_SUCCESS;
}
