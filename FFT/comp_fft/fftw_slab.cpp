// 1D slab decomposition

/*
$ mpic++ -O3 -fopenmp fftw_slab.cpp -lm -lfftw3f_mpi -lfftw3f_threads -lfftw3f -o fftw_slab
$ mpic++ -O3 -fopenmp -DDOUBLEPRECISION_FFTW fftw_slab.cpp -lm -lfftw3_mpi -lfftw3_threads -lfftw3 -o fftw_slab
$ mpiexec -n 4 ./fftw_slab 256
*/

#include <iostream>

#define FFT_DECOMP (1)
#define __USE_FFTW__

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
  tf.nz_tot_p2 = (tf.nz_tot / 2 + 1) * 2;

  ptrdiff_t tmp_nx, tmp_x_start;
  fftw_init_threads();
  fftw_mpi_init();
  fftw_plan_with_nthreads(omp_get_max_threads());
  ptrdiff_t local_comp_size =
      fftw_mpi_local_size_3d(tf.nx_tot, tf.ny_tot, tf.nz_tot / 2 + 1, MPI_COMM_WORLD, &tmp_nx, &tmp_x_start);

  tf.loc_size = 2 * local_comp_size;

  tf.nx_loc = tmp_nx;
  tf.nx_loc_start = tmp_x_start;
  tf.nx_loc_end = tf.nx_loc_start + tf.nx_loc;

  tf.ny_loc = tf.ny_tot;
  tf.ny_loc_start = 0;
  tf.ny_loc_end = tf.ny_loc_start + tf.ny_loc;

  tf.nz_loc = tf.nz_tot;
  tf.nz_loc_start = 0;
  tf.nz_loc_end = tf.nz_loc_start + tf.nz_loc;
  tf.nz_loc_p2 = (tf.nz_loc / 2 + 1) * 2;

  tf.c_nx_loc = tmp_nx;
  tf.c_ny_loc = nmesh;
  tf.c_nz_loc = nmesh / 2 + 1;
  tf.c_nx_loc_start = tmp_x_start;
  tf.c_ny_loc_start = tf.c_nz_loc_start = 0;
  tf.c_nx_loc_end = tf.c_nx_loc_start + tf.c_nx_loc;
  tf.c_ny_loc_end = nmesh;
  tf.c_nz_loc_end = nmesh / 2 + 1;

  print_fft_param(tf);

  std::vector<fft_real> mesh(tf.loc_size);
  init_dens(mesh, tf, "gaussian");
  // init_dens(mesh, tf, "tophat");
  mean_dens(mesh, tf);

  //  output_dens(mesh, tf, "fftw_slab_dens");

  /* ------ FFT R2C ------ */
  using clock = std::chrono::steady_clock;
  auto t0 = clock::now();

  fftw_complex *mesh_hat;
  mesh_hat = (fftw_complex *)mesh.data();

  fftw_init_threads();
  fftw_plan_with_nthreads(omp_get_max_threads());

  fftw_plan pf = fftw_mpi_plan_dft_r2c_3d(tf.nx_tot, tf.ny_tot, tf.nz_tot, mesh.data(), (fftw_complex *)mesh_hat,
                                          MPI_COMM_WORLD, FFTW_ESTIMATE);
  fftw_execute(pf);
  fftw_destroy_plan(pf);
  fftw_cleanup_threads();
  /* ------ FFT R2C ------ */

  auto t1 = clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cerr << "elapsed FFT forward: " << ms << " ms\n";

  int nk = nmesh;
  std::vector<double> power, weight;
  calc_pk(mesh_hat, power, weight, nk, tf);

  if(p.thistask == 0) {
    output_pk(power, weight, nk, "fftw_slab_pk.dat");
  }

  MPI_Finalize();
  return EXIT_SUCCESS;
}
