// no-mpi decomposition

/*
$ mpic++ -O3 -fopenmp fftw_no_mpi.cpp -lm -lfftw3f_threads -lfftw3f -o fftw_no_mpi
$ mpic++ -O3 -fopenmp -DDOUBLEPRECISION_FFTW fftw_no_mpi.cpp -lm -lfftw3_threads -lfftw3 -o fftw_no_mpi
$ ./fftw_no_mpi 256
*/

#include <iostream>

#define FFT_DECOMP (1)
#define __USE_FFTW__

#include "init_mpi.hpp"
#include "fft.hpp"
#include "common.hpp"

int main(int argc, char **argv)
{
  mympi p;
  p.ntasks = p.ntasks_x = p.ntasks_y = p.ntasks_z = 1;
  p.thistask = p.thistask_x = p.thistask_y = p.thistask_z = 0;

  p.decomp = -1;

  if(argc < 2) {
    if(p.thistask == 0) {
      std::cerr << "Usage:: " << argv[0] << " nmesh_1d" << std::endl;
    }
    MPI_Finalize();
    std::exit(EXIT_SUCCESS);
  }

  int64_t nmesh = std::atol(argv[1]);

  fft_param tf;
  tf.p = p;
  tf.nx_tot = tf.nx_loc = nmesh;
  tf.ny_tot = tf.ny_loc = nmesh;
  tf.nz_tot = tf.nz_loc = nmesh;
  tf.nx_loc_start = tf.ny_loc_start = tf.nz_loc_start = 0;
  tf.nx_loc_end = tf.ny_loc_end = tf.nz_loc_end = nmesh;

  // tf.nz_loc_p2 = tf.nz_loc + 2;
  // tf.nz_tot_p2 = tf.nz_tot + 2;
  tf.nz_loc_p2 = (tf.nz_loc / 2 + 1) * 2;
  tf.nz_tot_p2 = (tf.nz_tot / 2 + 1) * 2;

  tf.c_nx_loc = tf.c_ny_loc = nmesh;
  tf.c_nz_loc = nmesh / 2 + 1;
  tf.c_nx_loc_start = tf.c_ny_loc_start = tf.c_nz_loc_start = 0;
  tf.c_nx_loc_end = tf.c_ny_loc_end = nmesh;
  tf.c_nz_loc_end = nmesh / 2 + 1;

  ptrdiff_t local_size = tf.nx_loc * tf.ny_loc * tf.nz_loc_p2;
  tf.loc_size = local_size;

  print_fft_param(tf);

  std::vector<fft_real> mesh(tf.loc_size);
  init_dens(mesh, tf, "gaussian");
  // init_dens(mesh, tf, "tophat");
  mean_dens(mesh, tf);

  //  output_dens(mesh, tf, "no_mpi_dens");

  /* ------ FFT R2C ------ */
  using clock = std::chrono::steady_clock;
  auto t0 = clock::now();

  fftw_complex *mesh_hat;
  mesh_hat = (fftw_complex *)mesh.data();
  fftw_init_threads();
  fftw_plan_with_nthreads(omp_get_max_threads());

  fftw_plan pf =
      fftw_plan_dft_r2c_3d(tf.nx_tot, tf.ny_tot, tf.nz_tot, mesh.data(), (fftw_complex *)mesh_hat, FFTW_ESTIMATE);
  fftw_execute(pf);
  fftw_destroy_plan(pf);
  fftw_cleanup_threads();

  auto t1 = clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cerr << "elapsed FFT forward: " << ms << " ms\n";
  /* ------ FFT R2C ------ */

  //  output_dens_hat(mesh_hat, tf, "no_mpi_hat_dens");

  int nk = nmesh;
  std::vector<double> power, weight;
  calc_pk(mesh_hat, power, weight, nk, tf);
  output_pk(power, weight, nk, "no_mpi_pk.dat");

  return EXIT_SUCCESS;
}
