// 3D brick or 2D pencil or 1D slab decomposition

/*
$ mpic++ -O3 -fopenmp heffte_brick.cpp -I/home/stanaka/software/heffte/include -L/home/stanaka/software/heffte/lib \
-lheffte -lfftw3_omp -lfftw3f_omp -lfftw3 -lfftw3f -lm -o heffte_brick
$ mpic++ -O3 -fopenmp -DDOUBLEPRECISION_FFTW heffte_brick.cpp -I/home/stanaka/software/heffte/include \
-L/home/stanaka/software/heffte/lib -lheffte -lfftw3_omp -lfftw3f_omp -lfftw3 -lfftw3f -lm -o heffte_brick
*/

#include <iostream>

#define FFT_DECOMP (3)
#define __USE_HEFFTE__

// true: 3D brick to 2D pencil,  false: 3D brick to 1D slab
constexpr bool use_pencils = true;

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

  // The pad area is not necessary in heffte. This is because R2C and C2R do not support in-place memory.
  tf.nz_tot_p2 = tf.nz_tot;

  // std::array<int, 3> order = {0, 1, 2}; // default,
  std::array<int, 3> order = {2, 1, 0};
  heffte::box3d<> rspace({0, 0, 0}, {(int)tf.nx_tot - 1, (int)tf.ny_tot - 1, (int)tf.nz_tot - 1}, order);
  heffte::box3d<> cspace({0, 0, 0}, {(int)tf.nx_tot - 1, (int)tf.ny_tot - 1, (int)tf.nz_tot / 2}, order);

  // code rank index (z-y-x order) heffte index (x-y-z order)
  // std::array<int, 3> proc_grid = {p.ntasks_x, p.ntasks_y, p.ntasks_z};
  std::array<int, 3> proc_grid = {p.ntasks_z, p.ntasks_y, p.ntasks_x};

  auto inbox = heffte::split_world(rspace, proc_grid)[p.thistask];
  auto outbox = heffte::split_world(cspace, proc_grid)[p.thistask];

  tf.nx_loc = inbox.size[0];
  tf.ny_loc = inbox.size[1];
  tf.nz_loc = inbox.size[2];
  tf.nx_loc_start = inbox.low[0];
  tf.ny_loc_start = inbox.low[1];
  tf.nz_loc_start = inbox.low[2];
  tf.nx_loc_end = tf.nx_loc_start + tf.nx_loc;
  tf.ny_loc_end = tf.ny_loc_start + tf.ny_loc;
  tf.nz_loc_end = tf.nz_loc_start + tf.nz_loc;

  tf.nz_loc_p2 = tf.nz_loc; // not +2
  tf.loc_size = tf.nx_loc * tf.ny_loc * tf.nz_loc;

  tf.c_nx_loc = outbox.size[0];
  tf.c_ny_loc = outbox.size[1];
  tf.c_nz_loc = outbox.size[2];
  tf.c_nx_loc_start = outbox.low[0];
  tf.c_ny_loc_start = outbox.low[1];
  tf.c_nz_loc_start = outbox.low[2];
  tf.c_nx_loc_end = tf.c_nx_loc_start + tf.c_nx_loc;
  tf.c_ny_loc_end = tf.c_ny_loc_start + tf.c_ny_loc;
  tf.c_nz_loc_end = tf.c_nz_loc_start + tf.c_nz_loc;

  print_fft_param(tf);

  std::vector<fft_real> mesh(tf.loc_size);
  init_dens(mesh, tf, "gaussian");
  // init_dens(mesh, tf, "tophat");
  mean_dens(mesh, tf);

  //  output_dens(mesh, tf, "heffte_dens");

  /* ------ FFT R2C ------ */
  using clock = std::chrono::steady_clock;
  auto t0 = clock::now();

  // In-place support is only available for C2C
  std::vector<std::complex<fft_real>> mesh_hat(tf.c_nx_loc * tf.c_ny_loc * tf.c_nz_loc);

  {
    // enable fftw threads
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());

    heffte::plan_options opts = heffte::default_options<heffte::backend::fftw>();
    opts.use_pencils = use_pencils;
    // opts.algorithm = heffte::reshape_algorithm::alltoallv;
    opts.algorithm = heffte::reshape_algorithm::p2p;
    // opts.algorithm = heffte::reshape_algorithm::p2p_plined;

    // create plan
    const int r2c_direction = 2; // 2 is z-axis (x:0, y:0). Nz to Nz/2 + 1
    heffte::fft3d_r2c<heffte::backend::fftw> fft_r2c(inbox, outbox, r2c_direction, MPI_COMM_WORLD, opts);

    fft_r2c.forward(mesh.data(), mesh_hat.data());

    /* ------ FFT R2C ------ */
  } //  destructor

  auto t1 = clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cerr << "elapsed FFT forward: " << ms << " ms\n";

  mesh.clear();
  mesh.shrink_to_fit();

  //  output_dens_hat(mesh_hat, tf, "heffte_hat_dens");

  int nk = nmesh;
  std::vector<double> power, weight;
  calc_pk(mesh_hat, power, weight, nk, tf);

  if(p.thistask == 0) {
    if(use_pencils) output_pk(power, weight, nk, "heffte_brick_to_pencil_pk.dat");
    else output_pk(power, weight, nk, "heffte_brick_to_slab_pk.dat");
  }

  MPI_Finalize();
  return EXIT_SUCCESS;
}
