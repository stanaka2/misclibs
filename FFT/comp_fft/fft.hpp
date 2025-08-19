#pragma once

#include <omp.h>

#define c_re(c) ((c)[0])
#define c_im(c) ((c)[1])

template <typename T>
static inline T SQR(T x)
{
  return x * x;
}

class fft_param
{
public:
  class mympi p;

  int nx_tot, ny_tot, nz_tot;
  int nz_tot_p2;
  int64_t tot_size;

  int nx_loc, nx_loc_start, nx_loc_end;
  int ny_loc, ny_loc_start, ny_loc_end;
  int nz_loc, nz_loc_start, nz_loc_end;
  int64_t nz_loc_p2;
  int64_t loc_size;

  int c_nx_loc, c_nx_loc_start, c_nx_loc_end;
  int c_ny_loc, c_ny_loc_start, c_ny_loc_end;
  int c_nz_loc, c_nz_loc_start, c_nz_loc_end;
};

void print_fft_param(fft_param &tf)
{
  std::cout << "rank " << tf.p.thistask << std::endl;
  std::cout << "rank_{x,y,z} " << tf.p.thistask_x << " " << tf.p.thistask_y << " " << tf.p.thistask_z << std::endl;
  std::cout << "n{x,y,z}_tot " << tf.nx_tot << " " << tf.ny_tot << " " << tf.nz_tot << std::endl;
  std::cout << "nz_tot_p2 " << tf.nz_tot_p2 << std::endl;

  std::cout << "n{x,y,z}_loc " << tf.nx_loc << " " << tf.ny_loc << " " << tf.nz_loc << std::endl;
  std::cout << "n{x,y,z}_loc_start " << tf.nx_loc_start << " " << tf.ny_loc_start << " " << tf.nz_loc_start
            << std::endl;
  std::cout << "n{x,y,z}_loc_end " << tf.nx_loc_end << " " << tf.ny_loc_end << " " << tf.nz_loc_end << std::endl;
  std::cout << "nz_loc_p2 " << tf.nz_loc_p2 << std::endl;

  std::cout << "c_n{x,y,z}_loc " << tf.c_nx_loc << " " << tf.c_ny_loc << " " << tf.c_nz_loc << std::endl;
  std::cout << "c_n{x,y,z}_loc_start " << tf.c_nx_loc_start << " " << tf.c_ny_loc_start << " " << tf.c_nz_loc_start
            << std::endl;
  std::cout << "c_n{x,y,z}_loc_end " << tf.c_nx_loc_end << " " << tf.c_ny_loc_end << " " << tf.c_nz_loc_end
            << std::endl;
}

#ifdef DOUBLEPRECISION_FFTW
typedef double fft_real;
#else
typedef float fft_real;
#endif

/* FFTW library */
#ifdef __USE_FFTW__

#include <fftw3.h>
#include <fftw3-mpi.h>

#ifdef DOUBLEPRECISION_FFTW
// empty
#else
#define fftw_complex fftwf_complex
#define fftw_plan fftwf_plan

#define fftw_init_threads fftwf_init_threads
#define fftw_cleanup_threads fftwf_cleanup_threads
#define fftw_plan_with_nthreads fftwf_plan_with_nthreads

#define fftw_mpi_init fftwf_mpi_init
#define fftw_mpi_local_size_3d fftwf_mpi_local_size_3d
#define fftw_plan_dft_r2c_3d fftwf_plan_dft_r2c_3d
#define fftw_mpi_plan_dft_r2c_3d fftwf_mpi_plan_dft_r2c_3d
#define fftw_mpi_plan_dft_c2r_3d fftwf_mpi_plan_dft_c2r_3d
#define fftw_execute fftwf_execute
#define fftw_mpi_execute_dft_r2c fftwf_mpi_execute_dft_r2c
#define fftw_mpi_execute_dft_c2r fftwf_mpi_execute_dft_c2r
#define fftw_destroy_plan fftwf_destroy_plan
#endif
#endif /* __USE_FFTW__ */

/* PFFT library */
#ifdef __USE_PFFT__

#include <pfft.h>

#ifdef DOUBLEPRECISION_FFTW
// empty
#else
#undef pfft_complex
#define pfft_complex pfftf_complex
#define pfft_get_nthreads pfftf_get_nthreads
#define pfft_init pfftf_init
#define pfft_plan pfftf_plan
#define pfft_plan_with_nthreads pfftf_plan_with_nthreads
#define pfft_plan_many_dft_r2c pfftf_plan_many_dft_r2c
#define pfft_plan_many_dft_c2r pfftf_plan_many_dft_c2r
#define pfft_execute pfftf_execute
#define pfft_execute_dft_c2r pfftf_execute_dft_c2r
#define pfft_execute_dft_r2c pfftf_execute_dft_r2c
#define pfft_destroy_plan pfftf_destroy_plan

#define pfft_create_procmesh pfftf_create_procmesh
#define pfft_local_size_many_dft_r2c pfftf_local_size_many_dft_r2c
#define pfft_local_size_dft_r2c_3d pfftf_local_size_dft_r2c_3d
#endif
#endif /* __USE_PFFT__ */

/* HeFFTe library */
#ifdef __USE_HEFFTE__

#undef c_re
#undef c_im
#define c_re(c) ((c).real())
#define c_im(c) ((c).imag())

#include <heffte.h>

#ifdef DOUBLEPRECISION_FFTW
// empty
#else
#define fftw_init_threads fftwf_init_threads
#define fftw_cleanup_threads fftwf_cleanup_threads
#define fftw_plan_with_nthreads fftwf_plan_with_nthreads
#endif
#endif /* __USE_HEFFTE__ */
