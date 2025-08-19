#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <chrono>
#include <mpi.h>

template <typename M>
void mean_dens(M &mesh, fft_param &tf)
{
  auto LOC_IDX_WP = [&](int ix, int iy, int iz) -> int64_t {
    return iz + tf.nz_loc_p2 * (iy + tf.ny_loc * (int64_t)ix);
  };

  double sum = 0.0;

#pragma omp parallel for collapse(3) reduction(+ : sum)
  for(int ix = 0; ix < tf.nx_loc; ix++) {
    for(int iy = 0; iy < tf.ny_loc; iy++) {
      for(int iz = 0; iz < tf.nz_loc; iz++) {
        sum += mesh[LOC_IDX_WP(ix, iy, iz)];
      }
    }
  }

  if(tf.p.decomp > 0) MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  const double ntot = (double)tf.nx_tot * (double)tf.ny_tot * (double)tf.nz_tot;
  double mean = sum / ntot;

#pragma omp parallel for collapse(3)
  for(int ix = 0; ix < tf.nx_loc; ix++) {
    for(int iy = 0; iy < tf.ny_loc; iy++) {
      for(int iz = 0; iz < tf.nz_loc; iz++) {
        auto idx = LOC_IDX_WP(ix, iy, iz);
        mesh[idx] = mesh[idx] / mean - 1.0;
      }
    }
  }
}

template <typename M>
void init_dens(M &mesh, fft_param &tf, std::string ic_type)
{
  auto GLB_IDX = [&](int ix, int iy, int iz) -> int64_t { return iz + tf.nz_tot * (iy + tf.ny_tot * (int64_t)ix); };
  auto LOC_IDX = [&](int ix, int iy, int iz) -> int64_t { return iz + tf.nz_loc * (iy + tf.ny_loc * (int64_t)ix); };

  auto GLB_IDX_WP = [&](int ix, int iy, int iz) -> int64_t {
    return iz + tf.nz_tot_p2 * (iy + tf.ny_tot * (int64_t)ix);
  };
  auto LOC_IDX_WP = [&](int ix, int iy, int iz) -> int64_t {
    return iz + tf.nz_loc_p2 * (iy + tf.ny_loc * (int64_t)ix);
  };

#pragma omp parallel for
  for(int64_t i = 0; i < tf.loc_size; i++) {
    mesh[i] = 0.0;
  }

  const auto L = 1.0;
  const auto dx = L / tf.nx_tot;
  const auto dy = L / tf.ny_tot;
  const auto dz = L / tf.nz_tot;

  const auto xcen = 0.5 * L;
  const auto ycen = 0.5 * L;
  const auto zcen = 0.5 * L;

  if(ic_type == "gaussian") {
    const auto sigma = 0.01 * L;

#pragma omp parallel for collapse(3)
    for(int ix = 0; ix < tf.nx_loc; ix++) {
      for(int iy = 0; iy < tf.ny_loc; iy++) {
        for(int iz = 0; iz < tf.nz_loc; iz++) {

          int gix = ix + tf.nx_loc_start;
          int giy = iy + tf.ny_loc_start;
          int giz = iz + tf.nz_loc_start;

          auto xpos = (0.5 + gix) * dx;
          auto ypos = (0.5 + giy) * dy;
          auto zpos = (0.5 + giz) * dz;

          auto tmpx = xpos - xcen;
          auto tmpy = ypos - ycen;
          auto tmpz = zpos - zcen;

          double r2 = tmpx * tmpx + tmpy * tmpy + tmpz * tmpz;
          mesh[LOC_IDX_WP(ix, iy, iz)] = std::exp(-0.5 * r2 / (sigma * sigma));
        }
      }
    }

  } else if(ic_type == "tophat") {
    double R = 0.15 * L;

#pragma omp parallel for collapse(3)
    for(int ix = 0; ix < tf.nx_loc; ix++) {
      for(int iy = 0; iy < tf.ny_loc; iy++) {
        for(int iz = 0; iz < tf.nz_loc; iz++) {

          int gix = ix + tf.nx_loc_start;
          int giy = iy + tf.ny_loc_start;
          int giz = iz + tf.nz_loc_start;

          auto xpos = (0.5 + gix) * dx;
          auto ypos = (0.5 + giy) * dy;
          auto zpos = (0.5 + giz) * dz;

          auto tmpx = xpos - xcen;
          auto tmpy = ypos - ycen;
          auto tmpz = zpos - zcen;

          double r2 = tmpx * tmpx + tmpy * tmpy + tmpz * tmpz;
          if(r2 < R * R) {
            mesh[LOC_IDX_WP(ix, iy, iz)] = 1.0;
          }
        }
      }
    }
  }

  std::cout << "set initial density : " << ic_type << " type" << std::endl;
}

template <typename M>
void output_dens(M &mesh, fft_param &tf, std::string filename)
{
  std::string output = filename + "_" + std::to_string(tf.p.thistask);
  std::ofstream fout(output, std::ios::binary);

  fout.write(reinterpret_cast<const char *>(&(tf.p.ntasks)), sizeof(int));
  fout.write(reinterpret_cast<const char *>(&(tf.p.ntasks_x)), sizeof(int));
  fout.write(reinterpret_cast<const char *>(&(tf.p.ntasks_y)), sizeof(int));
  fout.write(reinterpret_cast<const char *>(&(tf.p.ntasks_z)), sizeof(int));

  fout.write(reinterpret_cast<const char *>(&(tf.p.thistask)), sizeof(int));
  fout.write(reinterpret_cast<const char *>(&(tf.p.thistask_x)), sizeof(int));
  fout.write(reinterpret_cast<const char *>(&(tf.p.thistask_y)), sizeof(int));
  fout.write(reinterpret_cast<const char *>(&(tf.p.thistask_z)), sizeof(int));

  fout.write(reinterpret_cast<const char *>(&(tf.nx_loc)), sizeof(int));
  fout.write(reinterpret_cast<const char *>(&(tf.ny_loc)), sizeof(int));
  fout.write(reinterpret_cast<const char *>(&(tf.nz_loc)), sizeof(int));

  for(int64_t ix = 0; ix < tf.nx_loc; ix++) {
    for(int64_t iy = 0; iy < tf.ny_loc; iy++) {
      int64_t offset = tf.nz_loc_p2 * (iy + tf.ny_loc * ix);
      fout.write(reinterpret_cast<const char *>(&mesh[offset]), sizeof(fft_real) * tf.nz_loc);
    }
  }
  fout.close();

  std::cout << "output density : " << output << std::endl;
}

template <typename M>
void output_dens_hat(M &mesh, fft_param &tf, std::string filename)
{
  std::string output = filename + "_" + std::to_string(tf.p.thistask);
  std::ofstream fout(output, std::ios::binary);

  fout.write(reinterpret_cast<const char *>(&(tf.p.ntasks)), sizeof(int));
  fout.write(reinterpret_cast<const char *>(&(tf.p.ntasks_x)), sizeof(int));
  fout.write(reinterpret_cast<const char *>(&(tf.p.ntasks_y)), sizeof(int));
  fout.write(reinterpret_cast<const char *>(&(tf.p.ntasks_z)), sizeof(int));

  fout.write(reinterpret_cast<const char *>(&(tf.p.thistask)), sizeof(int));
  fout.write(reinterpret_cast<const char *>(&(tf.p.thistask_x)), sizeof(int));
  fout.write(reinterpret_cast<const char *>(&(tf.p.thistask_y)), sizeof(int));
  fout.write(reinterpret_cast<const char *>(&(tf.p.thistask_z)), sizeof(int));

  fout.write(reinterpret_cast<const char *>(&(tf.c_nx_loc)), sizeof(int));
  fout.write(reinterpret_cast<const char *>(&(tf.c_ny_loc)), sizeof(int));
  fout.write(reinterpret_cast<const char *>(&(tf.c_nz_loc)), sizeof(int));

  for(int64_t ix = 0; ix < tf.c_nx_loc; ix++) {
    for(int64_t iy = 0; iy < tf.c_ny_loc; iy++) {
      int64_t offset = tf.c_nz_loc * (iy + tf.c_ny_loc * ix);
      fout.write(reinterpret_cast<const char *>(&mesh[offset]), sizeof(mesh[0]) * tf.c_nz_loc);
    }
  }
  fout.close();

  std::cout << "output density in FFT space : " << output << std::endl;
}

template <typename M, typename T>
void calc_pk(const M &mesh_hat, T &power, T &weight, int nk, const fft_param &tf)
{
  constexpr double tiny = 1e-30;
  power.assign(nk, 0.0);
  weight.assign(nk, 0.0);

  const double norm = tf.nx_tot * tf.ny_tot * tf.nz_tot;

  for(uint64_t ix = 0; ix < tf.c_nx_loc; ix++) {
    for(uint64_t iy = 0; iy < tf.c_ny_loc; iy++) {
      for(uint64_t iz = 0; iz < tf.c_nz_loc; iz++) {

        const int64_t gix = tf.c_nx_loc_start + ix;
        const int64_t giy = tf.c_ny_loc_start + iy;
        const int64_t giz = tf.c_nz_loc_start + iz;

        const double kx = (double)(gix <= tf.nx_tot / 2 ? gix : tf.nx_tot - gix);
        const double ky = (double)(giy <= tf.ny_tot / 2 ? giy : tf.ny_tot - giy);
        const double kz = (double)giz;

        uint64_t ik;
        ik = (uint64_t)(sqrt(SQR(kx) + SQR(ky) + SQR(kz)));

        uint64_t loc_kid = iz + tf.c_nz_loc * (iy + tf.c_ny_loc * ix);

        if(ik < nk) {
          power[ik] += SQR(c_re(mesh_hat[loc_kid])) + SQR(c_im(mesh_hat[loc_kid]));
          weight[ik] += 1.0;
        }
      }
    }
  }

  for(int ik = 0; ik < nk; ik++) {
    power[ik] /= (norm * norm);
  }

  if(tf.p.decomp >= 1) {
    MPI_Allreduce(MPI_IN_PLACE, power.data(), nk, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, weight.data(), nk, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }

  for(int ik = 0; ik < nk; ik++) {
    power[ik] /= (weight[ik] + tiny);
  }
}

template <typename T>
void output_pk(const T &power, const T &weight, int nk, const std::string &filename)
{
  auto lbox = 1.0;
  auto dk = 2.0 * M_PI / (1.0 - 0.0); // xmax - xmin
  auto V = lbox * lbox * lbox;
  auto pk_box_norm = V;

  const auto sigma = 0.01 * lbox;
  const double A_th = V;

  std::ofstream fout(filename);
  fout << "# k P(k) Ptheory(k) weight" << std::endl;

  for(int ipk = 1; ipk < nk / 2; ipk++) {
    double kk = dk * ((double)ipk + 0.5) / lbox;
    double pk = power[ipk] * pk_box_norm;
    double w = weight[ipk];
    double Pth = A_th * std::exp(-(sigma * sigma) * kk * kk);
    if(Pth < 1e-20) Pth = 1e-20;
    fout << std::scientific << std::setprecision(10) << kk << " " << pk << " " << Pth << " " << w << std::endl;
  }

  fout.close();
  std::cerr << "Output written to " << filename << std::endl;
}
