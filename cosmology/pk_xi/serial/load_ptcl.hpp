#pragma once

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <cassert>

#include "field.hpp"
#include "cosm_tools.hpp"
#include "util.hpp"

#include "gadget_header.hpp"
#define SKIP fin.read((char *)&dummy, sizeof(int));

struct particle_str {
  float pos[3];
  // uint64_t id;
  bool in_flag = true; // If true, not ghost region in this particle
};

struct particle_vp_str {
  float pos[3];
  float vel[3];
  float pot;
  //  uint64_t id;
  bool in_flag = true; // If true, not ghost region in this particle
};

template <typename T>
class load_ptcl
{
public:
  gadget::header h;
  int nfiles;

  uint64_t npart_tot;
  double ptcl_mass;
  double lbox;
  int nmesh;

  double sampling_rate = 1.0;

  /* 1:NGP, 2:CIC, 3:TSC*/
  int scheme = -1;

  std::string los_axis = "z";
  bool do_RSD = false;  // Redshift-space distortion effect
  bool do_Gred = false; // Gravitational redshift effect

  std::vector<T> pdata;

  ~load_ptcl() {}

  void free_pdata()
  {
    pdata.clear();
    pdata.shrink_to_fit();
  }

  void check_scheme() const
  {
    std::cout << "Particle mesh assignment scheme = ";
    switch(scheme) {
    case 1:
      std::cout << "1 (NGP)" << std::endl;
      break;
    case 2:
      std::cout << "2 (CIC)" << std::endl;
      break;
    case 3:
      std::cout << "3 (TSC)" << std::endl;
      break;
    default:
      std::cerr << "Error: scheme is UNKNOWN (value = " << scheme << ")" << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  void read_header(std::string FileBase)
  {
    int dummy;
    gadget::header htmp;

    std::cerr << "Checking gadget snapshot header" << std::endl;
    std::ifstream fin;
    // fin.open((FileBase + ".0").c_str());
    std::string file_buff = detect_hierarchical_input(FileBase, 0);
    std::cerr << "Base File " << file_buff << std::endl;
    fin.open(file_buff.c_str());

    SKIP;
    fin.read((char *)&htmp, sizeof(gadget::header));
    fin.close();
    nfiles = htmp.num_files;
    npart_tot = (uint64_t)htmp.npartTotal[1] + ((uint64_t)htmp.npartTotal[2] << 32);
    std::cerr << "Found " << npart_tot << " particles in " << nfiles << " files." << std::endl;

    h = htmp;
    std::cerr << "Done check gadget snapshot header" << std::endl;

    ptcl_mass = h.mass[1] * 1.0e+10;
  }

  void load_gdt_ptcl(std::string FileBase)
  {
    constexpr bool has_vp = std::is_same_v<T, particle_vp_str>;
    if(do_RSD || do_Gred) assert(has_vp);

    int dummy;
    gadget::header htmp = h;
    std::vector<int> filenum_list;
    for(int i = 0; i < nfiles; i++) filenum_list.push_back(i);

    uint64_t neff = npart_tot * sampling_rate;
    std::cerr << "Sampling rate : " << sampling_rate << std::endl;
    std::cerr << "N effective : " << neff << std::endl;

    pdata.reserve(neff);
    pdata.clear();

    std::cerr << "Reading gadget snapshot files ...";

    std::mt19937 rng(100);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for(auto itr = filenum_list.begin(); itr != filenum_list.end(); ++itr) {
      std::cerr << "read Gadget snapshot : " << FileBase + "." + itos(*itr) << "..." << std::endl;
      std::string file_buff = detect_hierarchical_input(FileBase, *itr);
      std::ifstream fin(file_buff.c_str(), std::ios::binary);

      SKIP;
      fin.read((char *)&htmp, sizeof(gadget::header));
      SKIP;

      uint64_t np = htmp.npart[1];

      std::vector<float> pos(3 * np);
      SKIP;
      fin.read((char *)&pos[0], 3 * np * sizeof(float));
      SKIP;

      std::vector<float> vel;
      std::vector<float> pot;

      if(do_RSD) {
        vel.resize(3 * np);
        SKIP;
        fin.read((char *)&vel[0], 3 * np * sizeof(float));
        SKIP;
      } else {
        fin.seekg(3 * np * sizeof(float) + 2 * sizeof(int), std::ios_base::cur);
      }

      if(do_Gred) {
        fin.seekg(np * sizeof(uint64_t) + 2 * sizeof(int), std::ios_base::cur);
        pot.resize(2 * np);
        SKIP;
        fin.read((char *)&pot[0], 2 * np * sizeof(float));
        SKIP;
      }

      fin.close();

      double boxsize = h.BoxSize;
      const double a = 1.0 / (1.0 + htmp.redshift);
      const double sqrta = std::sqrt(a);

      for(uint64_t i = 0; i < np; i++) {

        if(sampling_rate < 1.0 && dist(rng) > sampling_rate) continue;

        T pdata1;

        for(int j = 0; j < 3; j++) {
          pdata1.pos[j] = pos[3 * i + j];
          if(pdata1.pos[j] < 0.0) pdata1.pos[j] += boxsize;
          if(pdata1.pos[j] >= boxsize) pdata1.pos[j] -= boxsize;
        }

        if constexpr(has_vp) {
          if(do_RSD) {
            for(int j = 0; j < 3; j++) {
              // Gadget to km/s
              pdata1.vel[j] = vel[3 * i + j] * sqrta;
            }
          }

          if(do_Gred) {
            // pdata1.id = idx[i];
            auto pot_tree = pot[2 * i];
            auto pot_pm = pot[2 * i + 1];
            pdata1.pot = pot_tree + pot_pm;
          }
        }
        pdata.emplace_back(std::move(pdata1));
      }
    }

    pdata.shrink_to_fit();

    h = htmp;
    std::cerr << " done." << std::endl;
  }

  void load_gdt_and_shift_ptcl(std::string FileBase)
  {
    // vel, pot does not hold the structure
    static_assert(std::is_same_v<T, particle_str>, "T must be particle_str in this function");

    int dummy;
    gadget::header htmp = h;
    std::vector<int> filenum_list(nfiles);
    std::iota(filenum_list.begin(), filenum_list.end(), 0);

    uint64_t neff = npart_tot * sampling_rate;
    std::cerr << "Sampling rate : " << sampling_rate << std::endl;
    std::cerr << "N effective : " << neff << std::endl;

    pdata.reserve(neff);
    pdata.clear();

    std::cerr << "Reading gadget snapshot files ..." << std::endl;

    std::mt19937 rng(100);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for(int ifile : filenum_list) {
      std::cerr << "read Gadget snapshot : " << FileBase + "." + itos(ifile) << "..." << std::endl;
      std::string file_buff = detect_hierarchical_input(FileBase, ifile);
      std::ifstream fin(file_buff.c_str(), std::ios::binary);

      SKIP;
      fin.read((char *)&htmp, sizeof(gadget::header));
      SKIP;

      uint64_t np = htmp.npart[1];
      std::vector<float> pos(3 * np);
      SKIP;
      fin.read((char *)&pos[0], 3 * np * sizeof(float));
      SKIP;

      std::vector<float> vel;
      if(do_RSD) {
        vel.resize(3 * np);
        SKIP;
        fin.read((char *)&vel[0], 3 * np * sizeof(float));
        SKIP;
      } else {
        fin.seekg(3 * np * sizeof(float) + 2 * sizeof(int), std::ios_base::cur);
      }

      std::vector<float> pot;
      if(do_Gred) {
        fin.seekg(np * sizeof(uint64_t) + 2 * sizeof(int), std::ios_base::cur);
        pot.resize(2 * np);
        SKIP;
        fin.read((char *)&pot[0], 2 * np * sizeof(float));
        SKIP;
      }

      fin.close();

      auto boxsize = h.BoxSize;
      auto a = 1.0 / (1.0 + htmp.redshift);
      auto sqrta = std::sqrt(a);
      auto Om = htmp.Omega0;
      auto Ol = htmp.OmegaLambda;
      auto _Ha = Ha(a, Om, Ol);
      auto factor_RSD = (do_RSD ? 1.0 / (a * _Ha) : 0.0);
      auto factor_Gred = (do_Gred ? cspeed / (a * _Ha) : 0.0);

      int los = (los_axis == "x") ? 0 : (los_axis == "y") ? 1 : 2;

      for(uint64_t i = 0; i < np; ++i) {
        if(sampling_rate < 1.0 && dist(rng) > sampling_rate) continue;

        T pdata1;
        for(int j = 0; j < 3; ++j) {
          auto pos_j = pos[3 * i + j];
          if(do_RSD && j == los) pos_j += vel[3 * i + j] * sqrta * factor_RSD;
          if(do_Gred && j == los) pos_j -= (pot[2 * i] + pot[2 * i + 1]) * factor_Gred;
          if(pos_j < 0.0) pos_j += boxsize;
          if(pos_j >= boxsize) pos_j -= boxsize;
          pdata1.pos[j] = pos_j;
        }
        pdata.emplace_back(std::move(pdata1));
      }
    }

    pdata.shrink_to_fit();

    h = htmp;
    std::cerr << " done." << std::endl;
  }

  template <typename U>
  void load_gdt_and_assing(std::string FileBase, U &mesh)
  {
    check_scheme();

    std::cerr << "Reading gadget snapshot files (no sampling)..." << std::endl;

    int dummy;
    gadget::header htmp = h; // initialization by rank 0 header
    std::vector<int> filenum_list;

    for(int i = 0; i < nfiles; i++) filenum_list.push_back(i);

    // std::cerr << "Reading gadget snapshot files ...";

    for(auto itr = filenum_list.begin(); itr != filenum_list.end(); ++itr) {
      // std::cerr << "read Gadget snapshot : " << FileBase + "." + itos(*itr) << "..." << std::endl;
      show_progress(*itr, nfiles, "read Gadget snapshot");

      std::string file_buff = detect_hierarchical_input(FileBase, *itr);
      std::ifstream fin(file_buff.c_str(), std::ios::binary);

      SKIP;
      fin.read((char *)&htmp, sizeof(gadget::header));
      SKIP;

      uint64_t np = htmp.npart[1];
      pdata.reserve(np);
      pdata.clear();

      std::vector<float> pos(3 * np);

      SKIP;
      fin.read((char *)&pos[0], 3 * np * sizeof(float));
      SKIP;

      fin.close();

      double boxsize = h.BoxSize;

      for(int64_t i = 0; i < np; i++) {
        T pdata1;
        for(int j = 0; j < 3; j++) {
          pdata1.pos[j] = pos[3 * i + j];
          if(pdata1.pos[j] < 0.0) pdata1.pos[j] += (double)boxsize;
          if(pdata1.pos[j] >= boxsize) pdata1.pos[j] -= (double)boxsize;
        }
        pdata.emplace_back(std::move(pdata1));
      }

      ptcl_assign_mesh(pdata, mesh, nmesh, boxsize, ptcl_mass, scheme);

    } // ifile loop

    h = htmp;
    std::cerr << " done." << std::endl;
  }

  template <typename U>
  void load_gdt_and_shift_assing(std::string FileBase, U &mesh)
  {
    check_scheme();
    static_assert(std::is_same_v<T, particle_str>, "T must be particle_str in this function");

    int dummy;
    gadget::header htmp = h;
    std::vector<int> filenum_list(nfiles);
    std::iota(filenum_list.begin(), filenum_list.end(), 0);

    std::cerr << "Reading gadget snapshot files with RSD/Gred shift (no sampling)..." << std::endl;

    for(int ifile : filenum_list) {
      show_progress(ifile, nfiles, "read Gadget snapshot");

      std::string file_buff = detect_hierarchical_input(FileBase, ifile);
      std::ifstream fin(file_buff.c_str(), std::ios::binary);

      SKIP;
      fin.read((char *)&htmp, sizeof(gadget::header));
      SKIP;

      uint64_t np = htmp.npart[1];
      pdata.reserve(np);
      pdata.clear();

      std::vector<float> pos(3 * np);

      SKIP;
      fin.read((char *)pos.data(), 3 * np * sizeof(float));
      SKIP;

      std::vector<float> vel;
      if(do_RSD) {
        vel.resize(3 * np);
        SKIP;
        fin.read((char *)vel.data(), 3 * np * sizeof(float));
        SKIP;
      } else {
        fin.seekg(3 * np * sizeof(float) + 2 * sizeof(int), std::ios_base::cur);
      }

      std::vector<float> pot;
      if(do_Gred) {
        fin.seekg(np * sizeof(uint64_t) + 2 * sizeof(int), std::ios_base::cur);
        pot.resize(2 * np);
        SKIP;
        fin.read((char *)pot.data(), 2 * np * sizeof(float));
        SKIP;
      }

      fin.close();

      const double boxsize = h.BoxSize;
      const double a = 1.0 / (1.0 + htmp.redshift);
      const double sqrta = std::sqrt(a);
      const double Om = htmp.Omega0;
      const double Ol = htmp.OmegaLambda;
      const double _Ha = Ha(a, Om, Ol);
      const double factor_RSD = do_RSD ? 1.0 / (a * _Ha) : 0.0;
      const double factor_Gred = do_Gred ? cspeed / (a * _Ha) : 0.0;

      const int los = (los_axis == "x") ? 0 : (los_axis == "y") ? 1 : 2;

      for(uint64_t i = 0; i < np; ++i) {
        T pdata1;
        for(int j = 0; j < 3; ++j) {
          float pos_j = pos[3 * i + j];
          if(do_RSD && j == los) pos_j += (vel[3 * i + j] * sqrta) * factor_RSD;
          if(do_Gred && j == los) pos_j -= (pot[2 * i] + pot[2 * i + 1]) * factor_Gred;
          if(pos_j < 0.0f) pos_j += boxsize;
          if(pos_j >= boxsize) pos_j -= boxsize;
          pdata1.pos[j] = pos_j;
        }
        pdata.emplace_back(std::move(pdata1));
      }

      ptcl_assign_mesh(pdata, mesh, nmesh, boxsize, ptcl_mass, scheme);
    }

    h = htmp;
    std::cerr << " done." << std::endl;
  }
};
