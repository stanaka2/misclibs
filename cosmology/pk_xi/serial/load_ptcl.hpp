#pragma once

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cassert>

#include "field.hpp"
#include "util.hpp"

#include "gadget_header.hpp"
#define SKIP fin.read((char *)&dummy, sizeof(int));

struct particle_pot_str {
  float pos[3];
  float pot, pot_tree, pot_pm;
  uint64_t id;
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

  /* 1:NGP, 2:CIC, 3:TSC*/
  int scheme = -1;

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

  void load_gdt_pot(std::string FileBase)
  {
    int dummy;
    gadget::header htmp = h; // initialization by rank 0 header
    std::vector<int> filenum_list;

    for(int i = 0; i < nfiles; i++) filenum_list.push_back(i);

    std::cerr << "Reading gadget snapshot files ...";

    pdata.reserve((long long int)(npart_tot));

    for(auto itr = filenum_list.begin(); itr != filenum_list.end(); ++itr) {
      std::cerr << "read Gadget snapshot : " << FileBase + "." + itos(*itr) << "..." << std::endl;
      std::string file_buff = detect_hierarchical_input(FileBase, *itr);
      std::ifstream fin(file_buff.c_str(), std::ios::binary);

      SKIP;
      fin.read((char *)&htmp, sizeof(gadget::header));
      SKIP;

      uint64_t np = htmp.npart[1];

      T pdata1;
      std::vector<float> pos(3 * np);
      std::vector<uint64_t> idx(np);
      std::vector<float> pot;

      SKIP;
      fin.read((char *)&pos[0], 3 * np * sizeof(float));
      SKIP;

      // Skip velocity information.
      // header + dummy (2+2+2) + pos + vel
      std::streamoff seek_count = sizeof(gadget::header) + (6 * sizeof(int)) + (6 * sizeof(float) * np);
      fin.seekg(seek_count, std::ios_base::beg); // Probably faster than std::ios_base::cur.

      SKIP;
      fin.read((char *)&idx[0], np * sizeof(uint64_t));
      SKIP;

      pot.resize(2 * np);
      SKIP;
      fin.read((char *)&pot[0], 2 * np * sizeof(float));
      SKIP;

      fin.close();

      double boxsize = h.BoxSize;

      for(int i = 0; i < np; i++) {
        for(int j = 0; j < 3; j++) {
          pdata1.pos[j] = pos[3 * i + j];

          if(pdata1.pos[j] < 0.0) pdata1.pos[j] += (double)boxsize;
          if(pdata1.pos[j] >= boxsize) pdata1.pos[j] -= (double)boxsize;
        }

        pdata1.id = idx[i];

        pdata1.pot_tree = pot[2 * i + 0]; // tree
        pdata1.pot_pm = pot[2 * i + 1];   // PM
        pdata1.pot = pdata1.pot_tree + pdata1.pot_pm;

        pdata.push_back(pdata1);
      }
    }

    h = htmp;
    std::cerr << " done." << std::endl;
  }

  template <typename U>
  void load_gdt_and_assing(std::string FileBase, U &mesh, int type)
  {
    check_scheme();

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

      pdata.resize(np);

      T pdata1;
      std::vector<float> pos(3 * np);
      std::vector<uint64_t> idx(np);
      std::vector<float> pot;

      SKIP;
      fin.read((char *)&pos[0], 3 * np * sizeof(float));
      SKIP;

      if(type == 1) {
        // Skip position and velocity and index information.
        // header + dummy (2+2+2) + pos + vel
        std::streamoff seek_count =
            sizeof(gadget::header) + (2 * 4 * sizeof(int)) + (3 * 2 * sizeof(float) * np) + (sizeof(uint64_t) * np);
        fin.seekg(seek_count, std::ios_base::beg); // Probably faster than std::ios_base::cur.

        pot.resize(2 * np);
        SKIP;
        fin.read((char *)&pot[0], 2 * np * sizeof(float));
        SKIP;
      }

      fin.close();

      double boxsize = h.BoxSize;

      for(int i = 0; i < np; i++) {

        for(int j = 0; j < 3; j++) {
          pdata1.pos[j] = pos[3 * i + j];
          if(pdata1.pos[j] < 0.0) pdata1.pos[j] += (double)boxsize;
          if(pdata1.pos[j] >= boxsize) pdata1.pos[j] -= (double)boxsize;
        }

        if(type == 1) {
          pdata1.pot_tree = pot[2 * i + 0]; // tree
          pdata1.pot_pm = pot[2 * i + 1];   // PM
          pdata1.pot = pdata1.pot_tree + pdata1.pot_pm;
        }

        pdata[i] = pdata1;
      }

      ptcl_assign_mesh(pdata, mesh, nmesh, boxsize, ptcl_mass, type, scheme);

    } // ifile loop

    h = htmp;
    std::cerr << " done." << std::endl;
  }
};
