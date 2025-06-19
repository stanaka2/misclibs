#pragma once

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cassert>
#include <hdf5.h>

class load_halos
{
public:
  /* attribute */
  int ifile = 0;
  int nfile;
  int64_t nhalos_all, nhalos_file;
  double boxsize;
  float a, z;
  float Om, Ol, h0;
  float box_size;
  double bound[6];
  float particle_mass;

  /* 1:NGP, 2:CIC, 3:TSC*/
  int scheme = -1;

  /* body */
  std::vector<float> pos, vel; // x3
  std::vector<float> mvir, rvir;

  std::vector<int64_t> id, parent_id;
  std::vector<int> level;
  std::vector<float> pot, pot_tree, pot_pm; // pot is pot_total

  ~load_halos() {}

  void print_header();
  void check_scheme() const;

  template <typename T>
  void read_attribute(hid_t, const char *, T *);
  template <typename T>
  void read_data_field(hid_t, const char *, T *);

  void read_header(std::string);
  void read_data(std::string);

  void read_pos_data(std::string);
  void read_mvir_data(std::string);
  void read_rvir_data(std::string);

  void read_pot_data(std::string);
  void read_pot_tree_data(std::string);
  void read_pot_pm_data(std::string);
  void read_level_data(std::string);

  template <typename T>
  void load_halo_pos(T &, std::string, std::string);
  template <typename T>
  void load_halo_pm(T &, T &, std::string, std::string);
  template <typename T, typename U>
  void load_halo_pml(T &, T &, U &, std::string, std::string);
  template <typename T, typename U>
  void load_halo_pmpl(T &, T &, T &, U &, std::string, std::string);
};

void load_halos::check_scheme() const
{
  std::cout << "Halo mesh assignment scheme = ";
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

void load_halos::read_data(std::string filename)
{
  // update header by this file
  read_header(filename);
  read_pos_data(filename);
  read_mvir_data(filename);
  read_rvir_data(filename);

  read_pot_data(filename);
  read_pot_tree_data(filename);
  read_pot_pm_data(filename);
  read_level_data(filename);
}

void load_halos::print_header()
{
  std::cout << "# --- check header ---" << std::endl;
  std::cout << "# ifile: " << ifile << std::endl;
  std::cout << "# Nfile: " << nfile << std::endl;
  std::cout << "# NumHalosAll: " << nhalos_all << " ~ " << (int)(std::pow(nhalos_all, 1.0 / 3.0)) << "^3" << std::endl;
  std::cout << "# NumHalos: " << nhalos_file << " ~ " << (int)(std::pow(nhalos_file, 1.0 / 3.0)) << "^3" << std::endl;
  std::cout << "# OmegaM(cb): " << Om << std::endl;
  std::cout << "# OmegaLambda: " << Ol << std::endl;
  std::cout << "# HubbleParam: " << h0 << std::endl;
  std::cout << "# a: " << a << std::endl;
  std::cout << "# z: " << z << std::endl;
  std::cout << "# BoxSize: " << box_size << std::endl;
  std::cout << "# Bounds: ";
  for(int i = 0; i < 6; ++i) {
    std::cout << bound[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "# ParticleMass: " << particle_mass << " [Msun/h]" << std::endl;
  std::cout << "# ------" << std::endl;
}

template <typename T>
void load_halos::read_attribute(hid_t handle, const char *name, T *data)
{
  herr_t status;
  hid_t datatype = H5T_NATIVE_INT64; // Default data type (can be overridden below)
  if constexpr(std::is_same_v<T, float>) datatype = H5T_NATIVE_FLOAT;
  else if constexpr(std::is_same_v<T, double>) datatype = H5T_NATIVE_DOUBLE;
  else if constexpr(std::is_same_v<T, uint64_t>) datatype = H5T_NATIVE_UINT64;
  else if constexpr(std::is_same_v<T, int>) datatype = H5T_NATIVE_INT;
  else if constexpr(std::is_same_v<T, uint>) datatype = H5T_NATIVE_UINT;
  else if constexpr(std::is_same_v<T, long long int>) datatype = H5T_NATIVE_INT64;
  else if constexpr(std::is_same_v<T, unsigned long long>) datatype = H5T_NATIVE_UINT64;

  hid_t haid = H5Aopen(handle, name, H5P_DEFAULT);
  assert(haid >= 0);

  status = H5Aread(haid, datatype, data);
  assert(status >= 0);
  status = H5Aclose(haid);
  assert(status >= 0);
}

template <typename T>
void load_halos::read_data_field(hid_t handle, const char *name, T *data)
{
  herr_t status;
  hid_t datatype = H5T_NATIVE_INT64; // Default data type (can be overridden below)
  if constexpr(std::is_same_v<T, float>) datatype = H5T_NATIVE_FLOAT;
  else if constexpr(std::is_same_v<T, double>) datatype = H5T_NATIVE_DOUBLE;
  else if constexpr(std::is_same_v<T, uint64_t>) datatype = H5T_NATIVE_UINT64;
  else if constexpr(std::is_same_v<T, int>) datatype = H5T_NATIVE_INT;
  else if constexpr(std::is_same_v<T, uint>) datatype = H5T_NATIVE_UINT;
  else if constexpr(std::is_same_v<T, long long int>) datatype = H5T_NATIVE_INT64;
  else if constexpr(std::is_same_v<T, unsigned long long>) datatype = H5T_NATIVE_UINT64;

  hid_t hdid = H5Dopen(handle, name, H5P_DEFAULT);
  status = H5Dread(hdid, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
  assert(status >= 0);
  status = H5Dclose(hdid);
  assert(status >= 0);
}

void load_halos::read_header(std::string filename)
{
  hid_t hfid = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  assert(hfid >= 0);

  // Open a group named "Header".
  hid_t hgid = H5Gopen(hfid, "/Header", H5P_DEFAULT);
  assert(hgid >= 0);

  read_attribute(hgid, "Nfile", &nfile);
  read_attribute(hgid, "OmegaM", &Om);
  read_attribute(hgid, "OmegaLambda", &Ol);
  read_attribute(hgid, "HubbleParam", &h0);
  read_attribute(hgid, "a", &a);
  read_attribute(hgid, "z", &z);
  read_attribute(hgid, "BoxSize", &box_size);
  read_attribute(hgid, "Bounds", bound);
  read_attribute(hgid, "NumHalosAll", &nhalos_all);
  read_attribute(hgid, "NumHalos", &nhalos_file);
  read_attribute(hgid, "ParticleMass", &particle_mass);

  assert(H5Gclose(hgid) >= 0); // Close the group.
  assert(H5Fclose(hfid) >= 0); // Close the file.
}

void load_halos::read_pos_data(std::string filename)
{
  pos.resize(nhalos_file * 3);
  hid_t hfid = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  assert(hfid >= 0);
  hid_t hgid = H5Gopen(hfid, "/Halos", H5P_DEFAULT);
  read_data_field(hgid, "pos", pos.data());
  assert(H5Gclose(hgid) >= 0); // Close the group.
  assert(H5Fclose(hfid) >= 0); // Close the file.
}

void load_halos::read_mvir_data(std::string filename)
{
  mvir.resize(nhalos_file);
  hid_t hfid = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  assert(hfid >= 0);
  hid_t hgid = H5Gopen(hfid, "/Halos", H5P_DEFAULT);
  read_data_field(hgid, "Mvir", mvir.data());
  assert(H5Gclose(hgid) >= 0); // Close the group.
  assert(H5Fclose(hfid) >= 0); // Close the file.
}

void load_halos::read_rvir_data(std::string filename)
{
  rvir.resize(nhalos_file);
  hid_t hfid = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  assert(hfid >= 0);
  hid_t hgid = H5Gopen(hfid, "/Halos", H5P_DEFAULT);
  read_data_field(hgid, "Rvir", rvir.data());
  assert(H5Gclose(hgid) >= 0); // Close the group.
  assert(H5Fclose(hfid) >= 0); // Close the file.
}

void load_halos::read_pot_data(std::string filename)
{
  pot.resize(nhalos_file);
  hid_t hfid = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  assert(hfid >= 0);
  hid_t hgid = H5Gopen(hfid, "/Halos", H5P_DEFAULT);
  read_data_field(hgid, "pot_total", pot.data());
  assert(H5Gclose(hgid) >= 0); // Close the group.
  assert(H5Fclose(hfid) >= 0); // Close the file.
}

void load_halos::read_pot_tree_data(std::string filename)
{
  pot_tree.resize(nhalos_file);
  hid_t hfid = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  assert(hfid >= 0);
  hid_t hgid = H5Gopen(hfid, "/Halos", H5P_DEFAULT);
  read_data_field(hgid, "pot_tree", pot_tree.data());
  assert(H5Gclose(hgid) >= 0); // Close the group.
  assert(H5Fclose(hfid) >= 0); // Close the file.
}

void load_halos::read_pot_pm_data(std::string filename)
{
  pot_pm.resize(nhalos_file);
  hid_t hfid = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  assert(hfid >= 0);
  hid_t hgid = H5Gopen(hfid, "/Halos", H5P_DEFAULT);
  read_data_field(hgid, "pot_pm", pot_pm.data());
  assert(H5Gclose(hgid) >= 0); // Close the group.
  assert(H5Fclose(hfid) >= 0); // Close the file.
}

void load_halos::read_level_data(std::string filename)
{
  level.resize(nhalos_file);
  hid_t hfid = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  assert(hfid >= 0);
  hid_t hgid = H5Gopen(hfid, "/Halos", H5P_DEFAULT);
  read_data_field(hgid, "child_level", level.data());
  assert(H5Gclose(hgid) >= 0); // Close the group.
  assert(H5Fclose(hfid) >= 0); // Close the file.
}

template <typename T>
void load_halos::load_halo_pos(T &in_pos, std::string input_prefix, std::string suffix)
{
  auto nhalos = nhalos_all;

  in_pos.clear();
  in_pos.reserve(nhalos * 3);

  for(int ifile = 0; ifile < nfile; ifile++) {
    char cifile[128];
    sprintf(cifile, ".%d", ifile);
    std::string input_file = input_prefix + cifile + suffix;
    read_data(input_file);
    in_pos.insert(in_pos.end(), pos.begin(), pos.end());
  }
  assert(in_pos.size() == nhalos * 3);

  pos.clear();
  pos.shrink_to_fit();
}

template <typename T>
void load_halos::load_halo_pm(T &in_pos, T &in_mvir, std::string input_prefix, std::string suffix)
{
  auto nhalos = nhalos_all;
  in_pos.clear();
  in_mvir.clear();
  in_pos.reserve(nhalos * 3);
  in_mvir.reserve(nhalos);

  for(int ifile = 0; ifile < nfile; ifile++) {
    char cifile[128];
    sprintf(cifile, ".%d", ifile);
    std::string input_file = input_prefix + cifile + suffix;
    read_data(input_file);
    in_pos.insert(in_pos.end(), pos.begin(), pos.end());
    in_mvir.insert(in_mvir.end(), mvir.begin(), mvir.end());
  }
  assert(in_pos.size() == nhalos * 3);
  assert(in_mvir.size() == nhalos);

  pos.clear();
  pos.shrink_to_fit();
  mvir.clear();
  mvir.shrink_to_fit();
}

template <typename T, typename U>
void load_halos::load_halo_pml(T &in_pos, T &in_mvir, U &in_level, std::string input_prefix, std::string suffix)
{
  auto nhalos = nhalos_all;

  in_pos.clear();
  in_mvir.clear();
  in_level.clear();

  in_pos.reserve(nhalos * 3);
  in_mvir.reserve(nhalos);
  in_level.reserve(nhalos);

  for(int ifile = 0; ifile < nfile; ifile++) {
    char cifile[128];
    sprintf(cifile, ".%d", ifile);
    std::string input_file = input_prefix + cifile + suffix;
    read_data(input_file);
    in_pos.insert(in_pos.end(), pos.begin(), pos.end());
    in_mvir.insert(in_mvir.end(), mvir.begin(), mvir.end());
    in_level.insert(in_level.end(), level.begin(), level.end());
  }

  assert(in_pos.size() == nhalos * 3);
  assert(in_mvir.size() == nhalos);
  assert(in_level.size() == nhalos);

  pos.clear();
  pos.shrink_to_fit();
  mvir.clear();
  mvir.shrink_to_fit();
  level.clear();
  level.shrink_to_fit();
}

template <typename T, typename U>
void load_halos::load_halo_pmpl(T &in_pos, T &in_mvir, T &in_pot, U &in_level, std::string input_prefix,
                                std::string suffix)
{
  auto nhalos = nhalos_all;

  in_pos.clear();
  in_mvir.clear();
  in_pot.clear();
  in_level.clear();

  in_pos.reserve(nhalos * 3);
  in_mvir.reserve(nhalos);
  in_pot.reserve(nhalos);
  in_level.reserve(nhalos);

  for(int ifile = 0; ifile < nfile; ifile++) {
    char cifile[128];
    sprintf(cifile, ".%d", ifile);
    std::string input_file = input_prefix + cifile + suffix;
    read_data(input_file);
    in_pos.insert(in_pos.end(), pos.begin(), pos.end());
    in_mvir.insert(in_mvir.end(), mvir.begin(), mvir.end());
    in_pot.insert(in_pot.end(), pot.begin(), pot.end());
    in_level.insert(in_level.end(), level.begin(), level.end());
  }

  assert(in_pos.size() == nhalos * 3);
  assert(in_mvir.size() == nhalos);
  assert(in_pot.size() == nhalos);
  assert(in_level.size() == nhalos);

  pos.clear();
  pos.shrink_to_fit();
  mvir.clear();
  mvir.shrink_to_fit();
  pot.clear();
  pot.shrink_to_fit();
  level.clear();
  level.shrink_to_fit();
}
