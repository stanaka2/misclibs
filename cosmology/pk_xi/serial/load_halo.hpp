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

  template <typename T>
  hid_t check_h5_type(hid_t);
  template <typename T>
  std::vector<T> read_halo_dataset(const std::string &, const std::string &, const std::string &);
  template <typename T>
  std::vector<T> load_halo_field(const std::string &, const std::string &, const std::string &);
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

template <typename T>
hid_t load_halos::check_h5_type(hid_t h5_datatype_id)
{
  H5T_class_t type_class = H5Tget_class(h5_datatype_id);
  size_t type_size = H5Tget_size(h5_datatype_id);
  if constexpr(std::is_same_v<T, float>) {
    assert(type_class == H5T_FLOAT && type_size == sizeof(float));
    return H5T_NATIVE_FLOAT;
  } else if constexpr(std::is_same_v<T, double>) {
    assert(type_class == H5T_FLOAT && type_size == sizeof(double));
    return H5T_NATIVE_DOUBLE;
  } else if constexpr(std::is_same_v<T, int>) {
    assert(type_class == H5T_INTEGER && type_size == sizeof(int));
    return H5T_NATIVE_INT;
  } else if constexpr(std::is_same_v<T, uint>) {
    assert(type_class == H5T_INTEGER && type_size == sizeof(uint));
    return H5T_NATIVE_UINT;
  } else if constexpr(std::is_same_v<T, int64_t>) {
    assert(type_class == H5T_INTEGER && type_size == sizeof(int64_t));
    return H5T_NATIVE_INT64;
  } else if constexpr(std::is_same_v<T, uint64_t>) {
    assert(type_class == H5T_INTEGER && type_size == sizeof(uint64_t));
    return H5T_NATIVE_UINT64;
  } else if constexpr(std::is_same_v<T, long long int>) {
    assert(type_class == H5T_INTEGER && type_size == sizeof(long long int));
    return H5T_NATIVE_INT64;
  } else if constexpr(std::is_same_v<T, unsigned long long>) {
    assert(type_class == H5T_INTEGER && type_size == sizeof(unsigned long long));
    return H5T_NATIVE_UINT64;
  } else {
    static_assert(!sizeof(T), "Unsupported type in HDF5 type checker");
  }
}

template <typename T>
std::vector<T> load_halos::read_halo_dataset(const std::string &filename, const std::string &group_name,
                                             const std::string &field_name)
{
  herr_t status;
  hid_t hfid = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  assert(hfid >= 0);
  hid_t hgid = H5Gopen(hfid, group_name.c_str(), H5P_DEFAULT);
  assert(hgid >= 0);
  hid_t hdid = H5Dopen(hgid, field_name.c_str(), H5P_DEFAULT);
  assert(hdid >= 0);

  // check type
  hid_t htid = H5Dget_type(hdid);
  hid_t datatype = check_h5_type<T>(htid);
  assert(H5Tclose(htid) >= 0);

  // check shape
  hid_t hsid = H5Dget_space(hdid);
  assert(hsid >= 0);
  hsize_t dims[2] = {0, 0};
  int ndims = H5Sget_simple_extent_dims(hsid, dims, NULL);
  assert(ndims >= 1);
  assert(H5Sclose(hsid) >= 0);

  size_t total_size = 1;
  for(int d = 0; d < ndims; d++) total_size *= dims[d];

  // allocate and read data
  std::vector<T> data(total_size);
  // read_data_field(hgid, field_name.c_str(), data.data());
  //// hid_t hdid = H5Dopen(handle, name, H5P_DEFAULT);
  status = H5Dread(hdid, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
  assert(status >= 0);

  assert(H5Dclose(hdid) >= 0);
  assert(H5Gclose(hgid) >= 0);
  assert(H5Fclose(hfid) >= 0);
  return data;
}

template <typename T>
std::vector<T> load_halos::load_halo_field(const std::string &input_prefix, const std::string &suffix,
                                           const std::string &field_name)
{
  std::vector<T> data;
  for(int ifile = 0; ifile < nfile; ifile++) {
    char cifile[256];
    sprintf(cifile, ".%d", ifile);
    std::string input_file = input_prefix + cifile + suffix;
    auto chunk = read_halo_dataset<T>(input_file, "/Halos", field_name);
    data.insert(data.end(), chunk.begin(), chunk.end());
  }
  return data;
}
