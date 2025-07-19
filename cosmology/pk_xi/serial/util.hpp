#pragma once

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <omp.h>
#include <vector>
#include <numeric>
#include <algorithm>

typedef unsigned long long int U64;
const double pi = 3.141592653589793238462643383279;

#pragma omp declare reduction(vec_double_plus : std::vector<double> : std::transform(         \
        omp_in.begin(), omp_in.end(), omp_out.begin(), omp_out.begin(), std::plus<double>())) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

#pragma omp declare reduction(vec_float_plus : std::vector<float> : std::transform(          \
        omp_in.begin(), omp_in.end(), omp_out.begin(), omp_out.begin(), std::plus<float>())) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

std::string itos(int n)
{
  std::stringstream str_stream;
  str_stream << n;
  return str_stream.str();
}

std::string itos2(int n)
{
  std::stringstream str_stream;
  if(n >= 10) str_stream << n;
  else str_stream << "0" << n;
  return str_stream.str();
}

std::string itos3(int n)
{
  std::stringstream str_stream;
  if(n < 10) str_stream << "00" << n;
  else if(n < 100) str_stream << "0" << n;
  else str_stream << n;
  return str_stream.str();
}

std::streamsize get_fs(std::string fname)
{
  std::ifstream ifs(fname.c_str(), std::ios_base::binary);
  std::streamsize size = ifs.seekg(0, std::ios::end).tellg();
  ifs.close();
  return size;
}

int is_directory(const char *input)
{
  struct stat st;
  if(stat(input, &st) != 0) return 0;
  return S_ISDIR(st.st_mode);
}

std::string detect_hierarchical_input(std::string fname, int file_id)
{
  static int isd = -1;
  if(isd == -1) {
    isd = is_directory((fname + ".0").c_str());
  }

  if(isd) {
    int dir_id = file_id / 1000;
    return (fname + "." + itos(dir_id) + "/" + itos(file_id));
  } else {
    return (fname + "." + itos(file_id));
  }
}

int is_directory_safe(const char *input)
{
  // Sleep just for the safety
  srand((unsigned int)time(NULL));
  int irand = rand() % 50 + 1;
  struct timespec ts;
  ts.tv_sec = 0;
  ts.tv_nsec = 1000 * irand; // 1-50[\mu s]
  clock_nanosleep(CLOCK_MONOTONIC, 0, &ts, NULL);

  struct stat st;
  if(stat(input, &st) != 0) return 0;
  return S_ISDIR(st.st_mode);
}

void make_directory(const char *directory_name)
{
  if(is_directory_safe(directory_name)) return;

  // Set the absolute PATH of the directory
  static char cwd_path[1024];
  if(getcwd(cwd_path, sizeof(cwd_path)) == NULL) {
    fprintf(stderr, "Error in make_directory");
    std::exit(EXIT_FAILURE);
  }
  strcat(cwd_path, "/");
  strcat(cwd_path, directory_name);

  mode_t mode = S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH;
  mkdir(cwd_path, mode);
}

template <typename T>
void show_progress(const int istep, const int total_step, const T &label)
{
  const int step_div = std::max(1, total_step / 100);
  const double ratio = 100.0 * (double)istep / (double)total_step;
  if(istep % step_div == 0) {
    std::cerr << "\r\033[2K";
    std::cerr << label << " :: " << istep << " / " << total_step << " :: " << ratio << " [%]";
  }
  if(istep == total_step - 1) std::cerr << std::endl;
}
