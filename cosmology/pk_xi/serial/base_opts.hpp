#pragma once

#include "../include/CLI/CLI.hpp"

class BaseOptions
{
public:
  /* default arguments */
  std::string input_prefix = "None";
  std::string output_filename = "./output.dat";

  bool log_bin = false;

  int nk = 100;
  std::vector<float> krange = {1e-2, 50.0};
  int nr = 100;
  std::vector<float> rrange = {1.0, 150.0};
  std::vector<float> mrange = {1.0f, 20.0f};
  std::vector<int> clevel = {0, 100};

  int nmesh = 1024;
  int p_assign = 3;
  bool no_shotnoise_corr = false;

  bool do_RSD = false;
  bool do_Gred = false;
  std::string los_axis = "z"; // line-of-sight axis for RSD and Gred

  std::string estimator = "ideal";
  int nrand_factor = 1;
  double sampling_rate = 1.0;

  bool verbose = false;

  /* constants arguments */
  const std::string h5_suffix = ".h5";

  /* end arguments */

  CLI::App app;

  /* constructors */
  BaseOptions() = default;
  BaseOptions(int argc, char **argv)
  {
    app.description(std::string(argv[0]) + " description");
    add_to_base_app(app);

    try {
      app.parse(argc, argv);
    } catch(const CLI::ParseError &e) {
      std::exit(app.exit(e));
    }
  }

  void print_args()
  {
    std::cout << "\n";
    auto maxlen = get_max_option_length();
    for(const auto &opt : app.get_options()) {
      if(opt->get_name() == "--help") continue;

      std::cout << std::left << std::setw(maxlen + 2) << opt->get_name() << "  ";
      const auto &vals = opt->results();
      if(!vals.empty()) {
        for(size_t i = 0; i < vals.size(); ++i) {
          if(i > 0) std::cout << " ";
          std::cout << vals[i];
        }
        std::cout << "\n";
      } else {
        std::string def = opt->get_default_str();
        std::replace(def.begin(), def.end(), ',', ' ');
        def.erase(std::remove(def.begin(), def.end(), '['), def.end());
        def.erase(std::remove(def.begin(), def.end(), ']'), def.end());
        std::cout << (def.empty() ? "false" : def) << "  (default)\n";
      }
    }
    std::cout << "\n";
  }

protected:
  /* argument function */
  template <typename T>
  void add_to_base_app(T &app)
  {
    // clang-format off
    app.add_option("-i,--input_prefix", input_prefix, "Input prefix")->capture_default_str();
    app.add_option("-o,--output_filename", output_filename, "Output filename")->capture_default_str();

    app.add_flag("--log_bin", log_bin, "use log binning")->capture_default_str();
    app.add_option("--nk", nk, "number of k-bins")->capture_default_str();
    app.add_option("--krange", krange, "kmin and kmax")->expected(2)->capture_default_str();
    app.add_option("--nr", nr, "number of r-bins")->capture_default_str();
    app.add_option("--rrange", rrange, "rmin and rmax")->expected(2)->capture_default_str();
    app.add_option("--mrange", mrange, "minimum halo mass (log10)")->expected(2)->capture_default_str();
    app.add_option("--clevel", clevel, "halo child level. 0:host- 1:sub- n:n-th child- halo")->expected(2)->capture_default_str();

    app.add_option("-n,--nmesh", nmesh, "number of FFT mesh size")->capture_default_str();
    app.add_option("-p,--p_assign", p_assign, "particle assign type")->check(CLI::IsMember({1, 2, 3}))->capture_default_str();
    app.add_flag("--no_shotnoise_corr", no_shotnoise_corr, "disable shot noise correction")->capture_default_str();

    app.add_flag("--RSD", do_RSD, "Apply redshift-space distortion (RSD) shift to positions")->capture_default_str();
    app.add_flag("--Gred", do_Gred, "Apply gravitational redshift shift to positions")->capture_default_str();
    app.add_option("--los_axis", los_axis, "Line-of-sight axis (x, y, or z) for applying RSD and gravitational redshift shifts")
        ->check(CLI::IsMember({"x", "y", "z"}))->capture_default_str();

    app.add_option("--est", estimator, "estimator for xi calculation")
        ->check(CLI::IsMember({"ideal", "RR", "LS"}))
        ->capture_default_str();
    app.add_option("--nrand_factor", nrand_factor, "factor of nrand to ngrp")->capture_default_str();
    app.add_option("--sampling_rate", sampling_rate, "sampling rate for pair-count")->capture_default_str();

    app.add_flag("-v,--verbose", verbose, "verbose output");
    // clang-format on
  }

  size_t get_max_option_length()
  {
    size_t maxlen = 0;
    for(const auto &opt : app.get_options()) {
      maxlen = std::max(maxlen, opt->get_name().size());
    }
    return maxlen;
  }
};
