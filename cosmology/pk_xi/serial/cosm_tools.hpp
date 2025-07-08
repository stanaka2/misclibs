#pragma once

#include <cmath>

constexpr double cspeed = 2.99792458e+5;       // [km/s]
constexpr double pc = 3.0856775814913673e+13;  // [km]
constexpr double mpc = 3.0856775814913673e+19; // [km]
constexpr double year = 3.15576e+7;            // [s]
constexpr double Msun = 1.98847e+33;           // [g]

inline double atoz(double a) { return 1.0 / a - 1.0; }
inline double ztoa(double z) { return 1.0 / (1.0 + z); }

inline double Ha(double a, double Om, double Ol)
{
  // H(a) returns Hubble parameter in units of km/s/(Mpc/h)
  auto H0 = 100.0; // km/s/(Mpc/h)
  return H0 * std::sqrt(Om / (a * a * a) + Ol);
}

inline double Hz(double z, double Om, double Ol)
{
  // H(z) returns Hubble parameter in units of km/s/(Mpc/h)
  auto a = ztoa(z);
  return Ha(a, Om, Ol);
}
