#pragma once

#include "num_integrate.hpp"

struct neutrino {
  int mass_nu_num;
  int deg[3];

  float sum_mass;
  float mass[3], frac[3];
};

struct cosmology {
  float omega_m, omega_v, omega_b, omega_nu, omega_r, hubble;
  float tend;
  struct neutrino nu;
};

double dtda(double a, const cosmology &cosm)
{
  auto om = cosm.omega_m, ov = cosm.omega_v;
  auto ok = 1.0 - om - ov;
  return std::sqrt(a / (om + ok * a + ov * a * a * a));
}

double atotime(double a, const cosmology &cosm)
{
  auto f = [&](double x) { return dtda(x, cosm); };
  return num::quad(f, 0.0, a, 1e-8);
}

double timetoa(double t, const cosmology &cosm)
{
  auto F = [&](double a) { return atotime(a, cosm) - t; };
  return num::brentq(F, 1e-4, 2.0, 1e-10, 200);
}

double timetoz(double t, const cosmology &cosm)
{
  auto a = timetoa(t, cosm);
  return 1.0 / a - 1.0;
}
