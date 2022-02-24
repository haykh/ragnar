#ifndef TOOLS_H
#define TOOLS_H

#include "particles.h"

#include <cmath>

auto randomReal() -> double {
  return static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
}

void generate_random_direction(const double& r1,
                               const double& r2,
                               vec_t<3> dir) {
  double Z = 2.0 * r1 - 1;
  double TH = r2 * 2.0 * M_PI;
  dir[0] = std::sqrt(1.0 - SQR(Z)) * std::cos(TH);
  dir[1] = std::sqrt(1.0 - SQR(Z)) * std::sin(TH);
  dir[2] = Z;
}

auto lorentz_boost(const vec_t<4>& p_particle,
                   const vec_t<4>& p_photon,
                   vec_t<4>& p_photon_boosted) -> void {
  double gamma_prtl = p_particle[0];
  double e_phot = p_photon[0];
  double u_prtl
    = std::sqrt(p_particle[1] * p_particle[1] + p_particle[2] * p_particle[2]
                + p_particle[3] * p_particle[3]);
  vec_t<3> u_prtl_unit
    = {p_particle[1] / u_prtl, p_particle[2] / u_prtl, p_particle[3] / u_prtl};
  double u_phot_dot_u_prtl = u_prtl_unit[0] * p_photon[1]
                             + u_prtl_unit[1] * p_photon[2]
                             + u_prtl_unit[2] * p_photon[3];
  vec_t<3> u_prtl_mult_Dot = {u_prtl_unit[0] * u_phot_dot_u_prtl,
                              u_prtl_unit[1] * u_phot_dot_u_prtl,
                              u_prtl_unit[2] * u_phot_dot_u_prtl};
  vec_t<3> u_prtl_mult_Eph
    = {p_particle[1] * e_phot, p_particle[2] * e_phot, p_particle[3] * e_phot};
  vec_t<3> u_phot_prime = {
    p_photon[1] + (gamma_prtl - 1.0) * u_prtl_mult_Dot[0] - u_prtl_mult_Eph[0],
    p_photon[2] + (gamma_prtl - 1.0) * u_prtl_mult_Dot[1] - u_prtl_mult_Eph[1],
    p_photon[3] + (gamma_prtl - 1.0) * u_prtl_mult_Dot[2] - u_prtl_mult_Eph[2]};
  double e_phot_prime = std::sqrt(u_phot_prime[0] * u_phot_prime[0]
                                  + u_phot_prime[1] * u_phot_prime[1]
                                  + u_phot_prime[2] * u_phot_prime[2]);
  p_photon_boosted[0] = e_phot_prime;
  p_photon_boosted[1] = u_phot_prime[0];
  p_photon_boosted[2] = u_phot_prime[1];
  p_photon_boosted[3] = u_phot_prime[2];
}

auto lorentz_boost(const Particle& lepton, const Particle& photon) -> Particle {
  vec_t<4> p_lepton {lepton.energy, lepton.px, lepton.py, lepton.pz};
  vec_t<4> p_photon {photon.energy, photon.px, photon.py, photon.pz};
  vec_t<4> p_photon_boosted;
  lorentz_boost(p_lepton, p_photon, p_photon_boosted);
  return Particle {p_photon_boosted[0],
                   p_photon_boosted[1],
                   p_photon_boosted[2],
                   p_photon_boosted[3],
                   photon.flag};
}

#endif // PHYSICS_TOOLS_H
