#ifndef PHYSICS_COMPTON_H
#define PHYSICS_COMPTON_H

#include "global.h"
#include "tools.h"
#include "particles.h"

#include <pbar/pbar.h>

#include <vector>
#include <cmath>

auto sigma_diff_compton(const double& mu, const double& e_ph) -> double {
  return 3.0 * (1.0 + SQR(mu))
         * (1.0
            + (SQR(e_ph) * SQR(1.0 - mu))
                / ((1.0 + e_ph * (1.0 - mu)) * (1.0 + mu * mu)))
         / (16.0 * M_PI * SQR(1.0 + e_ph * (1.0 - mu)));
}

auto sigma_compton(const double& e_ph) -> double {
  double sigma = (1.0 + e_ph) / CUBE(e_ph)
                   * (2.0 * e_ph * (1.0 + e_ph) / (1.0 + 2.0 * e_ph)
                      - std::log(1.0 + 2.0 * e_ph))
                 + std::log(1.0 + 2.0 * e_ph) / (2.0 * e_ph)
                 - (1.0 + 3.0 * e_ph) / SQR(1.0 + 2.0 * e_ph);
  return sigma / 2.0;
}

auto get_scattering_mu(const double& e_ph) -> double {
  double x2 = 1000.0;
  double x1 = 100.0;
  double sigma = 0.0;
  while (x2 > sigma) {
    x1 = (randomReal() - 0.5) * 2.0;
    x2 = randomReal() * 0.15;
    sigma = sigma_diff_compton(x1, e_ph);
  }
  return x1;
}

void ComptonScatter(const std::vector<Particle>& leptons,
                    std::vector<Particle>& photons,
                    double optical_depth) {
  pbar::ProgressBar lepton_progress(leptons.begin(), leptons.end(), 50, '#');
  for (const auto& lepton : lepton_progress) {
    for (auto& photon : photons) {
      if (photon.flag != 0) { continue; }
      auto p_photon_1 = lorentz_boost(lepton, photon);
      double prob = sigma_compton(p_photon_1.energy) * p_photon_1.energy
                    / (lepton.energy * photon.energy);
      prob *= optical_depth / (double)(leptons.size());
      if (randomReal() < prob) {
        // scattering occurs
        auto scat_mu = get_scattering_mu(p_photon_1.energy);
        vec_t<3> a_basis, b_, b_basis;
        a_basis[0] = p_photon_1.px / p_photon_1.energy;
        a_basis[1] = p_photon_1.py / p_photon_1.energy;
        a_basis[2] = p_photon_1.pz / p_photon_1.energy;
        generate_random_direction(randomReal(), randomReal(), b_);

        b_basis[0] = a_basis[1] * b_[2] - a_basis[2] * b_[1];
        b_basis[1] = a_basis[2] * b_[0] - a_basis[0] * b_[2];
        b_basis[2] = a_basis[0] * b_[1] - a_basis[1] * b_[0];

        double b_norm
          = std::sqrt(b_basis[0] * b_basis[0] + b_basis[1] * b_basis[1]
                      + b_basis[2] * b_basis[2]);
        b_basis[0] /= b_norm;
        b_basis[1] /= b_norm;
        b_basis[2] /= b_norm;

        double e_ph_2
          = p_photon_1.energy / (1.0 + p_photon_1.energy * (1.0 - scat_mu));

        vec_t<3> k_ph_2;
        k_ph_2[0]
          = scat_mu * a_basis[0] + std::sqrt(1.0 - SQR(scat_mu)) * b_basis[0];
        k_ph_2[1]
          = scat_mu * a_basis[1] + std::sqrt(1.0 - SQR(scat_mu)) * b_basis[1];
        k_ph_2[2]
          = scat_mu * a_basis[2] + std::sqrt(1.0 - SQR(scat_mu)) * b_basis[2];

        photon.px = e_ph_2 * k_ph_2[0];
        photon.py = e_ph_2 * k_ph_2[1];
        photon.pz = e_ph_2 * k_ph_2[2];
        photon.energy = e_ph_2;
        ++photon.flag;
        vec_t<4> ph_scat;
        lorentz_boost({lepton.energy, -lepton.px, -lepton.py, -lepton.pz},
                      {photon.energy, photon.px, photon.py, photon.pz},
                      ph_scat);
        photon.px = ph_scat[1];
        photon.py = ph_scat[2];
        photon.pz = ph_scat[3];
        photon.energy = ph_scat[0];
      } else {
        // no scattering
      }
    }
  }
}

#endif // PHYSICS_COMPTON_H

// double max_prob {-1.0};
// if (optical_depth == 0.0) {
//   for (const auto& lepton : leptons) {
//     for (const auto& photon : photons) {
//       if (photon.flag != 0) { continue; }
//       auto p_photon_1 = lorentz_boost(lepton, photon);
//       double prob = sigma_compton(p_photon_1.energy) * p_photon_1.energy
//                     / (lepton.energy * photon.energy);
//       if (prob > max_prob) { max_prob = prob; }
//     }
//   }
//   optical_depth = max_prob;
// }
// std::cout << " PROB: " << max_prob << std::endl;
