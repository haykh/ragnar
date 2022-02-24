#include "global.h"
#include "sim.h"
#include "particles.h"
#include "compton.h"
#include "io.h"

#include <vector>
#include <iostream>
#include <cmath>

void Simulation::run() {
  std::vector<Particle> leptons;
  std::vector<Particle> photons;
  double gamma = 10.0;
  double gamma_beta = std::sqrt(SQR(gamma) - 1.0);
  double ph_energy = 0.001;

  leptons.push_back(Particle {gamma_beta, 0.0, 0.0, gamma});

  for (std::size_t i = 0; i < 10000; i++) {
    vec_t<3> direction;
    generate_random_direction(randomReal(), randomReal(), direction);
    photons.push_back(Particle {ph_energy,
                                ph_energy * direction[0],
                                ph_energy * direction[1],
                                ph_energy * direction[2],
                                0});
  }
  ComptonScatter(leptons, photons, 1.0);

  std::vector<double> energies;
  for (std::size_t i = 0; i < photons.size(); i++) {
    energies.push_back(photons[i].energy);
  }
  IO::writeArray<double>("prtls.h5", "e_ph", energies);
}