#include "global.h"
#include "sim.h"
#include "particles.h"
#include "compton.h"
#include "io.h"

#include <vector>
#include <iostream>
#include <cmath>

// // Jones test
// void Simulation::run() {
//   std::vector<Particle> leptons;
//   std::vector<Particle> photons;
//   double gamma = 10.0;
//   double gamma_beta = std::sqrt(SQR(gamma) - 1.0);
//   double ph_energy = 0.001;

//   leptons.push_back(Particle {gamma_beta, 0.0, 0.0, gamma});

//   for (std::size_t i = 0; i < 10000; i++) {
//     vec_t<3> direction;
//     generate_random_direction(randomReal(), randomReal(), direction);
//     photons.push_back(Particle {ph_energy,
//                                 ph_energy * direction[0],
//                                 ph_energy * direction[1],
//                                 ph_energy * direction[2],
//                                 0});
//   }
//   ComptonScatter(leptons, photons, 1.0);

//   std::vector<double> energies;
//   for (std::size_t i = 0; i < photons.size(); i++) {
//     energies.push_back(photons[i].energy);
//   }
//   IO::writeArray<double>("prtls.h5", "e_ph", energies);
// }

// SSC test
void Simulation::run() {
  std::vector<Particle> leptons;
  std::vector<Particle> photons;

  const char* filename = "../prtl.tot.00100";

  {
    // read electrons
    std::vector<double> u_arr, v_arr, w_arr;
    IO::readArray(filename, "u_1", u_arr);
    IO::readArray(filename, "v_1", v_arr);
    IO::readArray(filename, "w_1", w_arr);
    for (std::size_t i {0}; i < u_arr.size(); ++i) {
      double gamma
        = std::sqrt(1.0 + SQR(u_arr[i]) + SQR(v_arr[i]) + SQR(w_arr[i]));
      leptons.push_back(Particle {u_arr[i], v_arr[i], w_arr[i], gamma, 0});
    }
    // read positrons
    IO::readArray(filename, "u_2", u_arr);
    IO::readArray(filename, "v_2", v_arr);
    IO::readArray(filename, "w_2", w_arr);
    for (std::size_t i {0}; i < u_arr.size(); ++i) {
      double gamma
        = std::sqrt(1.0 + SQR(u_arr[i]) + SQR(v_arr[i]) + SQR(w_arr[i]));
      leptons.push_back(Particle {u_arr[i], v_arr[i], w_arr[i], gamma, 0});
    }
    // read photons
    IO::readArray(filename, "u_3", u_arr);
    IO::readArray(filename, "v_3", v_arr);
    IO::readArray(filename, "w_3", w_arr);
    for (std::size_t i {0}; i < u_arr.size(); ++i) {
      double energy = std::sqrt(SQR(u_arr[i]) + SQR(v_arr[i]) + SQR(w_arr[i]));
      photons.push_back(Particle {u_arr[i], v_arr[i], w_arr[i], energy, 0});
    }
  }

  ComptonScatter(leptons, photons, 0.1);

  std::vector<double> energies;
  for (std::size_t i = 0; i < photons.size(); i++) {
    energies.push_back(photons[i].energy);
  }
  IO::writeArray<double>("ssc01.h5", "e_ph", energies);
}
