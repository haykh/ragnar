#ifndef CONTAINERS_PARTICLES_H
#define CONTAINERS_PARTICLES_H

#include "global.h"

#include <string>

struct Particle {
  double energy, px, py, pz;
  int flag;
  Particle(const double& energy,
           const double& px,
           const double& py,
           const double& pz,
           const int& flag)
    : energy(energy), px(px), py(py), pz(pz), flag(flag) {}
  Particle(const double& energy,
           const double& px,
           const double& py,
           const double& pz)
    : energy(energy), px(px), py(py), pz(pz), flag(0) {}
  Particle(const Particle& other)
    : energy(other.energy),
      px(other.px),
      py(other.py),
      pz(other.pz),
      flag(other.flag) {}
};

#endif // CONTAINERS_PARTICLES_H