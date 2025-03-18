#ifndef PHYSICS_DISTRIBUTIONS_HPP
#define PHYSICS_DISTRIBUTIONS_HPP

#include "utils/types.h"

#include <Kokkos_Core.hpp>
#include <pybind11/pybind11.h>

#include <string>

namespace math = Kokkos;
namespace py   = pybind11;

namespace rgnr {

  struct BrokenPlawDistribution {
    const real_t e_break;
    const real_t p1, p2;

    std::string unit;

    BrokenPlawDistribution(real_t e_break, real_t p1, real_t p2)
      : e_break { e_break }
      , p1 { p1 }
      , p2 { p2 } {}

    KOKKOS_INLINE_FUNCTION auto dn(const real_t& energy) const -> real_t {
      if (energy < e_break) {
        return math::pow(energy / e_break, p1);
      } else {
        return math::pow(energy / e_break, p2);
      }
    }

    void AssertEnergyUnits(const std::string& unit) const {
      if (unit != EnergyUnits::mec2) {
        throw std::runtime_error("Energy must be in units of mec^2");
      }
    }
  };

  void pyDefineBrokenPlawDistribution(py::module&);

} // namespace rgnr

#endif // PHYSICS_DISTRIBUTIONS_HPP
