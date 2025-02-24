#include "utils/types.h"

#include <Kokkos_Core.hpp>

namespace math = Kokkos;

namespace rgnr {

  struct SoftPhotonDistribution {
    const real_t eps_s_0 { static_cast<real_t>(1e-3) };

    void assertDimension(const EnergyUnits& u) const {
      if (u != EnergyUnits::eV) {
        throw std::invalid_argument(
          "SoftPhotonDistribution: invalid energy unit");
      }
    }

    KOKKOS_INLINE_FUNCTION auto dn(const real_t& eps) const -> real_t {
      if (eps < eps_s_0) {
        return eps_s_0 / eps;
      } else {
        return math::pow(eps / eps_s_0, -2.2);
      }
    }
  };

  void Simulation(int, char*[]);
} // namespace rgnr
