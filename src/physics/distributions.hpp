#ifndef PHYSICS_DISTRIBUTIONS_HPP
#define PHYSICS_DISTRIBUTIONS_HPP

namespace rgnr {

  struct BrokenPlawDistribution {
    const real_t e_break;
    const real_t p1, p2;

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
  };

} // namespace rgnr

#endif // PHYSICS_DISTRIBUTIONS_HPP
