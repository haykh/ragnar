#ifndef CONTAINERS_PARTICLES_HPP
#define CONTAINERS_PARTICLES_HPP

#include "utils/types.h"

#include <Kokkos_Core.hpp>

#include <string>

namespace rgnr {

  template <dim_t D>
  struct Particles {
    Kokkos::View<real_t* [D]> X;
    Kokkos::View<real_t* [3]> U, E, B;

    Particles(const std::string& label) : m_label { label } {}

    void setNactive(std::size_t);
    void setIgnoreCoords(bool);
    void allocate(std::size_t);

    void printHead(std::size_t = 0, std::size_t = 5) const;

    auto is_allocated() const -> bool;
    auto range() const -> Kokkos::RangePolicy<>;

    auto nactive() const -> std::size_t {
      return m_nactive;
    }

    auto label() const -> const std::string& {
      return m_label;
    }

    // computes dN / d(gamma - 1)
    auto energyDistribution(const Kokkos::View<real_t*>&) const
      -> Kokkos::View<real_t*>;

  private:
    bool        m_is_allocated { false };
    bool        m_coords_ignored { false };
    std::size_t m_nactive { 0 };

    const std::string m_label;
  };

} // namespace rgnr

#endif // CONTAINERS_PARTICLES_HPP
