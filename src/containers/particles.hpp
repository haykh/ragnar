#ifndef CONTAINERS_PARTICLES_HPP
#define CONTAINERS_PARTICLES_HPP

#include "utils/array.h"
#include "utils/types.h"

#include <Kokkos_Core.hpp>

#include <string>

namespace rgnr {

  template <dim_t D>
  struct Particles {
    Kokkos::View<real_t* [D]> X;
    Kokkos::View<real_t* [3]> U, E, B;

    Particles(const std::string& label) : m_label { label } {}

    auto range() const -> Kokkos::RangePolicy<>;
    void allocate(std::size_t);

    // setters
    void setNactive(std::size_t);
    void setIgnoreCoords(bool);

    // getters
    auto is_allocated() const -> bool {
      return m_is_allocated;
    }

    auto nactive() const -> std::size_t {
      return m_nactive;
    }

    auto label() const -> const std::string& {
      return m_label;
    }

    void printHead(std::size_t = 0, std::size_t = 5) const;
    auto repr() const -> std::string;

    // computes dN / de, where E is gamma or gamma * beta
    auto energyDistribution(const Array<real_t*>&, bool = true) const
      -> Array<real_t*>;

    // accessors
    auto Xarr(std::size_t) const -> Array<real_t*>;
    auto Uarr(std::size_t) const -> Array<real_t*>;
    auto Earr(std::size_t) const -> Array<real_t*>;
    auto Barr(std::size_t) const -> Array<real_t*>;

  private:
    bool        m_is_allocated { false };
    bool        m_coords_ignored { false };
    std::size_t m_nactive { 0 };

    const std::string m_label;
  };

  template <dim_t D>
  void pyDefineParticles(py::module&);

} // namespace rgnr

#endif // CONTAINERS_PARTICLES_HPP
