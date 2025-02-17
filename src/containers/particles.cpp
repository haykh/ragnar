#include "containers/particles.hpp"

#include "utils/types.h"

#include <Kokkos_Core.hpp>

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>

namespace rgnr {

  template <dim_t D>
  void Particles<D>::setNactive(std::size_t nactive) {
    if (!is_allocated()) {
      throw std::runtime_error("Particles not allocated");
    }
    m_nactive = nactive;
  }

  template <dim_t D>
  void Particles<D>::setIgnoreCoords(bool ignore_coords) {
    m_coords_ignored = ignore_coords;
  }

  template <dim_t D>
  auto Particles<D>::is_allocated() const -> bool {
    return m_is_allocated;
  }

  template <dim_t D>
  void Particles<D>::allocate(std::size_t nalloc) {
    if (is_allocated()) {
      throw std::runtime_error("Particles already allocated");
    }
    if (not m_coords_ignored) {
      X = Kokkos::View<real_t* [D]> {
        "X", nalloc
      };
    }
    U = Kokkos::View<real_t* [3]> {
      "U", nalloc
    };
    E = Kokkos::View<real_t* [3]> {
      "E", nalloc
    };
    B = Kokkos::View<real_t* [3]> {
      "B", nalloc
    };
    m_is_allocated = true;
  }

  template <dim_t D>
  void Particles<D>::printHead(std::size_t start, std::size_t number) const {
    std::cout << "Particles: " << m_label;
    if (is_allocated()) {
      if (start + number > m_nactive) {
        throw std::runtime_error(
          "Number of particles to print exceeds allocated space");
      }
      std::cout << " [ " << m_nactive << " / " << U.extent(0) << " ]" << std::endl;
      std::cout << " showing " << start << " to " << start + number << std::endl;
      const auto nelems = std::min(m_nactive, number);
      const auto slice  = std::make_pair(static_cast<int>(start),
                                        static_cast<int>(nelems));

      if (not m_coords_ignored) { // print X
        auto X_h = Kokkos::create_mirror_view(
          Kokkos::subview(X, slice, Kokkos::ALL));
        Kokkos::deep_copy(X_h, Kokkos::subview(X, slice, Kokkos::ALL));

        for (auto d = 0u; d < D; ++d) {
          std::cout << "  X_" << d + 1 << " : ";
          if (start > 0) {
            std::cout << "... ";
          }
          for (auto i = 0u; i < nelems; ++i) {
            std::cout << X_h(i, d) << " ";
          }
          if (m_nactive > number) {
            std::cout << "..." << std::endl;
          } else {
            std::cout << std::endl;
          }
        }
      }
      { // print U, E, B
        auto quantities_str = std::array<std::string, 3> { "U", "E", "B" };
        auto U_h            = Kokkos::create_mirror_view(
          Kokkos::subview(U, slice, Kokkos::ALL));
        auto E_h = Kokkos::create_mirror_view(
          Kokkos::subview(E, slice, Kokkos::ALL));
        auto B_h = Kokkos::create_mirror_view(
          Kokkos::subview(B, slice, Kokkos::ALL));
        Kokkos::deep_copy(U_h, Kokkos::subview(U, slice, Kokkos::ALL));
        Kokkos::deep_copy(E_h, Kokkos::subview(E, slice, Kokkos::ALL));
        Kokkos::deep_copy(B_h, Kokkos::subview(B, slice, Kokkos::ALL));

        auto quantities_view = std::array<decltype(U_h), 3> { U_h, E_h, B_h };

        for (auto q = 0; q < 3; ++q) {
          for (auto d = 0u; d < 3; ++d) {
            std::cout << "  " << quantities_str[q] << "_" << d + 1 << " : ";
            if (start > 0) {
              std::cout << "... ";
            }
            for (auto i = 0u; i < nelems; ++i) {
              std::cout << quantities_view[q](i, d) << " ";
            }
            if (m_nactive > number) {
              std::cout << "..." << std::endl;
            } else {
              std::cout << std::endl;
            }
          }
        }
      }
    } else {
      std::cout << " [ not allocated ]" << std::endl;
    }
  }

  template <dim_t D>
  auto Particles<D>::range() const -> Kokkos::RangePolicy<> {
    return Kokkos::RangePolicy<> { 0, m_nactive };
  }

  template class Particles<1>;
  template class Particles<2>;
  template class Particles<3>;

} // namespace rgnr
