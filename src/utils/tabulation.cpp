#include "utils/tabulation.h"

#include "utils/types.h"

#include <Kokkos_Core.hpp>

#include <iostream>
#include <stdexcept>
#include <vector>

namespace rgnr {
  template <bool LG>
  TabulatedFunction<LG>::TabulatedFunction(Kokkos::View<real_t*> x,
                                           Kokkos::View<real_t*> y,
                                           real_t                yfill)
    : m_x { x }
    , m_y { y }
    , m_yfill { yfill }
    , m_n { x.extent(0) } {
    findMinMax();
    verify();
  }

  template <bool LG>
  TabulatedFunction<LG>::TabulatedFunction(const std::vector<real_t>& x,
                                           const std::vector<real_t>& y,
                                           real_t                     yfill)
    : m_yfill { yfill }
    , m_n { x.size() } {
    m_x      = Kokkos::View<real_t*> { "x", m_n };
    m_y      = Kokkos::View<real_t*> { "y", m_n };
    auto x_h = Kokkos::create_mirror_view(m_x);
    auto y_h = Kokkos::create_mirror_view(m_y);
    for (auto i = 0u; i < m_n; ++i) {
      x_h(i) = x[i];
      y_h(i) = y[i];
    }
    Kokkos::deep_copy(m_x, x_h);
    Kokkos::deep_copy(m_y, y_h);

    findMinMax();
    verify();
  }

  template <bool LG>
  void TabulatedFunction<LG>::findMinMax() {
    Kokkos::MinMaxScalar<real_t> xminmax;
    const auto&                  xref = xArr();
    Kokkos::parallel_reduce(
      "XMinMax",
      m_n,
      KOKKOS_LAMBDA(std::size_t p, Kokkos::MinMaxScalar<real_t> & lxminmax) {
        if (xref(p) < lxminmax.min_val) {
          lxminmax.min_val = xref(p);
        }
        if (xref(p) > lxminmax.max_val) {
          lxminmax.max_val = xref(p);
        }
      },
      Kokkos::MinMax<real_t> { xminmax });
    m_xmin = xminmax.min_val;
    m_xmax = xminmax.max_val;
  }

  template <bool LG>
  void TabulatedFunction<LG>::verify() const {
    if (m_y.extent(0) != m_n) {
      throw std::runtime_error("y.size != x.size in TabulatedFunction");
    }
    if (m_xmin >= m_xmax) {
      throw std::runtime_error("xmin >= xmax in TabulatedFunction");
    }
    if constexpr (LG) {
      if (m_xmin <= 0.0) {
        throw std::runtime_error("xmin <= 0.0 in Logspace TabulatedFunction");
      }
    }
  }

  template class TabulatedFunction<true>;
  template class TabulatedFunction<false>;

} // namespace rgnr
