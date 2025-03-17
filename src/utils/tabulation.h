#ifndef UTILS_TABULATION_H
#define UTILS_TABULATION_H

#include "utils/array.h"
#include "utils/types.h"

#include <Kokkos_Core.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

namespace math = Kokkos;
namespace py   = pybind11;

namespace rgnr {

  template <bool LG>
  KOKKOS_INLINE_FUNCTION auto InterpolateTabulatedFunction(
    const real_t&                x0,
    const Kokkos::View<real_t*>& x,
    const Kokkos::View<real_t*>& y,
    const std::size_t&           n,
    const real_t&                xmin,
    const real_t&                xmax,
    const real_t&                yfill = 0.0) -> real_t {

    if (x0 < xmin or x0 >= xmax) {
      return yfill;
    }
    if constexpr (LG) {
      auto xi = static_cast<std::size_t>(static_cast<real_t>(n - 1) *
                                         math::abs(math::log10(x0 / xmin)) /
                                         math::log10(xmax / xmin));
      if (xi >= n - 1) {
        return y(n - 1);
      } else {
        return (y(xi + 1) * math::log10(x0 / x(xi)) +
                y(xi) * math::log10(x(xi + 1) / x0)) /
               math::log10(x(xi + 1) / x(xi));
      }
    } else {
      auto xi = static_cast<std::size_t>(
        static_cast<real_t>(n - 1) * math::abs(x0 - xmin) / (xmax - xmin));
      if (xi >= n - 1) {
        return y(n - 1);
      } else {
        return (y(xi + 1) * (x0 - x(xi)) + y(xi) * (x(xi + 1) - x0)) /
               (x(xi + 1) - x(xi));
      }
    }
  }

  template <bool LG>
  class TabulatedFunction {
    // Kokkos::View<real_t*> m_x, m_y;
    Array<real_t*> m_x, m_y;

    const real_t      m_yfill;
    const std::size_t m_n;
    real_t            m_xmin, m_xmax;

    void verify() const;

  public:
    TabulatedFunction(const Array<real_t*>& x,
                      const Array<real_t*>& y,
                      real_t                yfill = 0.0);

    TabulatedFunction(const Kokkos::View<real_t*>& x,
                      const Kokkos::View<real_t*>& y,
                      real_t                       yfill = 0.0);

    TabulatedFunction(const std::vector<real_t>& x,
                      const std::vector<real_t>& y,
                      real_t                     yfill = 0.0);

    void findMinMax();

    // getters
    auto xView() const -> const Kokkos::View<real_t*>& {
      return m_x.data;
    }

    auto yView() const -> const Kokkos::View<real_t*>& {
      return m_y.data;
    }

    auto xArr() const -> const Array<real_t*>& {
      return m_x;
    }

    auto yArr() const -> const Array<real_t*>& {
      return m_y;
    }

    auto nPoints() const -> std::size_t {
      return m_n;
    }

    auto yFill() const -> real_t {
      return m_yfill;
    }

    auto xMin() const -> real_t {
      return m_xmin;
    }

    auto xMax() const -> real_t {
      return m_xmax;
    }
  };

  template <bool LG>
  void pyDefineTabulatedFunction(py::module&);

} // namespace rgnr

#endif // UTILS_TABULATION_H
