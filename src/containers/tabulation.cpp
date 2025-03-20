#include "containers/tabulation.hpp"

#include "utils/global.h"

#include "containers/array.hpp"

#include <Kokkos_Core.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;

namespace rgnr {

  template <bool LG>
  TabulatedFunction<LG>::TabulatedFunction(const Array1D<real_t>& x,
                                           const Array1D<real_t>& y,
                                           real_t                 yfill)
    : m_x { x }
    , m_y { y }
    , m_yfill { yfill }
    , m_n { m_x.data.extent(0) } {
    findMinMax();
    verify();
  }

  template <bool LG>
  TabulatedFunction<LG>::TabulatedFunction(const Kokkos::View<real_t*>& x,
                                           const Kokkos::View<real_t*>& y,
                                           real_t                       yfill)
    : m_x { x }
    , m_y { y }
    , m_yfill { yfill }
    , m_n { x.extent(0) } {
    findMinMax();
    verify();
  }

  template <bool LG>
  TabulatedFunction<LG>::TabulatedFunction(const py::array_t<real_t>& x,
                                           const py::array_t<real_t>& y,
                                           real_t                     yfill)
    : m_yfill { yfill }
    , m_n { (std::size_t)x.size() } {
    m_x.data = Kokkos::View<real_t*> { "x", m_n };
    m_y.data = Kokkos::View<real_t*> { "y", m_n };
    auto x_h = Kokkos::create_mirror_view(m_x.data);
    auto y_h = Kokkos::create_mirror_view(m_y.data);
    for (auto i = 0u; i < m_n; ++i) {
      x_h(i) = x.at(i);
      y_h(i) = y.at(i);
    }
    Kokkos::deep_copy(m_x.data, x_h);
    Kokkos::deep_copy(m_y.data, y_h);

    findMinMax();
    verify();
  }

  template <bool LG>
  TabulatedFunction<LG>::TabulatedFunction(const std::vector<real_t>& x,
                                           const std::vector<real_t>& y,
                                           real_t                     yfill)
    : m_yfill { yfill }
    , m_n { x.size() } {
    m_x.data = Kokkos::View<real_t*> { "x", m_n };
    m_y.data = Kokkos::View<real_t*> { "y", m_n };
    auto x_h = Kokkos::create_mirror_view(m_x.data);
    auto y_h = Kokkos::create_mirror_view(m_y.data);
    for (auto i = 0u; i < m_n; ++i) {
      x_h(i) = x[i];
      y_h(i) = y[i];
    }
    Kokkos::deep_copy(m_x.data, x_h);
    Kokkos::deep_copy(m_y.data, y_h);
    findMinMax();
    verify();
  }

  template <bool LG>
  void TabulatedFunction<LG>::findMinMax() {
    Kokkos::MinMaxScalar<real_t> xminmax;
    const auto&                  xref = xView();
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
    if (m_y.data.extent(0) != m_n) {
      throw std::range_error("y.size != x.size in TabulatedFunction");
    }
    if (m_xmin >= m_xmax) {
      throw std::range_error("xmin >= xmax in TabulatedFunction");
    }
    if constexpr (LG) {
      if (m_xmin <= 0.0) {
        throw std::range_error("xmin <= 0.0 in Logspace TabulatedFunction");
      }
    }
  }

  template <bool LG>
  void pyDefineTabulatedFunction(py::module& m) {
    std::string variant = LG ? "_log" : "";
    py::class_<TabulatedFunction<LG>>(m, ("TabulatedFunction" + variant).c_str())
      .def(py::init<const Array1D<real_t>&, const Array1D<real_t>&, real_t>(),
           "x"_a,
           "y"_a,
           "yfill"_a = 0.0)
      .def(py::init<py::array_t<real_t>&, const py::array_t<real_t>&, real_t>(),
           "x"_a,
           "y"_a,
           "yfill"_a = 0.0)
      .def("xArr", &TabulatedFunction<LG>::xArr)
      .def("yArr", &TabulatedFunction<LG>::yArr)
      .def("nPoints", &TabulatedFunction<LG>::nPoints)
      .def("yFill", &TabulatedFunction<LG>::yFill)
      .def("xMin", &TabulatedFunction<LG>::xMin)
      .def("xMax", &TabulatedFunction<LG>::xMax)
      .doc() = R"rgnrdoc(
        A class holding a tabulated function (x -> y)

        Parameters
        ----------
        x : Array | List
          The x values

        y : Array | List
          The y values

        yfill : float
          The value to fill the function with outside the x range [default: 0.0]
        
        Methods
        -------
        xArr() -> Array
          Return the x values as an Array
        
        yArr() -> Array
          Return the y values as an Array
        
        nPoints() -> int
          Return the number of points
        
        yFill() -> float
          Return the fill value

        xMin() -> float
          Return the minimum x value

        xMax() -> float
          Return the maximum x value
      )rgnrdoc";
  }

  template class TabulatedFunction<true>;
  template class TabulatedFunction<false>;

  template void pyDefineTabulatedFunction<true>(py::module&);
  template void pyDefineTabulatedFunction<false>(py::module&);

} // namespace rgnr
