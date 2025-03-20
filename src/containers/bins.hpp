#ifndef CONTAINERS_BINS_HPP
#define CONTAINERS_BINS_HPP

#include "utils/global.h"

#include "containers/array.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <string>

namespace py = pybind11;

namespace rgnr {

  struct Bins : Array1D<real_t> {
    bool        log_spaced { false };
    std::string unit;

    Bins(const Array1D<real_t>& arr, const std::string& unit = "")
      : Array1D<real_t> { arr.data }
      , unit { unit } {}

    Bins(const std::string& unit = "") : Array1D<real_t> {}, unit { unit } {}

    Bins(const py::array_t<real_t>& arr, const std::string& unit = "")
      : Array1D<real_t> { arr }
      , unit { unit } {}
  };

  auto Linbins(real_t, real_t, std::size_t, const std::string& = "") -> Bins;
  auto Logbins(real_t, real_t, std::size_t, const std::string& = "") -> Bins;

  void pyDefineBins(py::module&);

} // namespace rgnr

#endif // CONTAINERS_BINS_HPP
