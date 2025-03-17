#ifndef IO_H5_HPP
#define IO_H5_HPP

#include "utils/array.h"

#include <Kokkos_Core.hpp>
#include <highfive/highfive.hpp>
#include <pybind11/pybind11.h>

#include <map>
#include <string>
#include <utility>

namespace py = pybind11;

namespace rgnr::io::h5 {

  template <typename T>
  auto Read1DArray(const std::string&,
                   const std::string&,
                   std::size_t = 0,
                   std::size_t = 1) -> Array<T*>;

  template <typename T>
  void Write1DArray(const std::string&, const std::string&, const Array<T*>&);

  template <typename T>
  void pyDefineRead1DArray(py::module&);

  template <typename T>
  void pyDefineWrite1DArray(py::module&);

} // namespace rgnr::io::h5

#endif // IO_H5_HPP
