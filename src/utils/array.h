#ifndef UTILS_ARRAY_H
#define UTILS_ARRAY_H

#include <Kokkos_Core.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#include <string>
#include <vector>

namespace rgnr {

  template <class T>
  struct Array {
    std::string     unit;
    Kokkos::View<T> data;

    Array(const std::string& unit = "") : unit { unit } {}

    Array(const std::string& unit, const Kokkos::View<T>& data)
      : unit { unit }
      , data { data } {}

    Array(const Kokkos::View<T>& data) : data { data } {}

    void head(std::size_t = 10, std::size_t = 0) const;
    auto repr() const -> std::string;
    auto as_array() const -> py::array_t<std::remove_pointer_t<T>>;

    auto extent(unsigned short d = 0) const -> std::size_t {
      return data.extent(d);
    }
  };

  template <class T>
  void pyDefineArray(py::module&, const std::string&, unsigned short);

} // namespace rgnr

#endif // UTILS_ARRAY_H
