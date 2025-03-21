#ifndef CONTAINERS_ARRAY_HPP
#define CONTAINERS_ARRAY_HPP

#include <Kokkos_Core.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#include <string>
#include <vector>

namespace rgnr {

  template <class T>
  struct Array1D {
    Kokkos::View<T*> data;

    Array1D() {}

    Array1D(const py::array_t<T>& arr)
      : data {
        Kokkos::View<T*> { "data", (std::size_t)(arr.size()) }
    } {
      auto host_view = Kokkos::create_mirror_view(data);
      for (auto i = 0u; i < arr.size(); ++i) {
        host_view(i) = arr.at(i);
      }
      Kokkos::deep_copy(data, host_view);
    }

    Array1D(const Kokkos::View<T*>& data) : data { data } {}

    void head(std::size_t = 10, std::size_t = 0) const;
    auto repr() const -> std::string;
    auto as_array() const -> py::array_t<T>;
    auto as_vector() const -> std::vector<T>;

    auto extent(unsigned short d = 0) const -> std::size_t {
      return data.extent(d);
    }
  };

  template <class T>
  void pyDefineArray(py::module&);

} // namespace rgnr

#endif // CONTAINERS_ARRAY_HPP
