#include "containers/array.hpp"

#include "utils/global.h"

#include <Kokkos_Core.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;

namespace rgnr {

  template <class T>
  void Array1D<T>::head(std::size_t n, std::size_t start) const {
    if (start + n > data.extent(0)) {
      throw std::range_error("Array::head: n > data.extent(0)");
    }
    n              = std::min(n, data.extent(0) - start);
    auto host_view = Kokkos::create_mirror_view(
      Kokkos::subview(data, slice_t { start, start + n }));
    Kokkos::deep_copy(host_view,
                      Kokkos::subview(data, slice_t { start, start + n }));
    for (auto i = 0u; i < n; ++i) {
      py::print(host_view(i), " ", "end"_a = "");
    }
    py::print();
  }

  template <class T>
  auto Array1D<T>::repr() const -> std::string {
    std::string size_str = "";
    for (auto i = 0u; i < data.rank(); ++i) {
      size_str += std::to_string(data.extent(i));
      if (i < data.rank() - 1) {
        size_str += " x ";
      }
    }
    return std::to_string(data.rank()) + "D Array [ size: " + size_str + " ]";
  }

  template <class T>
  auto Array1D<T>::as_array() const -> py::array_t<T> {
    auto host_view = Kokkos::create_mirror_view(data);
    Kokkos::deep_copy(host_view, data);
    return py::array_t<T>({ host_view.extent(0) }, { sizeof(T) }, host_view.data());
  }

  template <class T>
  auto Array1D<T>::as_vector() const -> std::vector<T> {
    auto host_view = Kokkos::create_mirror_view(data);
    Kokkos::deep_copy(host_view, data);
    std::vector<T> vec(host_view.extent(0));
    for (auto i = 0u; i < host_view.extent(0); ++i) {
      vec[i] = host_view(i);
    }
    return vec;
  }

  template <class T>
  void pyDefineArray(py::module& m) {
    py::class_<Array1D<T>>(m, ("Array1D_" + std::string(typeid(T).name())).c_str())
      .def(py::init<>())
      .def(py::init<const py::array_t<T>&>(), "arr"_a)
      .def("head", &Array1D<T>::head, "n"_a = 10, "start"_a = 0, R"rgnrdoc(
        Print the first n elements of the array starting from index start

        Parameters
        ----------
        n : int
          The number of elements to print

        start : int
          The index from which to start printing
      )rgnrdoc")
      .def("__repr__", &Array1D<T>::repr)
      .def("as_array", &Array1D<T>::as_array, R"rgnrdoc(
        Return the array as a numpy array

        Returns
        -------
        np.ndarray
      )rgnrdoc")
      .def("extent", &Array1D<T>::extent, "d"_a = 0, R"rgnrdoc(
        Return the extent of the array along dimension d

        Parameters
        ----------
        d : int
          The dimension along which to return the extent
      )rgnrdoc")
      .doc() = R"rgnrdoc(
        A simple wrapper around a 1D Kokkos View

        Methods
        -------
        head(n=10, start=0)
          Print the first n elements of the array starting from index start
      
        as_array() -> np.ndarray
          Return a numpy array

        extent(d=0) -> int
          Return the extent of the array along dimension d
      )rgnrdoc";
  }

  template struct Array1D<int>;
  template struct Array1D<float>;
  template struct Array1D<double>;

  template void pyDefineArray<int>(py::module&);
  template void pyDefineArray<float>(py::module&);
  template void pyDefineArray<double>(py::module&);

} // namespace rgnr
