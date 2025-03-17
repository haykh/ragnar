#include "utils/array.h"

#include "utils/types.h"

#include <Kokkos_Core.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;

namespace rgnr {

  template <class T>
  void Array<T>::head(std::size_t n, std::size_t start) const {
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
  auto Array<T>::repr() const -> std::string {
    std::string size_str = "";
    for (auto i = 0u; i < data.rank(); ++i) {
      size_str += std::to_string(data.extent(i));
      if (i < data.rank() - 1) {
        size_str += " x ";
      }
    }
    return std::to_string(data.rank()) + "D Array [ size: " + size_str + " ]" +
           (unit.empty() ? "" : " [ unit: " + unit + " ]");
  }

  template <class T>
  auto Array<T>::as_array() const -> py::array_t<std::remove_pointer_t<T>> {
    static_assert(std::is_pointer_v<T>, "T must be a pointer type");
    static_assert(!std::is_pointer_v<std::remove_pointer_t<T>>,
                  "T must be single a pointer type");
    auto host_view = Kokkos::create_mirror_view(data);
    Kokkos::deep_copy(host_view, data);
    return py::array_t<std::remove_pointer_t<T>>(
      { host_view.extent(0) },
      { sizeof(std::remove_pointer_t<T>) },
      host_view.data());
  }

  template <class T>
  void pyDefineArray(py::module& m, const std::string& variant, unsigned short dim) {
    py::class_<Array<T>>(m, ("Array" + std::to_string(dim) + "D_" + variant).c_str())
      .def(py::init<const std::string&>(), "unit"_a = "")
      .def("head", &Array<T>::head, "n"_a = 10, "start"_a = 0, R"rgnrdoc(
        Print the first n elements of the array starting from index start

        Parameters
        ----------
        n : int
          The number of elements to print

        start : int
          The index from which to start printing
      )rgnrdoc")
      .def("__repr__", &Array<T>::repr)
      .def("as_array", &Array<T>::as_array, R"rgnrdoc(
        Return the array as a numpy array

        Returns
        -------
        np.ndarray
      )rgnrdoc")
      .def("extent", &Array<T>::extent, "d"_a = 0, R"rgnrdoc(
        Return the extent of the array along dimension d

        Parameters
        ----------
        d : int
          The dimension along which to return the extent
      )rgnrdoc")
      .def_readwrite("unit", &Array<T>::unit)
      .doc() = R"rgnrdoc(
        A simple wrapper around a Kokkos View

        Attributes
        ----------
        unit : str
          Physical units of the array (if any)

        Methods
        -------
        head(n=10, start=0)
          Print the first n elements of the array starting from index start
      
        as_vector() -> list
          Return the array as a Python list
        
        extent(d=0) -> int
          Return the extent of the array along dimension d
      )rgnrdoc";
  }

  template struct Array<int*>;
  template struct Array<float*>;
  template struct Array<double*>;

  template void pyDefineArray<int*>(py::module&, const std::string&, unsigned short);
  template void pyDefineArray<float*>(py::module&, const std::string&, unsigned short);
  template void pyDefineArray<double*>(py::module&,
                                       const std::string&,
                                       unsigned short);

} // namespace rgnr
