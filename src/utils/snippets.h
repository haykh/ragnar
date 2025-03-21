#ifndef UTILS_SNIPPETS_H
#define UTILS_SNIPPETS_H

#include "utils/global.h"

#include "containers/array.hpp"

#include <pybind11/pybind11.h>

#include <cmath>
#include <limits>
#include <string>
#include <type_traits>

namespace py = pybind11;

namespace rgnr {

  auto Linspace(real_t, real_t, std::size_t) -> Array1D<real_t>;
  auto Logspace(real_t, real_t, std::size_t) -> Array1D<real_t>;

  void pyDefineLinLogSpaces(py::module&);

  template <class T>
  inline auto AlmostEqual(T a, T b, T eps = std::numeric_limits<T>::epsilon()) -> bool {
    static_assert(std::is_floating_point_v<T>, "T must be a floating point type");
    return (a == b) ||
           (std::abs(a - b) <= std::min(std::abs(a), std::abs(b)) * eps);
  }

  template <class T>
  inline auto AlmostZero(T a, T eps = std::numeric_limits<T>::epsilon()) -> bool {
    static_assert(std::is_floating_point_v<T>, "T must be a floating point type");
    return std::abs(a) <= eps;
  }

  auto TemplateReplace(const std::string&,
                       const std::map<std::string, real_t>&) -> std::string;

  template <class T>
  auto ToHumanReadable(T, bool) -> std::string;

  template <class T>
  auto ToShort(T) -> std::string;

} // namespace rgnr

#endif // UTILS_SNIPPETS_H
