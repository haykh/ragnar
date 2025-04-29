#ifndef UTILS_SNIPPETS_H
#define UTILS_SNIPPETS_H

#include "utils/global.h"

#include "containers/array.hpp"

#include <pybind11/pybind11.h>

#include <cmath>
#include <limits>
#include <string>
#include <type_traits>
#include <sstream>
#include <stdexcept>
#include <memory>

namespace py = pybind11;

namespace rgnr {

namespace fmt {

  template <typename... Args>
    inline auto format(const char* format, Args... args) -> std::string {
      auto size_s = std::snprintf(nullptr, 0, format, args...) + 1;
      if (size_s <= 0) {
        throw std::runtime_error("Error during formatting.");
      }
      auto                    size { static_cast<std::size_t>(size_s) };
      std::unique_ptr<char[]> buf(new char[size]);
      std::snprintf(buf.get(), size, format, args...);
      return std::string(buf.get(), buf.get() + size - 1);
    }
  }

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
