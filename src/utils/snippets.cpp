#include "utils/snippets.h"

#include "utils/array.h"
#include "utils/types.h"

#include <Kokkos_Core.hpp>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <map>
#include <string>
#include <vector>

namespace math = Kokkos;
namespace py   = pybind11;
using namespace pybind11::literals;

namespace rgnr {
  auto Linspace(real_t start, real_t stop, std::size_t num) -> std::vector<real_t> {
    if (start >= stop) {
      throw std::runtime_error("Linspace start must be < stop");
    }
    if (num == 0) {
      return {};
    } else if (num == 1) {
      return { start };
    }
    std::vector<real_t> result(num);
    for (std::size_t i = 0; i < num; ++i) {
      result[i] = start + static_cast<real_t>(i) * (stop - start) /
                            static_cast<real_t>(num - 1);
    }
    return result;
  }

  auto Logspace(real_t start, real_t stop, std::size_t num) -> std::vector<real_t> {
    if (start < 0 or stop < 0) {
      throw std::runtime_error("Logspace start and stop must be positive");
    }
    if (AlmostZero(start) or AlmostZero(stop)) {
      throw std::runtime_error("Logspace start and stop must be nonzero");
    }
    if (start >= stop) {
      throw std::runtime_error("Logspace start must be < stop");
    }
    if (num == 0) {
      return {};
    } else if (num == 1) {
      return { start };
    }
    std::vector<real_t> result(num);
    for (std::size_t i = 0; i < num; ++i) {
      result[i] = std::pow(
        10,
        std::log10(start) + static_cast<real_t>(i) *
                              (std::log10(stop) - std::log10(start)) /
                              static_cast<real_t>(num - 1));
    }
    return result;
  }

  auto LinspaceView(real_t start, real_t stop, std::size_t num) -> Array<real_t*> {
    if (start >= stop) {
      throw std::runtime_error("Linspace start must be < stop");
    }
    auto arr = Kokkos::View<real_t*> { "linspace", num };
    Kokkos::parallel_for(
      "Linspace",
      num,
      KOKKOS_LAMBDA(std::size_t i) {
        if (num == 1) {
          arr(i) = start;
        } else {
          arr(i) = start + i * (stop - start) / (num - 1);
        }
      });
    return arr;
  }

  auto LogspaceView(real_t start, real_t stop, std::size_t num) -> Array<real_t*> {
    auto arr = Kokkos::View<real_t*> { "logspace", num };
    if (start < 0 or stop < 0) {
      throw std::runtime_error("Logspace start and stop must be positive");
    }
    if (AlmostZero(start) or AlmostZero(stop)) {
      throw std::runtime_error("Logspace start and stop must be nonzero");
    }
    if (start >= stop) {
      throw std::runtime_error("Logspace start must be < stop");
    }
    Kokkos::parallel_for(
      "Logspace",
      num,
      KOKKOS_LAMBDA(std::size_t i) {
        if (num == 1) {
          arr(i) = start;
        } else {
          arr(i) = math::pow(10,
                             math::log10(start) + static_cast<real_t>(i) *
                                                    (math::log10(stop / start)) /
                                                    static_cast<real_t>(num - 1));
        }
      });
    return arr;
  }

  template <class T>
  auto ToHumanReadable(T value, bool use_suffixes) -> std::string {
    bool   negative;
    double value_double;
    if constexpr (std::is_signed_v<T>) {
      negative     = value < 0;
      value_double = static_cast<double>(negative ? -value : value);
    } else {
      negative     = false;
      value_double = static_cast<double>(value);
    }

    if (use_suffixes) {
      const auto suffixes = std::array<std::string, 9> { "p", "n", "μ", "m", "",
                                                         "k", "M", "G", "T" };
      std::size_t sidx    = 4;
      while (value_double < 0.01 or value_double >= 1000) {
        if (value_double < 0.01) {
          if (sidx == 0) {
            break;
          }
          value_double *= 1000;
          --sidx;
        } else {
          if (sidx == suffixes.size() - 1) {
            break;
          }
          value_double /= 1000;
          ++sidx;
        }
      }
      return (negative ? "-" : "") + std::format("{:.2f}", value_double) + " " +
             suffixes[sidx];
    } else {
      int pow = 0;
      while (value_double < 0.1 or value_double >= 10) {
        if (value_double < 0.1) {
          value_double *= 10;
          --pow;
        } else {
          value_double /= 10;
          ++pow;
        }
      }
      return (negative ? "-" : "") + std::format("{:.2f}", value_double) +
             "·10^" + std::to_string(pow);
    }
  }

  template <class T>
  auto ToShort(T value) -> std::string {
    bool negative;
    T    value_abs;
    if constexpr (std::is_signed_v<T>) {
      negative  = value < 0;
      value_abs = negative ? -value : value;
    } else {
      negative  = false;
      value_abs = value;
    }
    std::string value_str;
    if (value_abs >= 0.001 and value_abs <= 9999) {
      if (value_abs < 1) {
        value_str = std::to_string(value_abs);
        value_str.erase(value_str.find_last_not_of('0') + 1, std::string::npos);
        std::replace(value_str.begin(), value_str.end(), '.', 'p');
      } else {
        value_str = std::to_string((int)value_abs);
      }
    } else if (value_abs < 0.001) {
      int pow = 0;
      while (value_abs < 1) {
        value_abs *= 10;
        pow++;
      }
      value_str = std::to_string((int)value_abs) + "em" + std::to_string(pow);
    } else {
      int pow = 0;
      while (value_abs >= 10) {
        value_abs /= 10;
        pow++;
      }
      value_str = std::to_string((int)value_abs) + "e" + std::to_string(pow);
    }
    if (negative) {
      value_str = "m" + value_str;
    }
    return value_str;
  }

  auto TemplateReplace(const std::string& tmpl,
                       const std::map<std::string, real_t>& table) -> std::string {
    std::string result = tmpl;
    while (result.find('%') != std::string::npos) {
      const auto start = result.find('%');
      const auto end   = result.find('%', start + 1);
      if (start == std::string::npos or end == std::string::npos) {
        throw std::runtime_error("Invalid template string");
      }
      const auto key   = result.substr(start + 1, end - start - 1);
      const auto value = table.at(key);
      result.replace(start, end - start + 1, ToShort<real_t>(value));
    }
    return result;
  }

  void pyDefineLinLogSpaces(py::module& m) {
    m.def("Linspace", &Linspace, "start"_a, "stop"_a, "num"_a, R"rgnrdoc(
        Create a linearly spaced vector of `num` elements between `start` and `stop`

        Parameters
        ----------
        start : float
          The start of the range

        stop : float
          The end of the range

        num : int
          The number of elements in the range

        Returns
        -------
        list 
          A list of `num` elements linearly spaced between `start` and `stop`
          )rgnrdoc");
    m.def("Logspace", &Logspace, "start"_a, "stop"_a, "num"_a, R"rgnrdoc(
        Create a logarithmically spaced vector of `num` elements between `start` and `stop`

        Parameters
        ----------
        start : float
          The start of the range

        stop : float
          The end of the range

        num : int
          The number of elements in the range

        Returns
        -------
        list 
          A list of `num` elements logarithmically spaced between `start` and `stop`
        )rgnrdoc");

    m.def("LinspaceView", &LinspaceView, "start"_a, "stop"_a, "num"_a, R"rgnrdoc(
        Create a linearly spaced view of `num` elements between `start` and `stop`

        Parameters
        ----------
        start : float
          The start of the range

        stop : float
          The end of the range

        num : int
          The number of elements in the range

        Returns
        -------
        Array
          A view of `num` elements linearly spaced between `start` and `stop`
        )rgnrdoc");
    m.def("LogspaceView", &LogspaceView, "start"_a, "stop"_a, "num"_a, R"rgnrdoc(
        Create a logarithmically spaced view of `num` elements between `start` and `stop`

        Parameters
        ----------
        start : float
          The start of the range

        stop : float
          The end of the range

        num : int
          The number of elements in the range

        Returns
        -------
        Array
          A view of `num` elements logarithmically spaced between `start` and `stop`
        )rgnrdoc");
  }

  template auto ToHumanReadable<float>(float, bool) -> std::string;
  template auto ToHumanReadable<double>(double, bool) -> std::string;
  template auto ToHumanReadable<int>(int, bool) -> std::string;
  template auto ToHumanReadable<std::size_t>(std::size_t, bool) -> std::string;

  template auto ToShort<float>(float) -> std::string;
  template auto ToShort<double>(double) -> std::string;
  template auto ToShort<int>(int) -> std::string;
  template auto ToShort<std::size_t>(std::size_t) -> std::string;

} // namespace rgnr
