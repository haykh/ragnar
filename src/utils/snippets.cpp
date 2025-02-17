#include "utils/snippets.h"

#include "utils/types.h"

#include <array>
#include <cmath>
#include <format>
#include <string>
#include <vector>

namespace math = Kokkos;

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

  auto LinspaceView(real_t start, real_t stop, std::size_t num)
    -> Kokkos::View<real_t*> {
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

  auto LogspaceView(real_t start, real_t stop, std::size_t num)
    -> Kokkos::View<real_t*> {
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
    const bool negative     = (value < 0);
    auto       value_double = static_cast<double>(negative ? -value : value);

    if (use_suffixes) {
      const auto suffixes = std::array<std::string, 9> { "p", "n", "μ", "m", "",
                                                         "k", "M", "G", "T" };
      std::size_t sidx = 4;
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

  template auto ToHumanReadable<float>(float, bool) -> std::string;
  template auto ToHumanReadable<double>(double, bool) -> std::string;
  template auto ToHumanReadable<int>(int, bool) -> std::string;
  template auto ToHumanReadable<std::size_t>(std::size_t, bool) -> std::string;

} // namespace rgnr
