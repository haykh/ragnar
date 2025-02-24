#ifndef UTILS_SNIPPETS_H
#define UTILS_SNIPPETS_H

#include "utils/types.h"

#include <Kokkos_Core.hpp>

#include <cmath>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace rgnr {

  auto Linspace(real_t start, real_t stop, std::size_t num) -> std::vector<real_t>;
  auto Logspace(real_t start, real_t stop, std::size_t num) -> std::vector<real_t>;

  auto LinspaceView(real_t start, real_t stop, std::size_t num)
    -> Kokkos::View<real_t*>;
  auto LogspaceView(real_t start, real_t stop, std::size_t num)
    -> Kokkos::View<real_t*>;

  template <class T>
  inline auto AlmostEqual(T a, T b, T eps = std::numeric_limits<T>::epsilon())
    -> bool {
    static_assert(std::is_floating_point_v<T>, "T must be a floating point type");
    return (a == b) ||
           (std::abs(a - b) <= std::min(std::abs(a), std::abs(b)) * eps);
  }

  template <class T>
  inline auto AlmostZero(T a, T eps = std::numeric_limits<T>::epsilon()) -> bool {
    static_assert(std::is_floating_point_v<T>, "T must be a floating point type");
    return std::abs(a) <= eps;
  }

  auto TemplateReplace(const std::string&, const std::map<std::string, real_t>&)
    -> std::string;

  template <class T>
  auto ToHumanReadable(T, bool) -> std::string;

  template <class T>
  auto ToShort(T) -> std::string;

} // namespace rgnr

#endif // UTILS_SNIPPETS_H
