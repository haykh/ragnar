#ifndef CONTAINERS_DIMENSIONALS_HPP
#define CONTAINERS_DIMENSIONALS_HPP

#include "utils/types.h"

#include <Kokkos_Core.hpp>

namespace rgnr {

  template <typename T>
  struct DimensionalQuantity {
    Unit<T> dimension;

    const real_t value;

    DimensionalQuantity(const Unit<T>& dimension, real_t value)
      : dimension { dimension }
      , value { value } {}

    DimensionalQuantity(const Quantity& q, const T& unit, real_t value)
      : dimension { q, unit }
      , value { value } {}

    auto unit() const -> const T& {
      return dimension.unit;
    }
  };

  template <typename T>
  struct DimensionalArray {
    Unit<T> dimension;

    const Kokkos::View<real_t*> data;

    DimensionalArray(const Unit<T>& dimension, const Kokkos::View<real_t*>& data)
      : dimension { dimension }
      , data { data } {}

    DimensionalArray(const Quantity&              q,
                     const T&                     unit,
                     const Kokkos::View<real_t*>& data)
      : dimension { q, unit }
      , data { data } {}

    auto unit() const -> const T& {
      return dimension.unit;
    }
  };

} // namespace rgnr

#endif // CONTAINERS_DIMENSIONALS_HPP
