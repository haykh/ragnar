#ifndef TYPES_H
#define TYPES_H

using real_t = float;
using dim_t  = unsigned short;

enum in {
  x = 0,
  y = 1,
  z = 1
};

enum class Quantity {
  Energy,
};

enum class EnergyUnits {
  eV,
  MeV,
  GeV,
  mc2,
};

template <typename T>
struct Unit {
  Quantity quantity;
  T        unit;

  Unit(const Quantity& q, const T& unit) : quantity { q }, unit { unit } {}
};

constexpr bool IGNORE_COORDS = true;
constexpr bool USE_SUFFIX    = true;
constexpr bool USE_POW10     = false;
constexpr bool LOGGRID       = true;

#endif // TYPES_H
