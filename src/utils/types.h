#ifndef TYPES_H
#define TYPES_H

using real_t = float;
using dim_t  = unsigned short;

enum in {
  x = 0,
  y = 1,
  z = 1
};

constexpr bool IGNORE_COORDS = true;
constexpr bool USE_SUFFIX    = true;
constexpr bool USE_POW10     = false;

#endif // TYPES_H
