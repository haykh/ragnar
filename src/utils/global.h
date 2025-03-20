#ifndef UTILS_GLOBAL_H
#define UTILS_GLOBAL_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

#include <string>
#include <utility>

using real_t  = float;
using dim_t   = unsigned short;
using slice_t = std::pair<std::size_t, std::size_t>;

enum in {
  x = 0,
  y = 1,
  z = 1
};

struct EnergyUnits {
  inline static const std::string eV   = "eV";
  inline static const std::string MeV  = "MeV";
  inline static const std::string GeV  = "GeV";
  inline static const std::string mec2 = "mec2";
  inline static const std::string mpc2 = "mpc2";
};

constexpr bool IGNORE_COORDS = true;
constexpr bool USE_SUFFIX    = true;
constexpr bool USE_POW10     = false;
constexpr bool LOGGRID       = true;

namespace rgnr {

  void pyDefineUnits(py::module&);

} // namespace rgnr

#endif // UTILS_GLOBAL_H
