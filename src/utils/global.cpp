#include "utils/global.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace rgnr {

  void pyDefineUnits(py::module& m) {
    py::class_<EnergyUnits>(m, "EnergyUnits")
      .def_readonly_static("eV", &EnergyUnits::eV)
      .def_readonly_static("MeV", &EnergyUnits::MeV)
      .def_readonly_static("GeV", &EnergyUnits::GeV)
      .def_readonly_static("mec2", &EnergyUnits::mec2)
      .def_readonly_static("mpc2", &EnergyUnits::mpc2);
  }

} // namespace rgnr
