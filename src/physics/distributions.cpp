#include "physics/distributions.hpp"

namespace rgnr {

  void pyDefineBrokenPlawDistribution(py::module& m) {
    py::class_<BrokenPlawDistribution>(m, "BrokenPlawDistribution")
      .def(py::init<real_t, real_t, real_t>())
      .def_readonly("e_break", &BrokenPlawDistribution::e_break)
      .def_readonly("p1", &BrokenPlawDistribution::p1)
      .def_readonly("p2", &BrokenPlawDistribution::p2)
      .def_readwrite("unit", &BrokenPlawDistribution::unit)
      .doc() = R"rgnrdoc(
    A broken power-law distribution

    Parameters
    ----------
    e_break : float
      The break energy
    p1 : float
      The power-law index below the break energy
    p2 : float
      The power-law index above the break energy
    )rgnrdoc";
  }

} // namespace rgnr
