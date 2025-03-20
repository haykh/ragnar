#include "utils/global.h"
#include "utils/snippets.h"

#include "containers/array.hpp"
#include "containers/bins.hpp"
#include "containers/distributions.hpp"
#include "containers/particles.hpp"
#include "containers/tabulation.hpp"
#include "io/h5.hpp"
#include "physics/ic.hpp"
#include "physics/synchrotron.hpp"
#include "plugins/tristan-v2.hpp"

#include <Kokkos_Core.hpp>
#include <pybind11/pybind11.h>

PYBIND11_MAKE_OPAQUE(Kokkos::View<int*>);
PYBIND11_MAKE_OPAQUE(Kokkos::View<float*>);
PYBIND11_MAKE_OPAQUE(Kokkos::View<double*>);

namespace py = pybind11;

PYBIND11_MODULE(ragnar, m) {
  m.doc() = "Ragnar: A simple module for radiative post-processing of PIC data";

  // Kokkos interface
  m.def(
    "Initialize",
    []() {
      if (!Kokkos::is_initialized()) {
        Kokkos::initialize();
      } else {
        py::print("Kokkos is already initialized");
      }
    },
    "Initialize Kokkos");
  m.def(
    "Finalize",
    []() {
      if (Kokkos::is_initialized()) {
        Kokkos::finalize();
      } else {
        py::print("Kokkos is not initialized");
      }
    },
    "Finalize Kokkos");

  // utils
  rgnr::pyDefineUnits(m);

  rgnr::pyDefineLinLogSpaces(m);
  rgnr::pyDefineTabulatedFunction<true>(m);
  rgnr::pyDefineTabulatedFunction<false>(m);

  rgnr::pyDefineArray<int>(m);
  rgnr::pyDefineArray<float>(m);
  rgnr::pyDefineArray<double>(m);
  rgnr::pyDefineBins(m);

  // containers
  rgnr::pyDefineParticles<1>(m);
  rgnr::pyDefineParticles<2>(m);
  rgnr::pyDefineParticles<3>(m);

  // io
  rgnr::io::h5::pyDefineRead1DArray<int>(m);
  rgnr::io::h5::pyDefineRead1DArray<float>(m);
  rgnr::io::h5::pyDefineRead1DArray<double>(m);

  rgnr::io::h5::pyDefineWrite1DArray<int>(m);
  rgnr::io::h5::pyDefineWrite1DArray<float>(m);
  rgnr::io::h5::pyDefineWrite1DArray<double>(m);

  // plugins
  rgnr::pyDefineTristanV2Plugin<1>(m);
  rgnr::pyDefineTristanV2Plugin<2>(m);
  rgnr::pyDefineTristanV2Plugin<3>(m);

  // physics
  rgnr::pyDefineGenerators(m);
  rgnr::pyDefineSynchrotronSpectrum<1>(m);
  rgnr::pyDefineSynchrotronSpectrum<2>(m);
  rgnr::pyDefineSynchrotronSpectrum<3>(m);
  rgnr::pyDefineSynchrotronSpectrumFromDist(m);
  rgnr::pyDefineICSpectrum(m);
}
