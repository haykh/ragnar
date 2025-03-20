#include "physics/ic.hpp"

#include "utils/snippets.h"

#include "containers/distributions.hpp"

#include <Kokkos_Core.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace rgnr {

  auto ICSpectrum(const TabulatedDistribution& dist_prtls,
                  const TabulatedDistribution& dist_soft_photons,
                  const Bins&                  bins_e_ic) -> Array1D<real_t> {
    py::print("Computing IC spectrum ...", "flush"_a = true);

    const auto nbins_prtls        = dist_prtls.extent();
    const auto nbins_soft_photons = dist_soft_photons.extent();
    const auto nbins_ic           = bins_e_ic.extent(0);

    py::print(
      " Launching",
      ToHumanReadable(nbins_prtls * nbins_soft_photons * nbins_ic, USE_POW10),
      "threads",
      "end"_a   = "",
      "flush"_a = true);

    const auto range_policy = Kokkos::MDRangePolicy<Kokkos::Rank<3>> {
      {           0,                  0,        0 },
      { nbins_prtls, nbins_soft_photons, nbins_ic }
    };
    auto eic_2_f_ic = Kokkos::View<real_t*> { "eic_2_f_ic", nbins_ic };
    auto eic_2_f_ic_scat = Kokkos::Experimental::create_scatter_view(eic_2_f_ic);
    Kokkos::parallel_for(
      "ICSpectrum",
      range_policy,
      ic::Kernel(dist_prtls, dist_soft_photons, bins_e_ic, eic_2_f_ic_scat));
    Kokkos::Experimental::contribute(eic_2_f_ic, eic_2_f_ic_scat);

    Kokkos::fence();
    py::print(": OK", "flush"_a = true);
    return eic_2_f_ic;
  }

  void pyDefineICSpectrum(py::module& m) {
    m.def("ICSpectrum",
          &ICSpectrum,
          "dist_prtls"_a,
          "dist_soft_photons"_a,
          "bins_e_ic"_a,
          R"rgnrdoc(
      Compute the IC spectrum for a given distribution

      Parameters
      ----------
      dist_prtls : TabulatedDistribution
        The distribution of particles (gamma*beta)

      dist_soft_photons : TabulatedDistribution
        The distribution of soft photons (energy in units of mec^2)

      bins_e_ic : Bins
        The energy bins for the IC spectrum

      Returns
      -------
      Array1D
        The IC spectrum: E^2 dN / dE
    )rgnrdoc");
  }

} // namespace rgnr
