#include "physics/ic.hpp"

#include "utils/snippets.h"

#include "physics/distributions.hpp"

#include <Kokkos_Core.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace rgnr {

  template <class S>
  auto ICSpectrumFromDist(const Array<real_t*>& gamma_bins,
                          const Array<real_t*>& dn_dgamma,
                          const Array<real_t*>& esoft_bins,
                          const S&              dn_desoft,
                          const Array<real_t*>& eic_bins) -> Array<real_t*> {
    py::print("Computing IC spectrum ...", "flush"_a = true);

    const auto nprtl_bins   = gamma_bins.extent(0);
    const auto nsoft_bins   = esoft_bins.extent(0);
    const auto nphoton_bins = eic_bins.extent(0);

    py::print(" Launching ",
              ToHumanReadable(nprtl_bins * nphoton_bins * nsoft_bins, USE_POW10),
              "threads",
              "end"_a   = "",
              "flush"_a = true);

    const auto range_policy = Kokkos::MDRangePolicy<Kokkos::Rank<3>> {
      {          0,            0,          0 },
      { nprtl_bins, nphoton_bins, nsoft_bins }
    };
    auto eic2_dn_deic = Kokkos::View<real_t*> { "eic2_dn_deic", nphoton_bins };
    auto eic2_dn_deic_scat = Kokkos::Experimental::create_scatter_view(eic2_dn_deic);
    Kokkos::parallel_for("ICSpectrum",
                         range_policy,
                         ic::Kernel<S>(gamma_bins,
                                       dn_dgamma,
                                       esoft_bins,
                                       dn_desoft,
                                       eic_bins,
                                       eic2_dn_deic_scat));
    Kokkos::Experimental::contribute(eic2_dn_deic, eic2_dn_deic_scat);

    Kokkos::fence();
    py::print(": OK", "flush"_a = true);
    return eic2_dn_deic;
  }

  void pyDefineICSpectrumFromDist(py::module& m) {
    m.def("ICSpectrumFromBrokenPlaw",
          &ICSpectrumFromDist<BrokenPlawDistribution>,
          "gamma_bins"_a,
          "dn_dgamma"_a,
          "esoft_bins"_a,
          "dn_desoft"_a,
          "eic_bins"_a,
          R"rgnrdoc(
      Compute the IC spectrum for a given distribution

      Parameters
      ----------
      gamma_bins : Array
        The Lorentz factor bins for the particle distribution

      dn_dgamma : Array
        The distribution of particles in gamma_bins

      esoft_bins : Array
        The energy bins for the soft photon distribution

      dn_desoft : BrokenPlawDistribution
        The distribution of soft photons

      eic_bins : Array
        The energy bins for the IC spectrum

      Returns
      -------
      Array
        The IC spectrum: E^2 dN / dE (or E dN / dE if E bins are linear)
    )rgnrdoc");
  }

} // namespace rgnr
