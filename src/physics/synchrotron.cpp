#include "physics/synchrotron.hpp"

#include "utils/array.h"
#include "utils/snippets.h"
#include "utils/tabulation.h"
#include "utils/types.h"

#include "containers/particles.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <pybind11/pybind11.h>

#include <cmath>
#include <iostream>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;

#define FIVE_THIRD 1.6666666666666667

namespace rgnr {

  namespace sync {

    auto Ffunc_integrand(real_t x) -> real_t {
      if (x < 1e-5) {
        return 4.0 * M_PI / (std::sqrt(3.0) * std::tgamma(1.0 / 3.0)) *
               std::pow(0.5 * x, 1.0 / 3.0);
      } else if (x > 20) {
        return 0.0;
      } else {
        const auto          xs = Logspace(x, 20, 100);
        std::vector<real_t> ys(100);
        for (std::size_t i = 0; i < 100; ++i) {
          ys[i] = std::cyl_bessel_k(FIVE_THIRD, xs[i]);
        }
        real_t integrand = 0.0;
        for (std::size_t i = 0; i < 99; ++i) {
          integrand += 0.5 * (ys[i] + ys[i + 1]) * (xs[i + 1] - xs[i]);
        }
        return x * integrand;
      }
    }

    auto TabulateFfunc(std::size_t npoints,
                       real_t      xmin,
                       real_t      xmax) -> TabulatedFunction<true> {
      const auto          xs = Logspace(xmin, xmax, npoints);
      std::vector<real_t> ys(npoints);
      for (std::size_t i = 0u; i < npoints; ++i) {
        if (xs[i] < 1e-6 or ys[i] > 100.0) {
          ys[i] = 0.0;
        } else {
          ys[i] = Ffunc_integrand(xs[i]);
        }
      }
      return TabulatedFunction<true> { xs, ys };
    }

  } // namespace sync

  auto SynchrotronSpectrumFromDist(const Array<real_t*>& gbeta_bins,
                                   const Array<real_t*>& dn_dgbeta,
                                   const Array<real_t*>& esyn_bins,
                                   real_t                gamma_syn,
                                   real_t esyn_at_gamma_syn) -> Array<real_t*> {
    if (gbeta_bins.extent(0) != dn_dgbeta.extent(0)) {
      throw std::invalid_argument(
        "gbeta_bins and dn_dgbeta must have the same size");
    }

    py::print("Computing synchrotron spectrum from a distribution",
              "flush"_a = true);
    const auto synchrotron_f_func = sync::TabulateFfunc();

    const auto n_esyn_bins = esyn_bins.extent(0);
    auto esyn2_dn_desyn = Kokkos::View<real_t*> { "esyn2_dn_desyn", n_esyn_bins };
    auto esyn2_dn_desyn_scat = Kokkos::Experimental::create_scatter_view(
      esyn2_dn_desyn);

    const auto range_policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>> {
      {                    0,           0 },
      { gbeta_bins.extent(0), n_esyn_bins }
    };

    py::print(" Launching",
              ToHumanReadable(gbeta_bins.extent(0) * n_esyn_bins, USE_POW10),
              "threads",
              "end"_a   = "",
              "flush"_a = true);
    Kokkos::parallel_for("SynchrotronSpectrumFromDist",
                         range_policy,
                         sync::KernelFromDist(gbeta_bins,
                                              dn_dgbeta,
                                              synchrotron_f_func,
                                              esyn_bins,
                                              esyn2_dn_desyn_scat,
                                              gamma_syn,
                                              esyn_at_gamma_syn));
    Kokkos::Experimental::contribute(esyn2_dn_desyn, esyn2_dn_desyn_scat);

    Kokkos::fence();
    py::print(": OK", "flush"_a = true);
    return esyn2_dn_desyn;
  }

  template <dim_t D>
  auto SynchrotronSpectrum(const Particles<D>&   prtls,
                           const Array<real_t*>& esyn_bins,
                           real_t                B0,
                           real_t                gamma_syn,
                           real_t esyn_at_gamma_syn) -> Array<real_t*> {
    py::print("Computing synchrotron spectrum for", prtls.label(), "flush"_a = true);
    const auto synchrotron_f_func = sync::TabulateFfunc();
    const auto n_esyn_bins        = esyn_bins.extent(0);

    py::print(" Launching",
              ToHumanReadable(prtls.nactive() * n_esyn_bins, USE_POW10),
              "threads",
              "end"_a   = "",
              "flush"_a = true);

    const auto range_policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>> {
      {               0,           0 },
      { prtls.nactive(), n_esyn_bins }
    };
    auto esyn2_dn_desyn = Kokkos::View<real_t*> { "esyn2_dn_desyn", n_esyn_bins };
    auto esyn2_dn_desyn_scat = Kokkos::Experimental::create_scatter_view(
      esyn2_dn_desyn);
    Kokkos::parallel_for("SynchrotronSpectrum",
                         range_policy,
                         sync::Kernel<D>(prtls,
                                         synchrotron_f_func,
                                         esyn_bins,
                                         esyn2_dn_desyn_scat,
                                         B0,
                                         gamma_syn,
                                         esyn_at_gamma_syn));
    Kokkos::Experimental::contribute(esyn2_dn_desyn, esyn2_dn_desyn_scat);

    Kokkos::fence();
    py::print(": OK", "flush"_a = true);
    return esyn2_dn_desyn;
  }

  void pyDefineSynchrotronSpectrumFromDist(py::module& m) {
    m.def("SynchrotronSpectrumFromDist",
          &SynchrotronSpectrumFromDist,
          "gbeta_bins"_a,
          "dn_dgbeta"_a,
          "esyn_bins"_a,
          "gamma_syn"_a,
          "esyn_at_gamma_syn"_a,
          R"rgnrdoc(
      Compute the synchrotron spectrum from a distribution of particles
      Assumes a constant magnetic field strength

      Parameters
      ----------
      gbeta_bins : Array
        The 4-velocity bins for particles

      dn_dgbeta : Array
        The distribution of particles

      esyn_bins : Array
        The energy bins for the spectrum, E, logarithmically spaced

      gamma_syn : float
        Synchrotron burnoff Lorentz factor

      esyn_at_gamma_syn : float
        Peak energy at the synchrotron burnoff (units of mc^2)

      Returns
      -------
      Array
        The synchrotron spectrum: E^2 dN / dE
      )rgnrdoc");
  }

  template <dim_t D>
  void pyDefineSynchrotronSpectrum(py::module& m) {
    m.def(("SynchrotronSpectrum_" + std::to_string(D) + "D").c_str(),
          &SynchrotronSpectrum<D>,
          "prtls"_a,
          "esyn_bins"_a,
          "B0"_a,
          "gamma_syn"_a,
          "esyn_at_gamma_syn"_a,
          R"rgnrdoc(
      Compute the synchrotron spectrum for a set of particles

      Parameters
      ----------
      prtls : Particles
        The particles to compute the spectrum for

      esyn_bins : Array
        The energy bins for the spectrum, E, logarithmically spaced

      B0 : float
        The magnetic field strength to which B in particles should be normalized

      gamma_syn : float
        Synchrotron burnoff Lorentz factor (w.r.t. B0)

      esyn_at_gamma_syn : float
        Peak energy at the synchrotron burnoff in the field of B0 (units of mc^2)

      Returns
      -------
      Array
        The synchrotron spectrum: E^2 dN / dE
      )rgnrdoc");
  }

  template auto SynchrotronSpectrum(const Particles<1>&,
                                    const Array<real_t*>&,
                                    real_t,
                                    real_t,
                                    real_t) -> Array<real_t*>;

  template auto SynchrotronSpectrum(const Particles<2>&,
                                    const Array<real_t*>&,
                                    real_t,
                                    real_t,
                                    real_t) -> Array<real_t*>;

  template auto SynchrotronSpectrum(const Particles<3>&,
                                    const Array<real_t*>&,
                                    real_t,
                                    real_t,
                                    real_t) -> Array<real_t*>;

  template void pyDefineSynchrotronSpectrum<1>(py::module&);
  template void pyDefineSynchrotronSpectrum<2>(py::module&);
  template void pyDefineSynchrotronSpectrum<3>(py::module&);

} // namespace rgnr
