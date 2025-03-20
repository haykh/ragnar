#include "physics/synchrotron.hpp"

#include "utils/global.h"
#include "utils/snippets.h"

#include "containers/array.hpp"
#include "containers/distributions.hpp"
#include "containers/particles.hpp"
#include "containers/tabulation.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <pybind11/pybind11.h>

#include <cmath>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;

#define ONE_THIRD  0.3333333333333333
#define FIVE_THIRD 1.6666666666666667

namespace rgnr {

  namespace sync {

    auto Ffunc_integrand(real_t x) -> real_t {
      if (x < 1e-5) {
        return 4.0 * M_PI / (std::sqrt(3.0) * std::tgamma(ONE_THIRD)) *
               std::pow(0.5 * x, 1.0 / 3.0);
      } else if (x > 20) {
        return 0.0;
      } else {
        const auto          xs = Logspace(x, 20, 100).as_vector();
        std::vector<real_t> ys(100);
        for (auto i = 0u; i < 100u; ++i) {
          ys[i] = std::cyl_bessel_k(FIVE_THIRD, xs[i]);
        }
        real_t integrand = 0.0;
        for (auto i = 0u; i < 99u; ++i) {
          integrand += 0.5 * (ys[i] + ys[i + 1]) * (xs[i + 1] - xs[i]);
        }
        return x * integrand;
      }
    }

    auto TabulateFfunc(std::size_t npoints,
                       real_t      xmin,
                       real_t      xmax) -> TabulatedFunction<true> {
      const auto xs_arr = Logspace(xmin, xmax, npoints).data;
      auto       xs_h   = Kokkos::create_mirror_view(xs_arr);
      Kokkos::deep_copy(xs_h, xs_arr);
      std::vector<real_t> ys(npoints);
      std::vector<real_t> xs(npoints);
      for (auto i = 0u; i < npoints; ++i) {
        xs[i] = xs_h(i);
        if (xs[i] < 1e-6 or ys[i] > 100.0) {
          ys[i] = 0.0;
        } else {
          ys[i] = Ffunc_integrand(xs[i]);
        }
      }
      return TabulatedFunction<true> { xs, ys };
    }

  } // namespace sync

  auto SynchrotronSpectrumFromDist(const TabulatedDistribution& dist_prtls,
                                   const Bins&                  bins_e_syn,
                                   real_t                       g_syn,
                                   real_t e_syn_at_g_syn) -> Array1D<real_t> {
    py::print("Computing synchrotron spectrum from a distribution",
              "flush"_a = true);
    const auto fkernel_sync = sync::TabulateFfunc();

    const auto nbins_e_syn = bins_e_syn.extent(0);
    auto e_syn_2_f_syn = Kokkos::View<real_t*> { "e_syn_2_f_syn", nbins_e_syn };
    auto e_syn_2_f_syn_scat = Kokkos::Experimental::create_scatter_view(
      e_syn_2_f_syn);

    const auto range_policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>> {
      {                   0,           0 },
      { dist_prtls.extent(), nbins_e_syn }
    };

    py::print(" Launching",
              ToHumanReadable(dist_prtls.extent() * nbins_e_syn, USE_POW10),
              "threads",
              "end"_a   = "",
              "flush"_a = true);
    Kokkos::parallel_for("SynchrotronSpectrum",
                         range_policy,
                         sync::KernelFromDist(dist_prtls,
                                              fkernel_sync,
                                              bins_e_syn,
                                              e_syn_2_f_syn_scat,
                                              g_syn,
                                              e_syn_at_g_syn));
    Kokkos::Experimental::contribute(e_syn_2_f_syn, e_syn_2_f_syn_scat);

    Kokkos::fence();
    py::print(": OK", "flush"_a = true);
    return e_syn_2_f_syn;
  }

  template <dim_t D>
  auto SynchrotronSpectrum(const Particles<D>& prtls,
                           const Bins&         bins_e_syn,
                           real_t              B0,
                           real_t              g_syn,
                           real_t e_syn_at_g_syn) -> Array1D<real_t> {
    py::print("Computing synchrotron spectrum for", prtls.label(), "flush"_a = true);
    const auto fkernel_sync = sync::TabulateFfunc();
    const auto nbins_e_syn  = bins_e_syn.extent(0);

    py::print(" Launching",
              ToHumanReadable(prtls.nactive() * nbins_e_syn, USE_POW10),
              "threads",
              "end"_a   = "",
              "flush"_a = true);

    const auto range_policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>> {
      {               0,           0 },
      { prtls.nactive(), nbins_e_syn }
    };
    auto e_syn_2_f_syn = Kokkos::View<real_t*> { "e_syn_2_f_syn", nbins_e_syn };
    auto e_syn_2_f_syn_scat = Kokkos::Experimental::create_scatter_view(
      e_syn_2_f_syn);
    Kokkos::parallel_for("SynchrotronSpectrum",
                         range_policy,
                         sync::Kernel<D>(prtls,
                                         fkernel_sync,
                                         bins_e_syn,
                                         e_syn_2_f_syn_scat,
                                         B0,
                                         g_syn,
                                         e_syn_at_g_syn));
    Kokkos::Experimental::contribute(e_syn_2_f_syn, e_syn_2_f_syn_scat);

    Kokkos::fence();
    py::print(": OK", "flush"_a = true);
    return e_syn_2_f_syn;
  }

  void pyDefineSynchrotronSpectrumFromDist(py::module& m) {
    m.def("Ffunc_integrand", &sync::Ffunc_integrand, "x"_a);

    m.def("SynchrotronSpectrumFromDist",
          &SynchrotronSpectrumFromDist,
          "dist_prtls"_a,
          "bins_e_syn"_a,
          "g_syn"_a,
          "e_syn_at_g_syn"_a,
          R"rgnrdoc(
      Compute the synchrotron spectrum from a distribution of particles
      Assumes a constant magnetic field strength

      Parameters
      ----------
      dist_prtls : TabulatedDistribution
        The distribution of particles

      bins_e_syn : Bins
        The energy bins for the spectrum

      g_syn : float
        Synchrotron burnoff Lorentz factor

      e_syn_at_g_syn : float
        Peak energy at the synchrotron burnoff (units of mc^2)

      Returns
      -------
      Array1D
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

      bins_e_syn : Bins
        The energy bins for the spectrum

      B0 : float
        The magnetic field strength to which B in particles should be normalized

      g_syn : float
        Synchrotron burnoff Lorentz factor

      e_syn_at_g_syn : float
        Peak energy at the synchrotron burnoff (units of mc^2)

      Returns
      -------
      Array1D
        The synchrotron spectrum: E^2 dN / dE
      )rgnrdoc");
  }

  template auto SynchrotronSpectrum(const Particles<1>&,
                                    const Bins&,
                                    real_t,
                                    real_t,
                                    real_t) -> Array1D<real_t>;

  template auto SynchrotronSpectrum(const Particles<2>&,
                                    const Bins&,
                                    real_t,
                                    real_t,
                                    real_t) -> Array1D<real_t>;

  template auto SynchrotronSpectrum(const Particles<3>&,
                                    const Bins&,
                                    real_t,
                                    real_t,
                                    real_t) -> Array1D<real_t>;

  template void pyDefineSynchrotronSpectrum<1>(py::module&);
  template void pyDefineSynchrotronSpectrum<2>(py::module&);
  template void pyDefineSynchrotronSpectrum<3>(py::module&);

} // namespace rgnr
