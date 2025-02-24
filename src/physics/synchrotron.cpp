#include "physics/synchrotron.hpp"

#include "utils/snippets.h"
#include "utils/tabulation.h"
#include "utils/types.h"

#include "containers/dimensionals.hpp"
#include "containers/particles.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <cmath>
#include <iostream>
#include <vector>

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

    auto TabulateFfunc(std::size_t npoints, real_t xmin, real_t xmax)
      -> TabulatedFunction<true> {
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

  template <dim_t D>
  auto SynchrotronSpectrum(const Particles<D>& prtls,
                           const DimensionalArray<EnergyUnits>& photon_energy_bins,
                           real_t B0,
                           real_t gamma_syn,
                           const DimensionalQuantity<EnergyUnits>& eps_at_gamma_syn)
    -> Kokkos::View<real_t*> {
    std::cout << "Computing synchrotron spectrum for " << prtls.label()
              << " ..." << std::endl;
    const auto synchrotron_f_func = sync::TabulateFfunc();
    std::cout << "  Synchrotron kernel function tabulated : OK" << std::endl;

    const auto nphoton_bins = photon_energy_bins.data.extent(0);
    auto photon_spectrum = Kokkos::View<real_t*> { "photon_spectrum", nphoton_bins };
    auto photon_spectrum_scat = Kokkos::Experimental::create_scatter_view(
      photon_spectrum);

    std::cout << "  Launching "
              << ToHumanReadable(prtls.nactive() * nphoton_bins, USE_POW10)
              << " threads" << std::endl;
    const auto range_policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>> {
      {              0,            0},
      {prtls.nactive(), nphoton_bins}
    };
    Kokkos::parallel_for("SynchrotronSpectrum",
                         range_policy,
                         sync::Kernel<D>(prtls,
                                         synchrotron_f_func,
                                         photon_energy_bins,
                                         photon_spectrum_scat,
                                         B0,
                                         gamma_syn,
                                         eps_at_gamma_syn));
    Kokkos::Experimental::contribute(photon_spectrum, photon_spectrum_scat);

    std::cout << "  Spectrum computed : OK" << std::endl;
    return photon_spectrum;
  }

  template auto SynchrotronSpectrum(const Particles<1>&,
                                    const DimensionalArray<EnergyUnits>&,
                                    real_t,
                                    real_t,
                                    const DimensionalQuantity<EnergyUnits>&)
    -> Kokkos::View<real_t*>;

  template auto SynchrotronSpectrum(const Particles<2>&,
                                    const DimensionalArray<EnergyUnits>&,
                                    real_t,
                                    real_t,
                                    const DimensionalQuantity<EnergyUnits>&)
    -> Kokkos::View<real_t*>;

  template auto SynchrotronSpectrum(const Particles<3>&,
                                    const DimensionalArray<EnergyUnits>&,
                                    real_t,
                                    real_t,
                                    const DimensionalQuantity<EnergyUnits>&)
    -> Kokkos::View<real_t*>;
} // namespace rgnr
