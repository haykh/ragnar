#ifndef PHYSICS_IC_HPP
#define PHYSICS_IC_HPP

#include "utils/types.h"

#include "containers/particles.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <pybind11/pybind11.h>

#include <iostream>
#include <map>
#include <string>

namespace math = Kokkos;
namespace py   = pybind11;

namespace rgnr {

  namespace ic {

    KOKKOS_INLINE_FUNCTION
    auto KNfunc(real_t Gamma, real_t q) -> real_t {
      return 2 * q * math::log(q) + (1 + 2 * q) * (1 - q) +
             0.5 * (1 - q) * (Gamma * q) * (Gamma * q) / (1 + Gamma * q);
    }

    /*
     * Computes `E dN / d(ln E)` or `E dN / dE`
     * ... for IC radiation from a given distribution
     * - if eic_bins is logarithmically spaced --> `E^2 dN / dE`
     * - if eic_bins is linearly spaced --> `E dN / dE`
     */
    template <class S>
    class Kernel {
      const Kokkos::View<real_t*> m_gamma_bins;
      const Kokkos::View<real_t*> m_dn_dgamma;

      const S m_dn_desoft;

      const Kokkos::View<real_t*> m_esot_bins;

      const Kokkos::View<real_t*> m_eic_bins;

      Kokkos::Experimental::ScatterView<real_t*> m_eic2_dn_deic_scat;

      const real_t m_mc2;

    public:
      Kernel(const Kokkos::View<real_t*>&         prtl_energy_bins,
             const Kokkos::View<real_t*>&         prtl_energy_distribution,
             const DimensionalArray<EnergyUnits>& soft_photon_energy_bins,
             const S&                             soft_photon_distribution,
             const DimensionalArray<EnergyUnits>& photon_energy_bins,
             const Kokkos::Experimental::ScatterView<real_t*>& photon_spectrum,
             real_t                                            gamma_scale,
             const DimensionalQuantity<EnergyUnits>&           mc2)
        : m_prtl_energy_bins { prtl_energy_bins }
        , m_prtl_energy_distribution { prtl_energy_distribution }
        , m_soft_photon_distribution { soft_photon_distribution }
        , m_soft_photon_energy_bins { soft_photon_energy_bins.data }
        , m_photon_energy_bins { photon_energy_bins.data }
        , m_photon_spectrum { photon_spectrum }
        , m_gamma_scale { gamma_scale }
        , m_mc2 { mc2.value } {
        soft_photon_distribution.assertDimension(soft_photon_energy_bins.unit());
        if (photon_energy_bins.unit() != EnergyUnits::mc2) {
          throw std::runtime_error(
            "photon_energy_bins must be in units of mc^2");
        }
        if (mc2.unit() != soft_photon_energy_bins.unit()) {
          throw std::runtime_error(
            "mc2 must be in the same units as soft_photon_energy_bins");
        }
      }

      KOKKOS_INLINE_FUNCTION
      void operator()(std::size_t gidx, std::size_t eidx, std::size_t esidx) const {
        const auto gamma   = m_prtl_energy_bins(gidx) * m_gamma_scale + 1.0;
        const auto dn_prtl = m_prtl_energy_distribution(gidx);

        const auto eps_soft = m_soft_photon_energy_bins(esidx);
        const auto dns      = m_soft_photon_distribution.dn(eps_soft);

        const auto eps_mc2 = m_photon_energy_bins(eidx);

        const auto Gamma = gamma * m_mc2 / (4 * eps_soft);

        if (eps_mc2 > gamma * Gamma / (1 + Gamma)) {
          return;
        }
        const auto q = (eps_mc2 / gamma) / (Gamma * (1 - (eps_mc2 / gamma)));

        const auto KNval = KNfunc(Gamma, q);

        auto photon_spectrum_acc   = m_photon_spectrum.access();
        photon_spectrum_acc(eidx) += dn_prtl * (dns / eps_soft) * eps_mc2 *
                                     KNval / (gamma * gamma);
      }
    };

  } // namespace ic

  template <class S>
  auto ICSpectrum(const Kokkos::View<real_t*>&         prtl_energy_bins,
                  const Kokkos::View<real_t*>&         prtl_energy_distribution,
                  const DimensionalArray<EnergyUnits>& soft_photon_energy_bins,
                  const S&                             soft_photon_distribution,
                  const DimensionalArray<EnergyUnits>& photon_energy_bins,
                  real_t                               gamma_scale,
                  const DimensionalQuantity<EnergyUnits>& mc2)
    -> Kokkos::View<real_t*> {
    std::cout << "Computing IC spectrum ..." << std::endl;

    const auto nprtl_bins = prtl_energy_bins.extent(0);
    const auto nsoft_bins = soft_photon_energy_bins.data.extent(0);

    const auto nphoton_bins = photon_energy_bins.data.extent(0);
    auto photon_spectrum = Kokkos::View<real_t*> { "photon_spectrum", nphoton_bins };
    auto photon_spectrum_scat = Kokkos::Experimental::create_scatter_view(
      photon_spectrum);

    std::cout << "  Launching "
              << ToHumanReadable(nprtl_bins * nphoton_bins * nsoft_bins, USE_POW10)
              << " threads" << std::endl;
    const auto range_policy = Kokkos::MDRangePolicy<Kokkos::Rank<3>> {
      {          0,            0,          0 },
      { nprtl_bins, nphoton_bins, nsoft_bins }
    };
    Kokkos::parallel_for("ICSpectrum",
                         range_policy,
                         ic::Kernel<S>(prtl_energy_bins,
                                       prtl_energy_distribution,
                                       soft_photon_energy_bins,
                                       soft_photon_distribution,
                                       photon_energy_bins,
                                       photon_spectrum_scat,
                                       gamma_scale,
                                       mc2));
    Kokkos::Experimental::contribute(photon_spectrum, photon_spectrum_scat);

    Kokkos::fence();
    std::cout << "  Spectrum computed : OK" << std::endl;
    return photon_spectrum;
  }

} // namespace rgnr

#endif // PHYSICS_IC_HPP
