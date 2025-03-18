#ifndef PHYSICS_IC_HPP
#define PHYSICS_IC_HPP

#include "utils/array.h"
#include "utils/types.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <pybind11/pybind11.h>

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

      const S                     m_dn_desoft;
      const Kokkos::View<real_t*> m_esoft_bins_mc2;
      const Kokkos::View<real_t*> m_eic_bins_mc2;

      Kokkos::Experimental::ScatterView<real_t*> m_eic2_dn_deic_scat;

    public:
      Kernel(const Array<real_t*>& gamma_bins,
             const Array<real_t*>& dn_dgamma,
             const Array<real_t*>& esoft_bins,
             const S&              dn_desoft_func,
             const Array<real_t*>& eic_bins,
             const Kokkos::Experimental::ScatterView<real_t*>& eic2_dn_deic_scat)
        : m_gamma_bins { gamma_bins.data }
        , m_dn_dgamma { dn_dgamma.data }
        , m_dn_desoft { dn_desoft_func }
        , m_esoft_bins_mc2 { esoft_bins.data }
        , m_eic_bins_mc2 { eic_bins.data }
        , m_eic2_dn_deic_scat { eic2_dn_deic_scat } {
        // dn_desoft_func.AssertEnergyUnits(esoft_bins.unit);
        if (dn_desoft_func.unit != esoft_bins.unit) {
          throw std::runtime_error(
            "dn_desoft_func must be in the same units as esoft_bins");
        }
        if (eic_bins.unit != EnergyUnits::mec2) {
          throw std::runtime_error("eic_bins must be in units of mc^2");
        }
        if (gamma_bins.extent(0) != dn_dgamma.extent(0)) {
          throw std::runtime_error(
            "gamma_bins and dn_dgamma must have the same size");
        }
      }

      KOKKOS_INLINE_FUNCTION
      void operator()(std::size_t gidx, std::size_t eidx, std::size_t esidx) const {
        const auto gamma = m_gamma_bins(gidx);
        const auto dn    = m_dn_dgamma(gidx);

        const auto esoft_mc2 = m_esoft_bins_mc2(esidx);
        const auto dn_desoft = m_dn_desoft.dn(esoft_mc2);

        const auto eic_mc2 = m_eic_bins_mc2(eidx);

        const auto Gamma = gamma / (4 * esoft_mc2);

        if (eic_mc2 > gamma * Gamma / (1 + Gamma)) {
          return;
        }
        const auto q = (eic_mc2 / gamma) / (Gamma * (1 - (eic_mc2 / gamma)));

        const auto KNval = KNfunc(Gamma, q);

        auto eic2_dn_deic_acc   = m_eic2_dn_deic_scat.access();
        eic2_dn_deic_acc(eidx) += dn * (dn_desoft / esoft_mc2) * eic_mc2 *
                                  KNval / (gamma * gamma);
      }
    };

  } // namespace ic

  template <class S>
  auto ICSpectrumFromDist(const Array<real_t*>&,
                          const Array<real_t*>&,
                          const Array<real_t*>&,
                          const S&,
                          const Array<real_t*>&) -> Array<real_t*>;

  void pyDefineICSpectrumFromDist(py::module&);

} // namespace rgnr

#endif // PHYSICS_IC_HPP
