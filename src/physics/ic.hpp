#ifndef PHYSICS_IC_HPP
#define PHYSICS_IC_HPP

#include "utils/global.h"

#include "containers/array.hpp"
#include "containers/distributions.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <pybind11/pybind11.h>

namespace math = Kokkos;
namespace py   = pybind11;

namespace rgnr {

  namespace ic {

    KOKKOS_INLINE_FUNCTION
    auto KNfunc(real_t Gamma, real_t q) -> real_t {
      const real_t Gq = Gamma * q;
      return 2 * q * math::log(q) +
             (1 - q) * ((1 + 2 * q) + 0.5 * (Gq) * (Gq) / (1 + Gq));
    }

    /*
     * Computes `E_ic^2 f(E_ic)`
     */
    class Kernel {
      const Kokkos::View<real_t*>                m_bins_g_prtls;
      const Kokkos::View<real_t*>                m_f_prtls;
      const bool                                 m_islog_bins_prtls;
      const Kokkos::View<real_t*>                m_bins_e_soft_photons;
      const Kokkos::View<real_t*>                m_f_soft_photons;
      const Kokkos::View<real_t*>                m_bins_e_ic;
      Kokkos::Experimental::ScatterView<real_t*> m_e_ic_2_f_ic_scat;

    public:
      Kernel(const TabulatedDistribution& dist_prtls,
             const TabulatedDistribution& dist_soft_photons,
             const Bins&                  bins_e_ic,
             const Kokkos::Experimental::ScatterView<real_t*>& e_ic_2_f_ic_scat)
        : m_bins_g_prtls { dist_prtls.EnergyBins().data }
        , m_f_prtls { dist_prtls.F().data }
        , m_islog_bins_prtls { dist_prtls.log_spaced() }
        , m_bins_e_soft_photons { dist_soft_photons.EnergyBins().data }
        , m_f_soft_photons { dist_soft_photons.F().data }
        , m_bins_e_ic { bins_e_ic.data }
        , m_e_ic_2_f_ic_scat { e_ic_2_f_ic_scat } {
        if (dist_soft_photons.EnergyBins().unit != EnergyUnits::mec2) {
          throw std::runtime_error(
            "Soft photons energy bins must be in units of mec^2");
        }
        if (bins_e_ic.unit != EnergyUnits::mec2) {
          throw std::runtime_error("E_ic must be in units of mec^2");
        }
      }

      KOKKOS_INLINE_FUNCTION
      void operator()(std::size_t gidx, std::size_t eidx, std::size_t esidx) const {
        const auto g_prtls = m_bins_g_prtls(gidx);
        const auto f_prtls = m_f_prtls(gidx);

        const auto e_soft_photons = m_bins_e_soft_photons(esidx);
        const auto f_soft_photons = m_f_soft_photons(esidx);

        const auto e_ic = m_bins_e_ic(eidx);

        const auto Gamma = 4 * g_prtls * e_soft_photons;

        if (e_ic > g_prtls * Gamma / (1 + Gamma)) {
          return;
        }
        const auto q = (e_ic / g_prtls) / (Gamma * (1 - (e_ic / g_prtls)));

        const auto KNval = KNfunc(Gamma, q);

        auto e_ic_2_f_ic_acc = m_e_ic_2_f_ic_scat.access();
        if (m_islog_bins_prtls) {
          e_ic_2_f_ic_acc(eidx) += f_prtls * (f_soft_photons / e_soft_photons) *
                                   e_ic * e_ic * KNval / g_prtls;
        } else {
          e_ic_2_f_ic_acc(eidx) += f_prtls * (f_soft_photons / e_soft_photons) *
                                   e_ic * e_ic * KNval / (g_prtls * g_prtls);
        }
      }
    };

  } // namespace ic

  auto ICSpectrum(const TabulatedDistribution&,
                  const TabulatedDistribution&,
                  const Bins&) -> Array1D<real_t>;

  void pyDefineICSpectrum(py::module&);

} // namespace rgnr

#endif // PHYSICS_IC_HPP
