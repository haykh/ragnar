#ifndef PHYSICS_SYNCHROTRON_HPP
#define PHYSICS_SYNCHROTRON_HPP

#include "utils/tabulation.h"
#include "utils/types.h"

#include "containers/dimensionals.hpp"
#include "containers/particles.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <stdexcept>
#include <string>

namespace math = Kokkos;

namespace rgnr {
  namespace sync {

    auto Ffunc_integrand(real_t x) -> real_t;

    auto TabulateFfunc(std::size_t npoints = 200,
                       real_t      xmin    = static_cast<real_t>(1e-6),
                       real_t      xmax    = static_cast<real_t>(100))
      -> TabulatedFunction<true>;

    template <dim_t D>
    class Kernel {
      const Kokkos::View<real_t* [3]> m_U, m_E, m_B;
      const std::size_t               m_nprtls;

      const Kokkos::View<real_t*> m_Ffunc_x, m_Ffunc_y;
      const real_t                m_Ffunc_xmin, m_Ffunc_xmax;
      const std::size_t           m_Ffunc_npoints;

      const Kokkos::View<real_t*> m_photon_energy_bins_mc2;

      Kokkos::Experimental::ScatterView<real_t*> m_photon_spectrum;

      const real_t m_B0, m_gamma_syn, m_photon_energy_at_gamma_syn_mc2;

    public:
      Kernel(const Particles<D>&                  prtls,
             const TabulatedFunction<true>&       f_func,
             const DimensionalArray<EnergyUnits>& photon_energy_bins,
             const Kokkos::Experimental::ScatterView<real_t*>& photon_spectrum,
             real_t                                            B0,
             real_t                                            gamma_syn,
             const DimensionalQuantity<EnergyUnits>&           eps_at_gamma_syn)
        : m_U { prtls.U }
        , m_E { prtls.E }
        , m_B { prtls.B }
        , m_nprtls { prtls.nactive() }
        , m_Ffunc_x { f_func.xArr() }
        , m_Ffunc_y { f_func.yArr() }
        , m_Ffunc_xmin { f_func.xMin() }
        , m_Ffunc_xmax { f_func.xMax() }
        , m_Ffunc_npoints { f_func.nPoints() }
        , m_photon_energy_bins_mc2 { photon_energy_bins.data }
        , m_photon_spectrum { photon_spectrum }
        , m_B0 { B0 }
        , m_gamma_syn { gamma_syn }
        , m_photon_energy_at_gamma_syn_mc2 { eps_at_gamma_syn.value } {
        if (photon_energy_bins.unit() != EnergyUnits::mc2) {
          throw std::runtime_error(
            "photon_energy_bins must be in units of mc^2");
        }
        if (eps_at_gamma_syn.unit() != EnergyUnits::mc2) {
          throw std::runtime_error("eps_at_gamma_syn must be in units of mc^2");
        }
      }

      KOKKOS_INLINE_FUNCTION
      void operator()(std::size_t pidx, std::size_t eidx) const {
        const auto photon_e_mc2 = m_photon_energy_bins_mc2(eidx);

        real_t peak_energy_mc2, chiR;
        OmegaSync_ChiR(peak_energy_mc2,
                       chiR,
                       m_U(pidx, in::x),
                       m_U(pidx, in::y),
                       m_U(pidx, in::z),
                       m_E(pidx, in::x),
                       m_E(pidx, in::y),
                       m_E(pidx, in::z),
                       m_B(pidx, in::x),
                       m_B(pidx, in::y),
                       m_B(pidx, in::z));

        if (peak_energy_mc2 > 0.0) {
          const auto Fval = InterpolateTabulatedFunction<LOGGRID>(
            photon_e_mc2 / peak_energy_mc2,
            m_Ffunc_x,
            m_Ffunc_y,
            m_Ffunc_npoints,
            m_Ffunc_xmin,
            m_Ffunc_xmax);
          auto photon_spectrum_acc   = m_photon_spectrum.access();
          photon_spectrum_acc(eidx) += photon_e_mc2 * chiR * Fval;
        }
      }

      /*
       * @in: ux, uy, uz, ex, ey, ez, bx, by, bz
       * @out: peak_energy, chiR
       *
       * peak_energy = gamma^2 * chiR
       * chiR = sqrt(e_perp^2 - (beta . e)^2)
       */
      KOKKOS_INLINE_FUNCTION
      void OmegaSync_ChiR(real_t& peak_energy,
                          real_t& chiR,
                          real_t  ux,
                          real_t  uy,
                          real_t  uz,
                          real_t  ex,
                          real_t  ey,
                          real_t  ez,
                          real_t  bx,
                          real_t  by,
                          real_t  bz) const {
        const auto gamma  = math::sqrt(1.0 + ux * ux + uy * uy + uz * uz);
        const auto beta_x = ux / gamma;
        const auto beta_y = uy / gamma;
        const auto beta_z = uz / gamma;

        const auto beta_dot_e = beta_x * ex + beta_y * ey + beta_z * ez;
        const auto beta_Sqr = beta_x * beta_x + beta_y * beta_y + beta_z * beta_z;

        const auto eperp_x = ex - beta_dot_e * beta_x / beta_Sqr;
        const auto eperp_y = ey - beta_dot_e * beta_y / beta_Sqr;
        const auto eperp_z = ez - beta_dot_e * beta_z / beta_Sqr;

        const auto beta_cross_b_x = beta_y * bz - beta_z * by;
        const auto beta_cross_b_y = beta_z * bx - beta_x * bz;
        const auto beta_cross_b_z = beta_x * by - beta_y * bx;

        auto eprime_x = eperp_x + beta_cross_b_x;
        auto eprime_y = eperp_y + beta_cross_b_y;
        auto eprime_z = eperp_z + beta_cross_b_z;

        const auto eprime = math::sqrt(
          eprime_x * eprime_x + eprime_y * eprime_y + eprime_z * eprime_z);
        eprime_x /= eprime;
        eprime_y /= eprime;
        eprime_z /= eprime;

        const auto e_plus_beta_cross_b_x = ex + beta_cross_b_x;
        const auto e_plus_beta_cross_b_y = ey + beta_cross_b_y;
        const auto e_plus_beta_cross_b_z = ez + beta_cross_b_z;

        const auto e_plus_beta_cross_b_Sqr = e_plus_beta_cross_b_x *
                                               e_plus_beta_cross_b_x +
                                             e_plus_beta_cross_b_y *
                                               e_plus_beta_cross_b_y +
                                             e_plus_beta_cross_b_z *
                                               e_plus_beta_cross_b_z;

        chiR = math::sqrt(e_plus_beta_cross_b_Sqr - beta_dot_e * beta_dot_e) / m_B0;
        peak_energy = m_photon_energy_at_gamma_syn_mc2 * gamma * gamma * chiR /
                      (m_gamma_syn * m_gamma_syn);
      }
    };

  } // namespace sync

  template <dim_t D>
  auto SynchrotronSpectrum(const Particles<D>&,
                           const DimensionalArray<EnergyUnits>&,
                           real_t,
                           real_t,
                           const DimensionalQuantity<EnergyUnits>&)
    -> Kokkos::View<real_t*>;

} // namespace rgnr

#endif // PHYSICS_SYNCHROTRON_HPP
