#ifndef PHYSICS_SYNCHROTRON_HPP
#define PHYSICS_SYNCHROTRON_HPP

#include "utils/array.h"
#include "utils/tabulation.h"
#include "utils/types.h"

#include "containers/particles.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>

namespace math = Kokkos;
namespace py   = pybind11;

namespace rgnr {

  namespace sync {

    auto Ffunc_integrand(real_t x) -> real_t;

    auto TabulateFfunc(std::size_t npoints = 200,
                       real_t      xmin    = static_cast<real_t>(1e-6),
                       real_t      xmax    = static_cast<real_t>(100))
      -> TabulatedFunction<true>;

    /*
     * Computes `E dN / d(ln E)` or `E dN / dE`
     * ... for synchrotron radiation from a given distribution
     * - if esyn_bins is logarithmically spaced --> `E^2 dN / dE`
     * - if esyn_bins is linearly spaced --> `E dN / dE`
     */
    class KernelFromDist {
      const Kokkos::View<real_t*> m_gbeta_bins, m_dn_dgbeta;

      const Kokkos::View<real_t*> m_Ffunc_x, m_Ffunc_y;
      const real_t                m_Ffunc_xmin, m_Ffunc_xmax;
      const std::size_t           m_Ffunc_npoints;

      const Kokkos::View<real_t*> m_esyn_bins_mc2;

      Kokkos::Experimental::ScatterView<real_t*> m_esyn2_dn_desyn_scat;

      const real_t m_gamma_syn, m_esyn_at_gamma_syn_mc2;

    public:
      KernelFromDist(
        const Array<real_t*>                              gbeta_bins,
        const Array<real_t*>                              dn_dgbeta,
        const TabulatedFunction<true>&                    f_func,
        const Array<real_t*>&                             esyn_bins,
        const Kokkos::Experimental::ScatterView<real_t*>& esyn2_dn_desyn_scat,
        real_t                                            gamma_syn,
        real_t                                            esyn_at_gamma_syn)
        : m_gbeta_bins { gbeta_bins.data }
        , m_dn_dgbeta { dn_dgbeta.data }
        , m_Ffunc_x { f_func.xView() }
        , m_Ffunc_y { f_func.yView() }
        , m_Ffunc_xmin { f_func.xMin() }
        , m_Ffunc_xmax { f_func.xMax() }
        , m_Ffunc_npoints { f_func.nPoints() }
        , m_esyn_bins_mc2 { esyn_bins.data }
        , m_esyn2_dn_desyn_scat { esyn2_dn_desyn_scat }
        , m_gamma_syn { gamma_syn }
        , m_esyn_at_gamma_syn_mc2 { esyn_at_gamma_syn } {
        if (esyn_bins.unit != EnergyUnits::mec2 and
            esyn_bins.unit != EnergyUnits::mpc2) {
          throw std::runtime_error("esyn_bins must be in units of mc^2");
        }
        if (gbeta_bins.extent(0) != dn_dgbeta.extent(0)) {
          throw std::invalid_argument(
            "gbeta_bins and dn_dgbeta must have the same size");
        }
      }

      KOKKOS_INLINE_FUNCTION
      void operator()(std::size_t gbidx, std::size_t eidx) const {
        const auto photon_e_mc2 = m_esyn_bins_mc2(eidx);
        const auto gbeta        = m_gbeta_bins(gbidx);
        const auto dn           = m_dn_dgbeta(gbidx);

        const auto peak_energy_mc2 = m_esyn_at_gamma_syn_mc2 * gbeta * gbeta /
                                     (m_gamma_syn * m_gamma_syn);

        if (peak_energy_mc2 > 0.0) {
          const auto Fval = InterpolateTabulatedFunction<LOGGRID>(
            photon_e_mc2 / peak_energy_mc2,
            m_Ffunc_x,
            m_Ffunc_y,
            m_Ffunc_npoints,
            m_Ffunc_xmin,
            m_Ffunc_xmax);
          auto esyn2_dn_desyn_scat_acc   = m_esyn2_dn_desyn_scat.access();
          esyn2_dn_desyn_scat_acc(eidx) += dn * photon_e_mc2 * Fval;
        }
      }
    };

    /*
     * Computes `E dN / d(ln E)` or `E dN / dE` for synchrotron radiation
     * ... from a given population of particles
     * - if esyn_bins is logarithmically spaced --> `E^2 dN / dE`
     * - if esyn_bins is linearly spaced --> `E dN / dE`
     */
    template <dim_t D>
    class Kernel {
      const Kokkos::View<real_t* [3]> m_U, m_E, m_B;

      const Kokkos::View<real_t*> m_Ffunc_x, m_Ffunc_y;
      const real_t                m_Ffunc_xmin, m_Ffunc_xmax;
      const std::size_t           m_Ffunc_npoints;

      const Kokkos::View<real_t*> m_esyn_bins_mc2;

      Kokkos::Experimental::ScatterView<real_t*> m_esyn2_dn_desyn_scat;

      const real_t m_B0, m_gamma_syn, m_esyn_at_gamma_syn_mc2;

    public:
      Kernel(const Particles<D>&            prtls,
             const TabulatedFunction<true>& f_func,
             const Array<real_t*>&          esyn_bins,
             const Kokkos::Experimental::ScatterView<real_t*>& esyn2_dn_desyn_scat,
             real_t B0,
             real_t gamma_syn,
             real_t esyn_at_gamma_syn)
        : m_U { prtls.U }
        , m_E { prtls.E }
        , m_B { prtls.B }
        , m_Ffunc_x { f_func.xView() }
        , m_Ffunc_y { f_func.yView() }
        , m_Ffunc_xmin { f_func.xMin() }
        , m_Ffunc_xmax { f_func.xMax() }
        , m_Ffunc_npoints { f_func.nPoints() }
        , m_esyn_bins_mc2 { esyn_bins.data }
        , m_esyn2_dn_desyn_scat { esyn2_dn_desyn_scat }
        , m_B0 { B0 }
        , m_gamma_syn { gamma_syn }
        , m_esyn_at_gamma_syn_mc2 { esyn_at_gamma_syn } {
        if (esyn_bins.unit != EnergyUnits::mec2 and
            esyn_bins.unit != EnergyUnits::mpc2) {
          throw std::runtime_error("esyn_bins must be in units of mc^2");
        }
      }

      KOKKOS_INLINE_FUNCTION
      void operator()(std::size_t pidx, std::size_t eidx) const {
        const auto photon_e_mc2 = m_esyn_bins_mc2(eidx);

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
          auto esyn2_dn_desyn_scat_acc   = m_esyn2_dn_desyn_scat.access();
          esyn2_dn_desyn_scat_acc(eidx) += photon_e_mc2 * chiR * Fval;
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
        peak_energy = m_esyn_at_gamma_syn_mc2 * gamma * gamma * chiR /
                      (m_gamma_syn * m_gamma_syn);
      }
    };

  } // namespace sync

  auto SynchrotronSpectrumFromDist(const Array<real_t*>&,
                                   const Array<real_t*>&,
                                   const Array<real_t*>&,
                                   real_t,
                                   real_t) -> Array<real_t*>;

  template <dim_t D>
  auto SynchrotronSpectrum(const Particles<D>&,
                           const Array<real_t*>&,
                           real_t,
                           real_t,
                           real_t) -> Array<real_t*>;

  template <dim_t D>
  void pyDefineSynchrotronSpectrum(py::module&);

  void pyDefineSynchrotronSpectrumFromDist(py::module&);

} // namespace rgnr

#endif // PHYSICS_SYNCHROTRON_HPP
