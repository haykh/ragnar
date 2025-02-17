#ifndef PHYSICS_SYNCHROTRON_HPP
#define PHYSICS_SYNCHROTRON_HPP

#include "utils/tabulation.h"
#include "utils/types.h"

#include "containers/particles.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <map>
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
      const Kokkos::View<real_t* [3]> U, E, B;
      const std::size_t               nprtls;

      const Kokkos::View<real_t*> Ffunc_x, Ffunc_y;
      const real_t                Ffunc_xmin, Ffunc_xmax;
      const std::size_t           Ffunc_npoints;

      const Kokkos::View<real_t*> photon_energy_bins;

      Kokkos::Experimental::ScatterView<real_t*> photon_spectrum;

      const real_t B0, gamma_syn, photon_energy_at_gamma_syn;

    public:
      Kernel(const Particles<D>&            prtls,
             const TabulatedFunction<true>& f_func,
             const Kokkos::View<real_t*>&   photon_energy_bins,
             const Kokkos::Experimental::ScatterView<real_t*>& photon_spectrum,
             const std::map<std::string, real_t>&              params)
        : U { prtls.U }
        , E { prtls.E }
        , B { prtls.B }
        , nprtls { prtls.nactive() }
        , Ffunc_x { f_func.xArr() }
        , Ffunc_y { f_func.yArr() }
        , Ffunc_xmin { f_func.xMin() }
        , Ffunc_xmax { f_func.xMax() }
        , Ffunc_npoints { f_func.nPoints() }
        , photon_energy_bins { photon_energy_bins }
        , photon_spectrum { photon_spectrum }
        , B0 { params.at("B0") }
        , gamma_syn { params.at("gamma_syn") }
        , photon_energy_at_gamma_syn { params.at(
            "photon_energy_at_gamma_syn") } {}

      KOKKOS_INLINE_FUNCTION
      void operator()(std::size_t pidx, std::size_t eidx) const {
        const auto photon_e = photon_energy_bins(eidx);

        real_t peak_energy, chiR;
        OmegaSync_ChiR(peak_energy,
                       chiR,
                       U(pidx, in::x),
                       U(pidx, in::y),
                       U(pidx, in::z),
                       E(pidx, in::x),
                       E(pidx, in::y),
                       E(pidx, in::z),
                       B(pidx, in::x),
                       B(pidx, in::y),
                       B(pidx, in::z));

        if (peak_energy > 0.0) {
          const auto Fval = InterpolateTabulatedFunction<true>(photon_e / peak_energy,
                                                               Ffunc_x,
                                                               Ffunc_y,
                                                               Ffunc_npoints,
                                                               Ffunc_xmin,
                                                               Ffunc_xmax);
          auto photon_spectrum_acc   = photon_spectrum.access();
          photon_spectrum_acc(eidx) += photon_e * chiR * Fval;
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

        chiR = math::sqrt(e_plus_beta_cross_b_Sqr - beta_dot_e * beta_dot_e) / B0;
        peak_energy = photon_energy_at_gamma_syn * gamma * gamma * chiR /
                      (gamma_syn * gamma_syn);
      }
    };

  } // namespace sync

  template <dim_t D>
  auto SynchrotronSpectrum(const Particles<D>&,
                           const Kokkos::View<real_t*>&,
                           const std::map<std::string, real_t>&)
    -> Kokkos::View<real_t*>;

} // namespace rgnr

#endif // PHYSICS_SYNCHROTRON_HPP
