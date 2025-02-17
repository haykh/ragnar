#include "simulation.h"

#include "utils/snippets.h"

#include "containers/particles.hpp"
#include "io/h5.hpp"
#include "physics/synchrotron.hpp"
#include "plugins/tristan-v2.hpp"

#include <Kokkos_Core.hpp>
#include <highfive/highfive.hpp>

#include <cmath>
#include <map>
#include <string>
#include <utility>

namespace rgnr {

  void computeSyncSpectrum(const TristanV2<3>&          plugin,
                           HighFive::File&              file,
                           unsigned short               sp,
                           const std::string&           label,
                           std::size_t                  stride,
                           const Kokkos::View<real_t*>& photon_energy_bins) {
    const real_t sigma = 100.0;
    const real_t c_omp = 2.0;
    const real_t cc    = 0.45;
    const real_t b0    = cc * cc * std::sqrt(sigma) / c_omp;

    const real_t gamma_syn                  = 50.0;
    const real_t photon_energy_at_gamma_syn = 16.0;

    Particles<3> prtls { label };
    plugin.readParticles(sp, &prtls, stride, IGNORE_COORDS);

    auto sync_spectrum = SynchrotronSpectrum<3>(
      prtls,
      photon_energy_bins,
      {
        {                        "B0",                         b0},
        {                 "gamma_syn",                  gamma_syn},
        {"photon_energy_at_gamma_syn", photon_energy_at_gamma_syn}
    });

    auto sync_spectrum_h = Kokkos::create_mirror_view(sync_spectrum);
    Kokkos::deep_copy(sync_spectrum_h, sync_spectrum);
    io::h5::Write1DArrays<real_t, decltype(sync_spectrum_h)>(
      file,
      "sync_intensity_" + label,
      sync_spectrum_h,
      sync_spectrum.extent(0));
  }

  void Simulation(int, char*[]) {
    {
      TristanV2<3> plugin;
      plugin.setPath("/mnt/e/Downloads");
      plugin.setStep(18);

      HighFive::File file("sync_spectrum.h5",
                          HighFive::File::ReadWrite | HighFive::File::Create);

      {
        const std::size_t nphoton_bins = 200;
        const auto photon_energy_bins  = LogspaceView(1e-3, 1e3, nphoton_bins);

        computeSyncSpectrum(plugin, file, 4, "e-", 1, photon_energy_bins);
        computeSyncSpectrum(plugin, file, 5, "e+", 1, photon_energy_bins);

        auto photon_energy_bins_h = Kokkos::create_mirror_view(photon_energy_bins);
        Kokkos::deep_copy(photon_energy_bins_h, photon_energy_bins);
        io::h5::Write1DArrays<real_t, decltype(photon_energy_bins_h)>(
          file,
          "photon_energy_e",
          photon_energy_bins_h,
          photon_energy_bins.extent(0));
      }

      {
        const std::size_t nphoton_bins = 200;
        const auto photon_energy_bins  = LogspaceView(1e-1, 1e5, nphoton_bins);
        computeSyncSpectrum(plugin, file, 6, "i", 1, photon_energy_bins);

        auto photon_energy_bins_h = Kokkos::create_mirror_view(photon_energy_bins);
        Kokkos::deep_copy(photon_energy_bins_h, photon_energy_bins);
        io::h5::Write1DArrays<real_t, decltype(photon_energy_bins_h)>(
          file,
          "photon_energy_i",
          photon_energy_bins_h,
          photon_energy_bins.extent(0));
      }
    }
  }

} // namespace rgnr
