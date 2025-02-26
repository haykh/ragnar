#include "simulation.h"

#include "utils/snippets.h"
#include "utils/types.h"

#include "containers/dimensionals.hpp"
#include "containers/particles.hpp"
#include "io/h5.hpp"
#include "io/toml.hpp"
#include "physics/ic.hpp"
#include "physics/synchrotron.hpp"
#include "plugins/tristan-v2.hpp"

#include <Kokkos_Core.hpp>
#include <highfive/highfive.hpp>
#include <toml.hpp>

#include <cmath>
#include <map>
#include <string>
#include <utility>

namespace rgnr {

  void computeSyncSpectrum(
    const Particles<3>&                     prtls,
    HighFive::File&                         file,
    const DimensionalArray<EnergyUnits>&    photon_energy_bins,
    const std::string&                      label,
    real_t                                  B0,
    real_t                                  gamma_syn,
    const DimensionalQuantity<EnergyUnits>& eps_at_gamma_syn) {

    auto sync_spectrum = SynchrotronSpectrum<3>(prtls,
                                                photon_energy_bins,
                                                B0,
                                                gamma_syn,
                                                eps_at_gamma_syn);

    io::h5::Write1DView<real_t>(file, "sync_intensity_" + label, sync_spectrum);
  }

  void computeICSpectrum(const Particles<3>&           prtls,
                         HighFive::File&               file,
                         const Kokkos::View<real_t*>&  prtl_energy_bins,
                         const SoftPhotonDistribution& soft_photon_distribution,
                         const DimensionalArray<EnergyUnits>& soft_photon_energy_bins,
                         const DimensionalArray<EnergyUnits>& photon_energy_bins,
                         const std::string&                      label,
                         real_t                                  gamma_scale,
                         const DimensionalQuantity<EnergyUnits>& mc2) {
    const auto prtl_energy_distribution = prtls.energyDistribution(prtl_energy_bins);
    auto ic_spectrum = ICSpectrum<SoftPhotonDistribution>(prtl_energy_bins,
                                                          prtl_energy_distribution,
                                                          soft_photon_energy_bins,
                                                          soft_photon_distribution,
                                                          photon_energy_bins,
                                                          gamma_scale,
                                                          mc2);

    io::h5::Write1DView<real_t>(file, "gammaM1_" + label, prtl_energy_bins);
    io::h5::Write1DView<real_t>(file,
                                "distribution_" + label,
                                prtl_energy_distribution);
    io::h5::Write1DView<real_t>(file, "ic_intensity_" + label, ic_spectrum);
  }

  void Simulation(int, char*[]) {
    const auto inputfile = io::toml::ReadFile("input.toml");

    const auto path = toml::find<std::string>(inputfile, "data", "path");
    const auto step = toml::find<std::size_t>(inputfile, "data", "step");

    auto outfile = toml::find<std::string>(inputfile, "output", "file");

    const auto gamma_syn_pairs = toml::find<real_t>(inputfile,
                                                    "parameters",
                                                    "gamma_syn_pairs");
    const auto gamma_syn_ions  = toml::find<real_t>(inputfile,
                                                   "parameters",
                                                   "gamma_syn_ions");

    const auto gamma_syn_pairs_real = toml::find<real_t>(
      inputfile,
      "parameters",
      "gamma_syn_pairs_real");

    const auto process_pairs = toml::find<bool>(inputfile, "process", "pairs");
    const auto process_ions  = toml::find<bool>(inputfile, "process", "ions");
    const auto process_synchrotron = toml::find<bool>(inputfile,
                                                      "process",
                                                      "synchrotron");
    const auto process_ic = toml::find<bool>(inputfile, "process", "ic");

    outfile += ".h5";

    outfile = TemplateReplace(
      outfile,
      toml::get<std::map<std::string, real_t>>(inputfile.at("parameters")));
    std::cout << "outfile: " << outfile << std::endl;

    TristanV2<3> plugin;
    plugin.setPath(path);
    plugin.setStep(step);

    HighFive::File file(outfile,
                        HighFive::File::ReadWrite | HighFive::File::Create);

    const std::size_t stride = 1;

    const auto photon_energy_at_gamma_syn = DimensionalQuantity<EnergyUnits>(
      Quantity::Energy,
      EnergyUnits::mc2,
      (27.0 / 8.0) * 0.1 * 137.0);

    const real_t sigma = 100.0;
    const real_t c_omp = 2.0;
    const real_t cc    = 0.45;
    const real_t b0    = cc * cc * std::sqrt(sigma) / c_omp;

    if (process_pairs) { // pairs
      const auto mc2_eV = DimensionalQuantity<EnergyUnits>(Quantity::Energy,
                                                           EnergyUnits::eV,
                                                           511000.0);

      const auto gamma_scale_for_ic = gamma_syn_pairs_real / gamma_syn_pairs;

      const auto pairs = std::vector<std::string> { "e-", "e+" };

      const auto pair_energy_bins = LogspaceView(1e-1, 200, 200);

      // define energy bins >
      const std::size_t sync_photon_nbins       = 200;
      const auto        sync_photon_energy_bins = DimensionalArray<EnergyUnits>(
        Quantity::Energy,
        EnergyUnits::mc2,
        LogspaceView(1e-3, 1e3, sync_photon_nbins));

      const std::size_t ic_photon_nbins       = 200;
      const auto        ic_photon_energy_bins = DimensionalArray<EnergyUnits>(
        Quantity::Energy,
        EnergyUnits::mc2,
        LogspaceView(10, 1e7, ic_photon_nbins));

      const auto soft_photon_energy_bins = DimensionalArray<EnergyUnits>(
        Quantity::Energy,
        EnergyUnits::eV,
        LogspaceView(1e-5, 1e2, 200));
      const auto soft_photon_distribution = SoftPhotonDistribution {};
      // <

      // write energy bins to file
      if (process_synchrotron) {
        io::h5::Write1DView<real_t>(file,
                                    "sync_photon_energy_mec2",
                                    sync_photon_energy_bins.data);
      }

      if (process_ic) {
        io::h5::Write1DView<real_t>(file,
                                    "ic_photon_energy_mec2",
                                    ic_photon_energy_bins.data);
      }
      // <

      for (auto i = 0u; i < pairs.size(); ++i) {
        const auto label = pairs[i];
        const auto sp    = i + 4;

        Particles<3> prtls { label };
        plugin.readParticles(sp, &prtls, stride, IGNORE_COORDS);

        if (process_synchrotron) {
          computeSyncSpectrum(prtls,
                              file,
                              sync_photon_energy_bins,
                              label,
                              b0,
                              gamma_syn_pairs,
                              photon_energy_at_gamma_syn);
        }
        if (process_ic) {
          computeICSpectrum(prtls,
                            file,
                            pair_energy_bins,
                            soft_photon_distribution,
                            soft_photon_energy_bins,
                            ic_photon_energy_bins,
                            label,
                            gamma_scale_for_ic,
                            mc2_eV);
        }
      }
    }

    if (process_ions) { // ions
      const std::size_t sync_photon_nbins       = 200;
      const auto        sync_photon_energy_bins = DimensionalArray<EnergyUnits>(
        Quantity::Energy,
        EnergyUnits::mc2,
        LogspaceView(1e-1, 1e5, sync_photon_nbins));

      Particles<3> prtls { "i" };
      plugin.readParticles(6, &prtls, stride, IGNORE_COORDS);

      if (process_synchrotron) {
        computeSyncSpectrum(prtls,
                            file,
                            sync_photon_energy_bins,
                            "i",
                            b0,
                            gamma_syn_ions,
                            photon_energy_at_gamma_syn);

        io::h5::Write1DView<real_t>(file,
                                    "sync_photon_energy_mic2",
                                    sync_photon_energy_bins.data);
      }
    }
  }

} // namespace rgnr
