#include "containers/distributions.hpp"

#include "utils/global.h"

#include "containers/array.hpp"
#include "containers/bins.hpp"

#include <Kokkos_Core.hpp>
#include <pybind11/pybind11.h>

#include <cmath>
#include <stdexcept>

namespace py = pybind11;
using namespace pybind11::literals;

namespace rgnr {

  auto computeDist(const Bins&          energy_bins,
                   const DistGenerator& generator) -> Array1D<real_t> {
    Array1D<real_t> f {};
    f.data        = Kokkos::View<real_t*> { "f", energy_bins.extent() };
    auto f_h      = Kokkos::create_mirror_view(f.data);
    auto e_bins_h = Kokkos::create_mirror_view(energy_bins.data);
    Kokkos::deep_copy(e_bins_h, energy_bins.data);
    for (auto i = 0u; i < energy_bins.extent(); ++i) {
      f_h(i) = generator.f(e_bins_h(i));
    }
    Kokkos::deep_copy(f.data, f_h);
    return f;
  }

  auto DistGenerator::compute(const Bins& energy_bins) const -> Array1D<real_t> {
    return computeDist(energy_bins, *this);
  }

  PlawGenerator::PlawGenerator(real_t p, real_t emin, real_t emax)
    : DistGenerator {}
    , p { p }
    , emin { emin }
    , emax { emax } {
    if (emin < 0.0) {
      throw std::runtime_error("emin < 0.0");
    } else if (emin > emax and emax > 0.0) {
      throw std::runtime_error("emin > emax");
    } else if (emin == 0.0 and p <= -1.0) {
      throw std::runtime_error(
        "p <= -1 and emin = 0.0 : normalization diverges");
    } else if (emax == 0.0 and p >= -1.0) {
      throw std::runtime_error(
        "p >= -1 and emax = 0.0 (infinity) : normalization diverges");
    } else if (emax == 0.0) {
      m_norm = -std::pow(emin, p + 1) / (p + 1);
    } else if (p != -1.0) {
      m_norm = (std::pow(emax, p + 1) - std::pow(emin, p + 1)) / (p + 1);
    } else {
      m_norm = std::log(emax / emin);
    }
  }

  auto PlawGenerator::f(real_t energy) const -> real_t {
    if (energy < emin or (emax > 0.0 and energy >= emax)) {
      return 0.0;
    }
    return std::pow(energy, p) / m_norm;
  }

  BrokenPlawGenerator::BrokenPlawGenerator(real_t e_break,
                                           real_t p1,
                                           real_t p2,
                                           real_t emin,
                                           real_t emax)
    : DistGenerator {}
    , e_break { e_break }
    , p1 { p1 }
    , p2 { p2 }
    , emin { emin }
    , emax { emax } {
    if (e_break <= 0.0) {
      throw std::runtime_error("e_break <= 0.0");
    }
    if (emin < 0.0) {
      throw std::runtime_error("emin < 0.0");
    }
    if (emin > emax and emax > 0.0) {
      throw std::runtime_error("emin > emax");
    }
    real_t pre_break, post_break;
    if (emin == 0.0 and p1 <= -1.0) {
      throw std::runtime_error(
        "p1 <= -1 and emin = 0.0 : normalization diverges");
    } else if (e_break <= emin) {
      pre_break = 0.0;
    } else {
      pre_break = (1 - std::pow(emin / e_break, p1 + 1)) / (p1 + 1);
    }

    if (emax == 0.0 and p2 >= -1.0) {
      throw std::runtime_error(
        "p2 >= -1 and emax = 0.0 (infinity) : normalization diverges");
    } else if (emax == 0.0) {
      post_break = 1 / (-p2 - 1);
    } else if (p2 == -1.0) {
      post_break = std::log(emax / e_break);
    } else {
      post_break = (std::pow(emax / e_break, p2 + 1) - 1) / (p2 + 1);
    }

    m_norm = (pre_break + post_break) * e_break;
  }

  auto BrokenPlawGenerator::f(real_t energy) const -> real_t {
    if (energy < emin or (emax > 0.0 and energy >= emax)) {
      return 0.0;
    }
    if (energy < e_break) {
      return std::pow(energy / e_break, p1) / m_norm;
    } else {
      return std::pow(energy / e_break, p2) / m_norm;
    }
  }

  DeltaGenerator::DeltaGenerator(real_t energy0, real_t denergy)
    : DistGenerator {}
    , energy0 { energy0 }
    , denergy { denergy } {
    m_norm = 1.0 / denergy;
  }

  auto DeltaGenerator::f(real_t energy) const -> real_t {
    if (std::abs(energy - energy0) < denergy * 0.5) {
      return 1.0 * m_norm;
    } else {
      return 0.0;
    }
  }

  TabulatedDistribution::TabulatedDistribution(const Bins&            e_bins,
                                               const Array1D<real_t>& f)
    : m_e_bins { e_bins }
    , m_f { f } {
    if (m_e_bins.extent() != m_f.extent()) {
      throw std::runtime_error("e_bins.extent() != f.extent()");
    }
  }

  TabulatedDistribution::TabulatedDistribution(const Bins&          e_bins,
                                               const DistGenerator& generator)
    : m_e_bins { e_bins }
    , m_f { computeDist(e_bins, generator) } {}

  auto TabulatedDistribution::EnergyBins() const -> const Bins& {
    return m_e_bins;
  }

  auto TabulatedDistribution::F() const -> const Array1D<real_t>& {
    return m_f;
  }

  auto TabulatedDistribution::extent() const -> std::size_t {
    return m_f.extent(0);
  }

  auto TabulatedDistribution::log_spaced() const -> bool {
    return m_e_bins.log_spaced;
  }

  void pyDefineGenerators(py::module& m) {
    py::class_<PlawGenerator>(m, PlawGenerator::name)
      .def(py::init<real_t, real_t, real_t>(), "p"_a, "emin"_a = 0.0, "emax"_a = 0.0)
      .def_readonly("p", &PlawGenerator::p)
      .def_readonly("emin", &PlawGenerator::emin)
      .def_readonly("emax", &PlawGenerator::emax)
      .def("compute", &PlawGenerator::compute, "energy_bins"_a, R"rgnrdoc(
        Compute the distribution for given energy bins
        Normalized to 1.0

        Parameters
        ----------
        energy_bins : Bins
          The energy bins for the distribution

        Returns
        -------
        Array1D
          The distribution for the given energy bins
        )rgnrdoc")
      .doc() = R"rgnrdoc(
        A power-law distribution from emin to emax (or infinity).
        Normalized to 1.0

        Attributes
        ----------
        p : float
          The power-law index

        emin : float, optional
          The minimum energy [default: 0.0]

        emax : float, optional
          The maximum energy, 0.0 = no maximum [default: 0.0]

        Methods
        -------
        compute(energy_bins)
          Compute the distribution for the given energy bins
        )rgnrdoc";

    py::class_<BrokenPlawGenerator>(m, BrokenPlawGenerator::name)
      .def(py::init<real_t, real_t, real_t, real_t, real_t>(),
           "e_break"_a,
           "p1"_a,
           "p2"_a,
           "emin"_a = 0.0,
           "emax"_a = 0.0)
      .def_readonly("e_break", &BrokenPlawGenerator::e_break)
      .def_readonly("p1", &BrokenPlawGenerator::p1)
      .def_readonly("p2", &BrokenPlawGenerator::p2)
      .def_readonly("emin", &BrokenPlawGenerator::emin)
      .def_readonly("emax", &BrokenPlawGenerator::emax)
      .def("compute", &BrokenPlawGenerator::compute, "energy_bins"_a, R"rgnrdoc(
        Compute the distribution for given energy bins

        Parameters
        ----------
        energy_bins : Bins
          The energy bins for the distribution

        Returns
        -------
        Array1D
          The distribution for the given energy bins
        )rgnrdoc")
      .doc() = R"rgnrdoc(
        A broken power-law distribution from emin to emax (or infinity).
        Normalized to 1.0

        Attributes
        ----------
        e_break : float
          The break energy

        p1 : float
          The power-law index below the break energy

        p2 : float
          The power-law index above the break energy

        emin : float, optional
          The minimum energy [default: 0.0]

        emax : float, optional
          The maximum energy, 0.0 = no maximum [default: 0.0]

        Methods
        -------
        compute(energy_bins)
          Compute the distribution for the given energy bins
        )rgnrdoc";

    py::class_<DeltaGenerator>(m, DeltaGenerator::name)
      .def(py::init<real_t, real_t>(), "energy0"_a, "denergy"_a)
      .def_readonly("energy0", &DeltaGenerator::energy0)
      .def_readonly("denergy", &DeltaGenerator::denergy)
      .def("compute", &DeltaGenerator::compute, "energy_bins"_a, R"rgnrdoc(
        Compute the distribution for given energy bins

        Parameters
        ----------
        energy_bins : Bins
          The energy bins for the distribution

        Returns
        -------
        Array1D
          The distribution for the given energy bins
        )rgnrdoc")
      .doc() = R"rgnrdoc(
        A Delta distribution
        Normalized to 1.0

        Attributes
        ----------
        energy0 : float
          The energy of the delta function

        denergy : float
          The width of the delta function

        Methods
        -------
        compute(energy_bins)
          Compute the distribution for the given energy bins
      )rgnrdoc";

    py::class_<TabulatedDistribution>(m, "TabulatedDistribution")
      .def(py::init<const Bins&, const Array1D<real_t>&>(), "bins_energy"_a, "f"_a)
      .def(py::init<const Bins&, const PlawGenerator&>(), "bins_energy"_a, "generator"_a)
      .def(py::init<const Bins&, const BrokenPlawGenerator&>(),
           "bins_energy"_a,
           "generator"_a)
      .def(py::init<const Bins&, const DeltaGenerator&>(), "bins_energy"_a, "generator"_a)
      .def("extent", &TabulatedDistribution::extent, R"rgnrdoc(
        Get the extent of the distribution

        Returns
        -------
        int
          Number of energy bins
      )rgnrdoc")
      .def("log_spaced", &TabulatedDistribution::log_spaced, R"rgnrdoc(
        Check if the energy bins are log spaced

        Returns
        -------
        bool
          True if the energy bins are log spaced
      )rgnrdoc")
      .def("EnergyBins", &TabulatedDistribution::EnergyBins, R"rgnrdoc(
        Get the energy bins

        Returns
        -------
        Bins
          The energy bins
      )rgnrdoc")
      .def("F", &TabulatedDistribution::F, R"rgnrdoc(
        Get the distribution

        Returns
        -------
        Array1D
          The distribution
      )rgnrdoc")
      .doc() = R"rgnrdoc(
        A tabulated distribution

        Attributes
        ----------
        energy_bins : Bins
          The energy bins

        f : Array1D
          The distribution
      )rgnrdoc";
  }

} // namespace rgnr
