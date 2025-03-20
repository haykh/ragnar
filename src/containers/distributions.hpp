#ifndef CONTAINERS_DISTRIBUTIONS_HPP
#define CONTAINERS_DISTRIBUTIONS_HPP

#include "utils/global.h"

#include "containers/array.hpp"
#include "containers/bins.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace rgnr {

  struct DistGenerator {
    static constexpr const char* name { "DistGenerator" };
    static constexpr const char* shortname { "Dist" };

    auto compute(const Bins& energy_bins) const -> Array1D<real_t>;

    virtual auto f(real_t energy) const -> real_t = 0;

  protected:
    real_t m_norm;
  };

  struct PlawGenerator : public DistGenerator {
    static constexpr const char* name { "PlawGenerator" };
    static constexpr const char* shortname { "Plaw" };

    const real_t p;
    const real_t emin, emax;

    PlawGenerator(real_t, real_t = 0.0, real_t = 0.0);

    auto f(real_t) const -> real_t override;
  };

  struct BrokenPlawGenerator : public DistGenerator {
    static constexpr const char* name { "BrokenPlawGenerator" };
    static constexpr const char* shortname { "BrokenPlaw" };

    const real_t e_break, emin, emax;
    const real_t p1, p2;

    BrokenPlawGenerator(real_t, real_t, real_t, real_t = 0.0, real_t = 0.0);

    auto f(real_t) const -> real_t override;
  };

  struct DeltaGenerator : public DistGenerator {
    static constexpr const char* name { "DeltaGenerator" };
    static constexpr const char* shortname { "Delta" };

    const real_t energy0;
    const real_t denergy;

    DeltaGenerator(real_t, real_t);

    auto f(real_t) const -> real_t override;
  };

  class TabulatedDistribution {
    Bins            m_e_bins;
    Array1D<real_t> m_f;

  public:
    TabulatedDistribution(const Bins&, const Array1D<real_t>&);

    TabulatedDistribution(const Bins&, const DistGenerator&);

    auto EnergyBins() const -> const Bins&;

    auto F() const -> const Array1D<real_t>&;

    auto extent() const -> std::size_t;
    auto log_spaced() const -> bool;
  };

  void pyDefineGenerators(py::module&);

} // namespace rgnr

#endif // CONTAINERS_DISTRIBUTIONS_HPP
