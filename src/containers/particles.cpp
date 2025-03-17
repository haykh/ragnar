#include "containers/particles.hpp"

#include "utils/array.h"
#include "utils/snippets.h"
#include "utils/types.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <pybind11/pybind11.h>

#include <array>
#include <stdexcept>
#include <string>

namespace math = Kokkos;
namespace py   = pybind11;
using namespace pybind11::literals;

namespace rgnr {

  template <dim_t D>
  void Particles<D>::setNactive(std::size_t nactive) {
    if (!is_allocated()) {
      throw std::runtime_error("Particles not allocated");
    }
    m_nactive = nactive;
  }

  template <dim_t D>
  void Particles<D>::setIgnoreCoords(bool ignore_coords) {
    m_coords_ignored = ignore_coords;
  }

  template <dim_t D>
  void Particles<D>::allocate(std::size_t nalloc) {
    if (is_allocated()) {
      throw std::runtime_error("Particles already allocated");
    }
    if (not m_coords_ignored) {
      X = Kokkos::View<real_t* [D]> {
        "X", nalloc
      };
    }
    U              = Kokkos::View<real_t* [3]> { "U", nalloc };
    E              = Kokkos::View<real_t* [3]> { "E", nalloc };
    B              = Kokkos::View<real_t* [3]> { "B", nalloc };
    m_is_allocated = true;
  }

  template <dim_t D>
  auto Particles<D>::energyDistribution(const Array<real_t*>& energy_bins,
                                        bool fourvel) const -> Array<real_t*> {
    py::print("Computing energy distribution for",
              label(),
              "...",
              "end"_a   = "",
              "flush"_a = true);
    auto energy_distribution = Kokkos::View<real_t*> { "energy_distribution",
                                                       energy_bins.data.extent(0) };

    Kokkos::MinMaxScalar<real_t> eminmax;
    const auto&                  e_bins = energy_bins.data;
    Kokkos::parallel_reduce(
      "EnergyBinsMinMax",
      e_bins.extent(0),
      KOKKOS_LAMBDA(std::size_t p, Kokkos::MinMaxScalar<real_t>& leminmax) {
        if (e_bins(p) < leminmax.min_val) {
          leminmax.min_val = e_bins(p);
        }
        if (e_bins(p) > leminmax.max_val) {
          leminmax.max_val = e_bins(p);
        }
      },
      Kokkos::MinMax<real_t> { eminmax });

    const auto energy_min = eminmax.min_val, energy_max = eminmax.max_val;
    const auto n = e_bins.extent(0);

    auto energy_distribution_scat = Kokkos::Experimental::create_scatter_view(
      energy_distribution);
    const auto& Uarr = this->U;

    Kokkos::parallel_for(
      "ComputeEnergyDistribution",
      range(),
      KOKKOS_LAMBDA(std::size_t pidx) {
        const auto Usqr = Uarr(pidx, in::x) * Uarr(pidx, in::x) +
                          Uarr(pidx, in::y) * Uarr(pidx, in::y) +
                          Uarr(pidx, in::z) * Uarr(pidx, in::z);
        const auto energy = fourvel ? math::sqrt(Usqr) : math::sqrt(1.0 + Usqr);

        std::size_t idx;
        if (energy < energy_min) {
          idx = 0;
        } else if (energy >= energy_max) {
          idx = n - 1;
        } else {
          auto ei = static_cast<std::size_t>(
            static_cast<real_t>(n - 1) *
            math::abs(math::log10(energy / energy_min)) /
            math::log10(energy_max / energy_min));

          idx = ei > n - 1 ? n - 1 : ei;
        }

        auto energy_distribution_acc  = energy_distribution_scat.access();
        energy_distribution_acc(idx) += 1.0 / energy;
      });

    Kokkos::Experimental::contribute(energy_distribution, energy_distribution_scat);
    Kokkos::fence();

    py::print(": OK", "flush"_a = true);
    return energy_distribution;
  }

  template <dim_t D>
  void Particles<D>::printHead(std::size_t start, std::size_t number) const {
    py::print("Particles:", m_label);
    if (is_allocated()) {
      if (start + number > m_nactive) {
        throw std::runtime_error(
          "Number of particles to print exceeds allocated space");
      }
      py::print(" [", m_nactive, "/", U.extent(0), "]");
      py::print(" showing", start, "to", start + number);
      const auto nelems = std::min(m_nactive, number);
      const auto slice  = std::make_pair(static_cast<int>(start),
                                        static_cast<int>(nelems));

      if (not m_coords_ignored) { // print X
        auto X_h = Kokkos::create_mirror_view(
          Kokkos::subview(X, slice, Kokkos::ALL));
        Kokkos::deep_copy(X_h, Kokkos::subview(X, slice, Kokkos::ALL));

        for (auto d = 0u; d < D; ++d) {
          py::print(" X_" + std::to_string(d + 1), ":", "end"_a = "");
          if (start > 0) {
            py::print("...", "end"_a = "");
          }
          for (auto i = 0u; i < nelems; ++i) {
            py::print(X_h(i, d), "end"_a = "");
          }
          if (m_nactive > number) {
            py::print("...");
          } else {
            py::print();
          }
        }
      }
      { // print U, E, B
        auto quantities_str = std::array<std::string, 3> { "U", "E", "B" };
        auto U_h            = Kokkos::create_mirror_view(
          Kokkos::subview(U, slice, Kokkos::ALL));
        auto E_h = Kokkos::create_mirror_view(
          Kokkos::subview(E, slice, Kokkos::ALL));
        auto B_h = Kokkos::create_mirror_view(
          Kokkos::subview(B, slice, Kokkos::ALL));
        Kokkos::deep_copy(U_h, Kokkos::subview(U, slice, Kokkos::ALL));
        Kokkos::deep_copy(E_h, Kokkos::subview(E, slice, Kokkos::ALL));
        Kokkos::deep_copy(B_h, Kokkos::subview(B, slice, Kokkos::ALL));

        auto quantities_view = std::array<decltype(U_h), 3> { U_h, E_h, B_h };

        for (auto q = 0; q < 3; ++q) {
          for (auto d = 0u; d < 3; ++d) {
            py::print(" ",
                      quantities_str[q] + "_" + std::to_string(d + 1),
                      ":",
                      "end"_a = "");
            if (start > 0) {
              py::print("...", "end"_a = "");
            }
            for (auto i = 0u; i < nelems; ++i) {
              py::print(quantities_view[q](i, d), " ", "end"_a = "");
            }
            if (m_nactive > number) {
              py::print("...");
            } else {
              py::print();
            }
          }
        }
      }
    } else {
      py::print(" [ not allocated ]");
    }
  }

  template <dim_t D>
  auto Particles<D>::repr() const -> std::string {
    return "Particles<" + std::to_string(D) + "D> (" + m_label + ") : " +
           (is_allocated() ? ToHumanReadable(nactive(), USE_POW10)
                           : "not allocated");
  }

  template <dim_t D>
  auto Particles<D>::range() const -> Kokkos::RangePolicy<> {
    return Kokkos::RangePolicy<> { 0, m_nactive };
  }

  template <dim_t N>
  auto getSubview(const std::string&               name,
                  std::size_t                      comp,
                  std::size_t                      n,
                  const Kokkos::View<real_t* [N]>& view) -> Array<real_t*> {
    if (comp >= N) {
      throw std::out_of_range("Invalid component");
    }
    Array<real_t*> Arr {};
    Arr.data = Kokkos::View<real_t*> { name + "_" + std::to_string(comp + 1), n };
    Kokkos::deep_copy(Arr.data, Kokkos::subview(view, Kokkos::ALL, comp));
    return Arr;
  }

  template <dim_t D>
  auto Particles<D>::Xarr(std::size_t d) const -> Array<real_t*> {
    return getSubview<D>("X", d, m_nactive, X);
  }

  template <dim_t D>
  auto Particles<D>::Uarr(std::size_t d) const -> Array<real_t*> {
    return getSubview<3>("U", d, m_nactive, U);
  }

  template <dim_t D>
  auto Particles<D>::Earr(std::size_t d) const -> Array<real_t*> {
    return getSubview<3>("E", d, m_nactive, E);
  }

  template <dim_t D>
  auto Particles<D>::Barr(std::size_t d) const -> Array<real_t*> {
    return getSubview<3>("B", d, m_nactive, B);
  }

  template <dim_t D>
  void pyDefineParticles(py::module& m) {
    py::class_<Particles<D>>(m, ("Particles_" + std::to_string(D) + "D").c_str())
      .def(py::init<const std::string&>())
      .def("__repr__", &Particles<D>::repr)
      .def("__len__", &Particles<D>::nactive)
      .def("setNactive", &Particles<D>::setNactive)
      .def("setIgnoreCoords", &Particles<D>::setIgnoreCoords)
      .def("allocate", &Particles<D>::allocate)
      .def("printHead",
           &Particles<D>::printHead,
           "start"_a  = 0,
           "number"_a = 5,
           R"rgnrdoc(
              Print the first `number` particles starting from `start`

              Parameters
              ----------
              start : int
                Starting index [default: 0]

              number : int
                Number of particles to print [default: 5]
          )rgnrdoc")
      .def("is_allocated", &Particles<D>::is_allocated)
      .def("nactive", &Particles<D>::nactive)
      .def("label", &Particles<D>::label)
      .def("energyDistribution",
           &Particles<D>::energyDistribution,
           "energy_bins"_a,
           "fourvel"_a = true,
           R"rgnrdoc(
              Compute energy distribution for the particles

              Parameters
              ----------
              energy_bins : Array
                Energy bins

              fourvel : bool
                Whether to compute the energy distribution using four-velocity or Lorentz factor [default: True]

              Returns
              -------
              Array
                Energy distribution dN / dE, where E is either U or sqrt(1 + U^2)
          )rgnrdoc")
      .def("X", &Particles<D>::Xarr, "d"_a, R"rgnrdoc(
        Return the array of particle coordinates in dimension `d`

        Parameters
        ----------
        d : int
          Dimension

        Returns
        -------
        Array
          Array of particle coordinates in dimension `d`
      )rgnrdoc")
      .def("U", &Particles<D>::Uarr, "d"_a, R"rgnrdoc(
        Return the array of particle four-velocities in dimension `d`

        Parameters
        ----------
        d : int
          Dimension

        Returns
        -------
        Array
          Array of particle four-velocities in dimension `d`
      )rgnrdoc")
      .def("E", &Particles<D>::Earr, "d"_a, R"rgnrdoc(
        Return the array of electric field components 

        Parameters
        ----------
        d : int
          Dimension

        Returns
        -------
        Array
          Array of electric field components in dimension `d`
      )rgnrdoc")
      .def("B", &Particles<D>::Barr, "d"_a, R"rgnrdoc(
        Return the array of magnetic field components 

        Parameters
        ----------
        d : int
          Dimension

        Returns
        -------
        Array
          Array of magnetic field components in dimension `d`
      )rgnrdoc")
      .doc() = R"rgnrdoc(
        A class holding particle data

        Methods
        -------
        setNactive(nactive: int)
          Set the number of active particles

        setIgnoreCoords(ignore_coords: bool)
          Set whether to ignore particle coordinates

        allocate(nalloc: int)
          Allocate space for the particles

        printHead(start: int = 0, number: int = 5)
          Print the first `number` particles starting from `start`

        energyDistribution(energy_bins: Array, fourvel: bool = True) -> Array
          Compute energy distribution for the particles

        Accessors
        ---------
        is_allocated() -> bool
          Return whether the particles are allocated

        nactive() -> int
          Return the number of active particles

        label() -> str
          Return the label for the particles

        X(d: int) -> Array
          Return the array of particle coordinates in dimension `d`
      
        U(d: int) -> Array
          Return the array of particle four-velocities in dimension `d`

        E(d: int) -> Array
          Return the array of electric field components

        B(d: int) -> Array
          Return the array of magnetic field components
      )rgnrdoc";
  }

  template class Particles<1>;
  template class Particles<2>;
  template class Particles<3>;

  template void pyDefineParticles<1>(py::module&);
  template void pyDefineParticles<2>(py::module&);
  template void pyDefineParticles<3>(py::module&);

} // namespace rgnr
