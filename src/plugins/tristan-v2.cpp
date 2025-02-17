#include "plugins/tristan-v2.hpp"

#include "utils/snippets.h"

#include "io/h5.hpp"

#include <Kokkos_Core.hpp>
#include <highfive/highfive.hpp>

#include <array>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace h5 = HighFive;

namespace rgnr {

  template <dim_t D>
  auto TristanV2<D>::label() const -> std::string {
    return "Tristan V2";
  }

  template <dim_t D>
  void TristanV2<D>::setPath(const std::string& path) {
    m_path      = path;
    is_path_set = true;
  }

  template <dim_t D>
  void TristanV2<D>::setStep(std::size_t st) {
    m_step      = st;
    is_step_set = true;
  }

  template <dim_t D>
  auto TristanV2<D>::getPath() const -> std::string {
    if (!is_path_set) {
      throw std::runtime_error("Path not set");
    }
    return m_path;
  }

  template <dim_t D>
  auto TristanV2<D>::getStep() const -> std::size_t {
    if (!is_step_set) {
      throw std::runtime_error("Step not set");
    }
    return m_step;
  }

  template <unsigned short N>
  auto readPrtlQuantity(h5::File                  file,
                        const std::string&        quantity,
                        std::size_t               stride,
                        std::size_t               idx,
                        Kokkos::View<real_t* [N]> arr) -> std::size_t {
    auto arr_h = Kokkos::create_mirror_view(Kokkos::subview(arr, Kokkos::ALL, idx));
    const auto nread = io::h5::Read1DArray<real_t, decltype(arr_h)>(file,
                                                                    quantity,
                                                                    arr_h,
                                                                    arr.extent(0),
                                                                    stride);
    Kokkos::deep_copy(Kokkos::subview(arr, Kokkos::ALL, idx), arr_h);
    return nread;
  }

  template <dim_t D>
  void TristanV2<D>::readParticles(unsigned short sp,
                                   Particles<D>*  prtls,
                                   std::size_t    read_every,
                                   bool           ignore_coordinates) const {
    const auto        step   = std::to_string(getStep());
    const auto        sp_str = std::to_string(sp);
    const std::string fname  = getPath() + "/output/prtl/prtl.tot." +
                              std::string(5 - step.length(), '0') + step;

    std::cout << "Reading particles #" << sp << " from " << fname << " ..."
              << std::endl;

    h5::File file { fname, h5::File::ReadOnly };

    const auto comps_coord = std::array<std::string, 3> { "x", "y", "z" };
    const auto comps_vel   = std::array<std::string, 3> { "u", "v", "w" };
    const auto comps_idx   = std::array<in, 3> { in::x, in::y, in::z };

    prtls->setIgnoreCoords(ignore_coordinates);

    const std::size_t nparticles = file.getDataSet("x_" + sp_str).getDimensions()[0] /
                                   read_every;
    prtls->allocate(nparticles);

    std::cout << "  found " << ToHumanReadable(nparticles, USE_POW10)
              << " particles" << std::endl;

    const auto report_ok =
      [](std::size_t np, std::size_t nread, const std::string& quantity) {
        if (np != nread) {
          throw std::runtime_error("Number of particles mismatch");
        }
        std::cout << "  " << std::left << std::setw(6) << quantity << ": OK "
                  << std::endl;
      };

    if (not ignore_coordinates) {
      for (auto d = 0u; d < D; ++d) {
        auto nread_coord = readPrtlQuantity<D>(file,
                                               comps_coord[d] + "_" + sp_str,
                                               read_every,
                                               comps_idx[d],
                                               prtls->X);
        report_ok(nparticles, nread_coord, comps_coord[d]);
      }
    }
    for (auto d = 0u; d < 3u; ++d) {
      auto nread_vel = readPrtlQuantity<3>(file,
                                           comps_vel[d] + "_" + sp_str,
                                           read_every,
                                           comps_idx[d],
                                           prtls->U);
      report_ok(nparticles, nread_vel, comps_vel[d]);
      auto nread_e = readPrtlQuantity<3>(file,
                                         "e" + comps_coord[d] + "_" + sp_str,
                                         read_every,
                                         comps_idx[d],
                                         prtls->E);
      report_ok(nparticles, nread_e, "e" + comps_coord[d]);
      auto nread_b = readPrtlQuantity<3>(file,
                                         "b" + comps_coord[d] + "_" + sp_str,
                                         read_every,
                                         comps_idx[d],
                                         prtls->B);
      report_ok(nparticles, nread_b, "b" + comps_coord[d]);
    }
    prtls->setNactive(nparticles);
  }

  template class TristanV2<1>;
  template class TristanV2<2>;
  template class TristanV2<3>;

} // namespace rgnr
