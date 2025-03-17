#include "plugins/tristan-v2.hpp"

#include "utils/snippets.h"

#include "io/h5.hpp"

#include <Kokkos_Core.hpp>
#include <highfive/highfive.hpp>
#include <pybind11/pybind11.h>

#include <array>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace py = pybind11;
using namespace pybind11::literals;

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

  template <typename T, class A>
  auto read1DArray(HighFive::File     file,
                   const std::string& dsetname,
                   A                  arr_h,
                   std::size_t        size,
                   std::size_t        stride) -> std::size_t {
    auto dataset = file.getDataSet(dsetname);
    auto dims    = dataset.getDimensions();
    if (dims.size() != 1) {
      throw std::runtime_error("Dataset is not 1D");
    }
    if (stride == 0) {
      throw std::runtime_error("Stride must be greater than 0");
    } else if (size == 0) {
      size = dims[0];
    } else if (dims[0] / stride > size) {
      throw std::runtime_error(
        "Number of read quantity exceeds allocated space");
    }
    dataset.select({ 0 }, { size }, { stride }).template read<T>(arr_h.data());
    return size;
  }

  template <unsigned short N>
  auto readPrtlQuantity(HighFive::File            file,
                        const std::string&        quantity,
                        std::size_t               stride,
                        std::size_t               idx,
                        Kokkos::View<real_t* [N]> arr) -> std::size_t {
    auto arr_h = Kokkos::create_mirror_view(Kokkos::subview(arr, Kokkos::ALL, idx));
    const auto nread = read1DArray<real_t, decltype(arr_h)>(file,
                                                            quantity,
                                                            arr_h,
                                                            arr.extent(0),
                                                            stride);
    Kokkos::deep_copy(Kokkos::subview(arr, Kokkos::ALL, idx), arr_h);
    return nread;
  }

  template <dim_t D>
  auto TristanV2<D>::readParticles(const std::string& label,
                                   unsigned short     sp,
                                   std::size_t        start,
                                   std::size_t        size,
                                   std::size_t        stride,
                                   bool ignore_coordinates) const -> Particles<D> {
    if (stride == 0) {
      throw std::runtime_error("Stride must be greater than 0");
    }
    auto prtls = Particles<D> { label };

    const auto        step   = std::to_string(getStep());
    const auto        sp_str = std::to_string(sp);
    const std::string fname  = getPath() + "/output/prtl/prtl.tot." +
                              std::string(5 - step.length(), '0') + step;

    py::print("Reading particles #", sp, "from", fname, "...", "flush"_a = true);

    HighFive::File file { fname, HighFive::File::ReadOnly };

    const auto comps_coord = std::array<std::string, 3> { "x", "y", "z" };
    const auto comps_vel   = std::array<std::string, 3> { "u", "v", "w" };
    const auto comps_idx   = std::array<in, 3> { in::x, in::y, in::z };

    prtls.setIgnoreCoords(ignore_coordinates);

    const std::size_t ntotal = file.getDataSet("x_" + sp_str).getDimensions()[0];
    if (start + size >= ntotal) {
      throw std::runtime_error("start + size >= total number of particles");
    }
    const std::size_t nparticles =
      (size == 0) ? file.getDataSet("x_" + sp_str).getDimensions()[0] / stride
                  : size;
    prtls.allocate(nparticles);

    py::print(" found",
              ToHumanReadable(ntotal, USE_POW10),
              "particles, reading",
              ToHumanReadable(nparticles, USE_POW10),
              "starting from",
              start,
              "flush"_a = true);

    const auto report_ok =
      [](std::size_t np, std::size_t nread, const std::string& quantity) {
        if (np != nread) {
          throw std::runtime_error("Number of particles mismatch");
        }
        py::print(" ", quantity, ": OK", "flush"_a = true);
      };

    if (not ignore_coordinates) {
      for (auto d = 0u; d < D; ++d) {
        auto nread_coord = readPrtlQuantity<D>(file,
                                               comps_coord[d] + "_" + sp_str,
                                               read_every,
                                               comps_idx[d],
                                               prtls.X);
        report_ok(nparticles, nread_coord, comps_coord[d]);
      }
    }
    for (auto d = 0u; d < 3u; ++d) {
      auto nread_vel = readPrtlQuantity<3>(file,
                                           comps_vel[d] + "_" + sp_str,
                                           read_every,
                                           comps_idx[d],
                                           prtls.U);
      report_ok(nparticles, nread_vel, comps_vel[d]);
      auto nread_e = readPrtlQuantity<3>(file,
                                         "e" + comps_coord[d] + "_" + sp_str,
                                         read_every,
                                         comps_idx[d],
                                         prtls.E);
      report_ok(nparticles, nread_e, "e" + comps_coord[d]);
      auto nread_b = readPrtlQuantity<3>(file,
                                         "b" + comps_coord[d] + "_" + sp_str,
                                         read_every,
                                         comps_idx[d],
                                         prtls.B);
      report_ok(nparticles, nread_b, "b" + comps_coord[d]);
    }
    prtls.setNactive(nparticles);
    return prtls;
  }

  template <dim_t D>
  void pyDefineTristanV2Plugin(py::module& m) {
    py::class_<TristanV2<D>>(m, ("TristanV2_" + std::to_string(D) + "D").c_str())
      .def(py::init<>())
      .def("label", &TristanV2<D>::label)
      .def("setPath", &TristanV2<D>::setPath)
      .def("setStep", &TristanV2<D>::setStep)
      .def("getPath", &TristanV2<D>::getPath)
      .def("getStep", &TristanV2<D>::getStep)
      .def("readParticles",
           &TristanV2<D>::readParticles,
           "label"_a,
           "sp"_a,
           "read_every"_a         = 1,
           "ignore_coordinates"_a = false,
           R"rgnrdoc(
              Read particles from Tristan V2 output

              Parameters
              ----------
              label : str
                Label for the particles

              sp : int
                Species number

              read_every : int, optional
                Read every nth particle [default: 1]

              ignore_coordinates : bool, optional
                Ignore particle coordinates [default: False]

              Returns
              -------
              Particles_1D, Particles_2D, Particles_3D
                Particle container
          )rgnrdoc")
      .doc() = R"rgnrdoc(
              Read particles from Tristan V2 output

              Parameters
              ----------
              label : str
                Label for the particles

              sp : int
                Species number

              read_every : int, optional
                Read every nth particle [default: 1]

              ignore_coordinates : bool, optional
                Ignore particle coordinates [default: False]

              Returns
              -------
              Particles_1D, Particles_2D, Particles_3D
                Particle container
          )rgnrdoc";
  }

  template class TristanV2<1>;
  template class TristanV2<2>;
  template class TristanV2<3>;

  template void pyDefineTristanV2Plugin<1>(py::module&);
  template void pyDefineTristanV2Plugin<2>(py::module&);
  template void pyDefineTristanV2Plugin<3>(py::module&);

} // namespace rgnr
