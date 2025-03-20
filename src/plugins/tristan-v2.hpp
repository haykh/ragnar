#ifndef PLUGINS_TRISTAN_V2_HPP
#define PLUGINS_TRISTAN_V2_HPP

#include "utils/global.h"

#include "containers/particles.hpp"

#include <pybind11/pybind11.h>

#include <string>

namespace py = pybind11;

namespace rgnr {

  template <dim_t D>
  class TristanV2 {
    std::string m_path;
    std::size_t m_step;

    bool is_path_set { false };
    bool is_step_set { false };

  public:
    TristanV2() {}

    auto readParticles(const std::string&,
                       unsigned short,
                       std::size_t = 0,
                       std::size_t = 0,
                       std::size_t = 1,
                       bool        = not IGNORE_COORDS) const -> Particles<D>;

    auto label() const -> std::string;

    // setters
    void setPath(const std::string&);
    void setStep(std::size_t);

    // getters
    auto getPath() const -> std::string;
    auto getStep() const -> std::size_t;
  };

  template <dim_t D>
  void pyDefineTristanV2Plugin(py::module&);

} // namespace rgnr

#endif // PLUGINS_TRISTAN_V2_HPP
