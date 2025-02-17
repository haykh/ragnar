#ifndef PLUGINS_TRISTAN_V2_HPP
#define PLUGINS_TRISTAN_V2_HPP

#include "utils/types.h"

#include "containers/particles.hpp"
#include "plugins/plugin.hpp"

#include <stdexcept>
#include <string>

namespace rgnr {

  template <dim_t D>
  class TristanV2 : public Plugin<D> {
    std::string m_path;
    std::size_t m_step;

    bool is_path_set { false };
    bool is_step_set { false };

  public:
    TristanV2() : Plugin<D>() {}

    void readParticles(unsigned short,
                       Particles<D>*,
                       std::size_t = 1,
                       bool        = not IGNORE_COORDS) const override;

    auto label() const -> std::string override;

    // setters
    void setPath(const std::string&);
    void setStep(std::size_t);

    // getters
    auto getPath() const -> std::string;
    auto getStep() const -> std::size_t;
  };

} // namespace rgnr

#endif // PLUGINS_TRISTAN_V2_HPP
