#ifndef PLUGINS_PLUGIN_HPP
#define PLUGINS_PLUGIN_HPP

#include "utils/types.h"

#include "containers/particles.hpp"

#include <string>

namespace rgnr {

  template <dim_t D>
  class Plugin {
  public:
    Plugin() = default;

    virtual void readParticles(unsigned short,
                               Particles<D>*,
                               std::size_t = 1,
                               bool        = not IGNORE_COORDS) const = 0;

    virtual auto label() const -> std::string = 0;
  };

} // namespace rgnr

#endif // PLUGINS_PLUGIN_HPP
