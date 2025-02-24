#ifndef IO_TOML_HPP
#define IO_TOML_HPP

#include <toml.hpp>

#include <string>

namespace toml11 = toml;

namespace rgnr::io::toml {

  auto ReadFile(const std::string& path) -> toml11::value {
    return toml11::parse(path);
  }

} // namespace rgnr::io::toml

#endif
