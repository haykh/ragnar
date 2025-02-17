#include "simulation.h"

#include <Kokkos_Core.hpp>

#include <iostream>

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  try {
    rgnr::Simulation(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}
