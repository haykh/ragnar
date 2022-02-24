#include "global.h"
#include "sim.h"

#include <stdexcept>

auto main(int, char**) -> int {
  try {
    Simulation sim;
    sim.run();
  }
  catch (std::exception& err) {
    std::cerr << err.what() << std::endl;
    return -1;
  }
  return 0;
}
