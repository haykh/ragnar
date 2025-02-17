#ifndef IO_H5_HPP
#define IO_H5_HPP

#include <Kokkos_Core.hpp>
#include <highfive/highfive.hpp>

#include <map>
#include <string>
#include <utility>

namespace rgnr::io::h5 {

  template <typename T, typename A>
  inline auto Read1DArray(HighFive::File     file,
                          const std::string& quantity,
                          A                  array,
                          std::size_t        size,
                          std::size_t        stride) -> std::size_t {
    auto dataset = file.getDataSet(quantity);
    auto dims    = dataset.getDimensions();
    if (dims[0] / stride > size) {
      throw std::runtime_error(
        "Number of read quantity exceeds allocated space");
    }
    dataset.select({ 0 }, { size }, { stride }).template read<T>(array.data());
    return dims[0] / stride;
  }

  template <typename T, typename A>
  inline void Write1DArrays(HighFive::File     file,
                            const std::string& name,
                            A                  array,
                            std::size_t        size) {
    std::cout << "Writing " << name << " to " << file.getName() << " ..."
              << std::endl;
    auto dataset = file.createDataSet<T>(name, HighFive::DataSpace({ size }));
    dataset.write_raw(array.data());
  }

} // namespace rgnr::io::h5

#endif // IO_H5_HPP
