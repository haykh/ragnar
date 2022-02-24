#ifndef IO_H
#define IO_H

#include <HighFive/H5DataSet.hpp>
#include <HighFive/H5DataSpace.hpp>
#include <HighFive/H5File.hpp>

#include <vector>

namespace IO {
  template <typename T>
  void writeArray(const char* filename,
                  const char* variable,
                  const std::vector<T>& data) {
    HighFive::File file(filename,
                        HighFive::File::ReadWrite | HighFive::File::Create
                          | HighFive::File::Truncate);
    HighFive::DataSet dataset
      = file.createDataSet<T>(variable, HighFive::DataSpace::From(data));
    dataset.write(data);
  }
} // namespace IO

#endif // IO_WRITE_H

  // template <typename T>
  // void readDataIntoView(const char* filename,
  //                       const char* variable,
  //                       RagnArray<T*>& data,
  //                       const bool& append = false) {
  //   HighFive::File file(filename, HighFive::File::ReadOnly);
  //   HighFive::DataSet dataset = file.getDataSet(variable);
  //   {
  //     std::vector<T> read_data;
  //     dataset.read(read_data);
  //     Kokkos::RangePolicy<AccelExeSpace> range;
  //     if (append) {
  //       range = Kokkos::RangePolicy<AccelExeSpace>(
  //         data.size(), data.extent(0) + read_data.size());
  //       Kokkos::resize(data, data.extent(0) + read_data.size());
  //     } else {
  //       range = Kokkos::RangePolicy<AccelExeSpace>(0, read_data.size());
  //       Kokkos::resize(data, read_data.size());
  //     }
  //     Kokkos::parallel_for(
  //       "copy", range, Lambda(const int i) { data(i) = read_data[i]; });
  //   }
  // }