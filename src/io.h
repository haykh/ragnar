#ifndef IO_H
#define IO_H

#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>

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

  template <typename T>
  void
  readArray(const char* filename, const char* variable, std::vector<T>& data) {
    data = std::vector<T>();
    HighFive::File file(filename, HighFive::File::ReadOnly);
    HighFive::DataSet dataset = file.getDataSet(variable);
    dataset.read(data);
  }
} // namespace IO

#endif // IO_WRITE_H
