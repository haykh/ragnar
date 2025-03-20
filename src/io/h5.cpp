#include "io/h5.hpp"

#include "containers/array.hpp"

#include <Kokkos_Core.hpp>
#include <highfive/highfive.hpp>
#include <pybind11/pybind11.h>

#include <string>

namespace py = pybind11;
using namespace pybind11::literals;

namespace rgnr::io::h5 {

  template <typename T>
  auto Read1DArray(const std::string& filename,
                   const std::string& dsetname,
                   std::size_t        size,
                   std::size_t        stride) -> Array1D<T> {
    auto file    = HighFive::File(filename, HighFive::File::ReadOnly);
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
    py::print("Reading", dsetname, "from", filename, "...", "end"_a = "", "flush"_a = true);
    Array1D<T> array;
    array.data   = Kokkos::View<T*> { dsetname, size };
    auto array_h = Kokkos::create_mirror_view(array.data);
    try {
      dataset.select({ 0 }, { size }, { stride }).template read<T>(array_h.data());
    } catch (const std::exception& e) {
      py::print("Error reading", dsetname, "from", filename, ":", e.what());
      throw e;
    }
    Kokkos::deep_copy(array.data, array_h);
    py::print(": OK", "flush"_a = true);
    return array;
  }

  template <typename T>
  void Write1DArray(const std::string& filename,
                    const std::string& dsetname,
                    const Array1D<T>&  array) {
    HighFive::File file(filename,
                        HighFive::File::ReadWrite | HighFive::File::Create);
    py::print("Writing", dsetname, "to", filename, "...", "end"_a = "", "flush"_a = true);
    const auto array_h = Kokkos::create_mirror_view(array.data);
    Kokkos::deep_copy(array_h, array.data);
    try {
      file
        .createDataSet<T>(dsetname, HighFive::DataSpace({ array.data.extent(0) }))
        .write_raw(array_h.data());
    } catch (const std::exception& e) {
      py::print("Error writing", dsetname, "to", filename, ":", e.what());
      throw e;
    }
    py::print(": OK", "flush"_a = true);
  }

  template <typename T>
  void pyDefineRead1DArray(py::module& m) {
    m.def(("H5read1DArray_" + std::string(typeid(T).name())).c_str(),
          &Read1DArray<T>,
          "filename"_a,
          "dsetname"_a,
          "size"_a   = 0,
          "stride"_a = 1,
          R"rgnrdoc(
        Read a 1D array from an HDF5 file

        Parameters
        ----------
        filename : str
          The name of the HDF5 file
        
        dsetname : str
          The name of the dataset to read

        size : int, optional
          The size of the array to read; if 0, the size of the dataset is used [default: 0]

        stride : int, optional
          The stride to use when reading the dataset [default: 1]
      )rgnrdoc");
  }

  template <typename T>
  void pyDefineWrite1DArray(py::module& m) {
    m.def(("H5write1DArray_" + std::string(typeid(T).name())).c_str(),
          &Write1DArray<T>,
          "filename"_a,
          "dsetname"_a,
          "array"_a,
          R"rgnrdoc(
        Write a 1D array to an HDF5 file

        Parameters
        ----------
        filename : str
          The name of the HDF5 file

        dsetname : str
          The name of the dataset to write

        array : Array
          The array to write
      )rgnrdoc");
  }

  template auto Read1DArray<int>(const std::string&,
                                 const std::string&,
                                 std::size_t,
                                 std::size_t) -> Array1D<int>;
  template auto Read1DArray<float>(const std::string&,
                                   const std::string&,
                                   std::size_t,
                                   std::size_t) -> Array1D<float>;
  template auto Read1DArray<double>(const std::string&,
                                    const std::string&,
                                    std::size_t,
                                    std::size_t) -> Array1D<double>;

  template void Write1DArray<int>(const std::string&,
                                  const std::string&,
                                  const Array1D<int>&);
  template void Write1DArray<float>(const std::string&,
                                    const std::string&,
                                    const Array1D<float>&);
  template void Write1DArray<double>(const std::string&,
                                     const std::string&,
                                     const Array1D<double>&);

  template void pyDefineRead1DArray<int>(py::module&);
  template void pyDefineRead1DArray<float>(py::module&);
  template void pyDefineRead1DArray<double>(py::module&);

  template void pyDefineWrite1DArray<int>(py::module&);
  template void pyDefineWrite1DArray<float>(py::module&);
  template void pyDefineWrite1DArray<double>(py::module&);

} // namespace rgnr::io::h5
