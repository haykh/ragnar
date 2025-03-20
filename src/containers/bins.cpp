#include "containers/bins.hpp"

#include "utils/snippets.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <string>

namespace py = pybind11;
using namespace pybind11::literals;

namespace rgnr {

  auto Linbins(real_t             start,
               real_t             stop,
               std::size_t        num,
               const std::string& unit) -> Bins {
    auto arr       = Bins(Linspace(start, stop, num), unit);
    arr.log_spaced = false;
    return arr;
  }

  auto Logbins(real_t             start,
               real_t             stop,
               std::size_t        num,
               const std::string& unit) -> Bins {
    auto arr       = Bins(Logspace(start, stop, num), unit);
    arr.log_spaced = true;
    return arr;
  }

  void pyDefineBins(py::module& m) {
    py::class_<Bins, Array1D<real_t>>(m, "Bins")
      .def(py::init<const Array1D<real_t>&, const std::string&>(),
           "arr"_a,
           "unit"_a = "")
      .def(py::init<const std::string&>(), "unit"_a = "")
      .def(py::init<const py::array_t<real_t>&, const std::string&>(),
           "arr"_a,
           "unit"_a = "")
      .def_readwrite("log_spaced", &Bins::log_spaced)
      .def_readwrite("unit", &Bins::unit)
      .doc() = R"rgnrdoc(
        A special type of array that stores bins for a histogram

        Attributes
        ----------
        unit : str
          Physical units of the array (if any)

        log_spaced : bool
          Whether the bins are logarithmically spaced

        Methods
        -------
        head(n=10, start=0)
          Print the first n elements of the array starting from index start
      
        as_array() -> np.ndarray
          Return a numpy array
        
        extent(d=0) -> int
          Return the extent of the array along dimension d
      )rgnrdoc";
    m.def("Linbins", &Linbins, "start"_a, "stop"_a, "num"_a, "unit"_a = "", R"rgnrdoc(
        Create linearly spaced binning

        Parameters
        ----------
        start : float
          The start of the range

        stop : float
          The end of the range

        num : int
          The number of elements in the range

        unit : str, optional
          The unit of the range [default: ""]

        Returns
        -------
        Bins
          Bins with `num` elements linearly spaced between `start` and `stop`
        )rgnrdoc");
    m.def("Logbins", &Logbins, "start"_a, "stop"_a, "num"_a, "unit"_a = "", R"rgnrdoc(
        Create logarithmically spaced bins with `num` elements between `start` and `stop`

        Parameters
        ----------
        start : float
          The start of the range

        stop : float
          The end of the range

        num : int
          The number of elements in the range

        unit : str, optional
          The unit of the range [default: ""]

        Returns
        -------
        Bins
          Bins with `num` elements logarithmically spaced between `start` and `stop`
        )rgnrdoc");
  }

} // namespace rgnr
