import ragnar as rg
import numpy as np

rg.Initialize()


def test_arrays_bins():
    array = np.arange(123)
    assert all(
        np.isclose(rg.Array1D_i(array.astype(int)).as_array(), array)
    ), "Array1D_i test failed"
    assert all(
        np.isclose(rg.Array1D_f(array.astype(np.float32)).as_array(), array)
    ), "Array1D_f test failed"
    assert all(
        np.isclose(rg.Array1D_d(array.astype(np.float64)).as_array(), array)
    ), "Array1D_d test failed"

    assert (
        rg.Array1D_d(array.astype(np.float64)).extent() == 123
    ), "Array1D extent test failed"

    bins = rg.Bins(np.logspace(-2.5, 2.5, 213))
    bins.log_spaced = True
    bins.unit = rg.EnergyUnits.mec2
    assert bins.unit == rg.EnergyUnits.mec2, "Bins unit test failed"
    assert bins.extent() == 213, "Bins extent test failed"
    assert bins.log_spaced, "Bins log_spaced test failed"
