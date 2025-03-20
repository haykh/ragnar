import ragnar as rg
import numpy as np

rg.Initialize()


def test_linspace_logspace():
    assert all(
        np.isclose(
            rg.Linspace(-32.0, 4832.0, 56).as_array(), np.linspace(-32.0, 4832.0, 56)
        )
    ), "linspace test failed"

    assert all(
        np.isclose(
            rg.Logspace(10**-2.5, 10**2.5, 213).as_array(), np.logspace(-2.5, 2.5, 213)
        )
    ), "logspace test failed"

    linbins = rg.Linbins(55.0, 56.0, 123)
    assert all(
        np.isclose(linbins.as_array(), np.linspace(55.0, 56.0, 123))
    ), "linbins test failed"
    assert not linbins.log_spaced, "linbins have log spacing"

    logbins = rg.Logbins(1, 1e3, 22)
    assert all(
        np.isclose(logbins.as_array(), np.logspace(0, 3, 22))
    ), "logbins test failed"
    assert logbins.log_spaced, "logbins have log spacing"

    assert all(
        np.isclose(
            rg.Logspace(10**-2.5, 10**2.5, 213).as_array(), np.logspace(-2.5, 2.5, 213)
        )
    ), "logspace test failed"
