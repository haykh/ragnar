import ragnar as rg
import numpy as np

rg.Initialize()


def test_distributions():
    for norm in [rg.Logspace, rg.Linspace]:
        bins = rg.Bins(norm(1e-2, 1, int(1e6)))
        dist = rg.TabulatedDistribution(bins, rg.DeltaGenerator(2e-2, 0.01))
        assert np.isclose(
            np.trapezoid(dist.F().as_array(), dist.EnergyBins().as_array()), 1
        ), "DeltaDistribution normalization test failed"

    for norm in [rg.Logspace, rg.Linspace]:
        bins = rg.Bins(norm(1e-3, 10, int(1e6)))
        dist = rg.TabulatedDistribution(
            bins, rg.BrokenPlawGenerator(0.3, 0.23, -1.0, 1e-2, 2)
        )
        assert np.isclose(
            np.trapezoid(dist.F().as_array(), dist.EnergyBins().as_array()), 1
        ), "BrokenPlawDistribution normalization test failed"

    for norm in [rg.Logspace, rg.Linspace]:
        bins = rg.Bins(norm(1e-2, 1, int(1e6)))
        dist = rg.TabulatedDistribution(bins, rg.PlawGenerator(-1.2, 1e-2, 1))
        assert np.isclose(
            np.trapezoid(dist.F().as_array(), dist.EnergyBins().as_array()), 1
        ), f"PlawDistribution normalization test failed"
