import ragnar as rg
import numpy as np

rg.Initialize()


def check_slope(xs, ys, p, xmin, xmax, atol=1e-2):
    mask = (xs > xmin) & (xs < xmax)
    pfit, _ = np.polyfit(np.log10(xs[mask]), np.log10(ys[mask]), 1)
    print(pfit, p)
    return np.isclose(p, pfit, atol=atol)


def test_ic_log():
    e_break = 1e-8

    p = 1.5
    dist_prtls = rg.TabulatedDistribution(
        rg.Logbins(1e3, 1e7, 200),
        rg.PlawGenerator(-p, 1e3, 1e7),
    )
    dist_soft_photons = rg.TabulatedDistribution(
        rg.Logbins(1e-11, 1e-7, 200, rg.EnergyUnits.mec2),
        rg.DeltaGenerator(e_break, e_break / 10),
    )

    x_prtls = dist_prtls.EnergyBins().as_array()
    y_prtls = dist_prtls.F().as_array()
    assert check_slope(
        x_prtls, y_prtls, -p, 1e3, 1e7
    ), "Prtls power law slope check failed"

    x_soft_photons, y_soft_photons = (
        dist_soft_photons.EnergyBins().as_array(),
        dist_soft_photons.F().as_array(),
    )

    bins_eic = rg.Bins(rg.Logspace(1e3, 1e7, 200), rg.EnergyUnits.mec2)

    eic_2_f_ic = rg.ICSpectrum(dist_prtls, dist_soft_photons, bins_eic)

    x_ic, y_ic = bins_eic.as_array(), eic_2_f_ic.as_array()

    assert check_slope(
        x_ic, y_ic, -p / 2 + 3 / 2, 2e3, 2e4
    ), "IC power law slope check failed"

    assert np.isclose(
        x_ic[np.argmax(y_ic)], e_break * 1e7**2, rtol=0.1
    ), "IC peak energy check failed"
