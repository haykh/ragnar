import ragnar as rg
import numpy as np

rg.Initialize()


def check_slope(xs, ys, p, xmin, xmax, atol=1e-2):
    mask = (xs > xmin) & (xs < xmax)
    pfit, _ = np.polyfit(np.log10(xs[mask]), np.log10(ys[mask]), 1)
    print(pfit, p)
    return np.isclose(p, pfit, atol=atol)


def test_sync_log():
    p = 2.23
    prtl_dist = rg.TabulatedDistribution(
        rg.Logbins(1, 1000, 200), rg.PlawGenerator(-p, 1, 1000)
    )

    esync_bins = rg.Logbins(0.01, 1e7, 200)
    esync_bins.unit = rg.EnergyUnits.mec2

    esync2_dn_desync = rg.SynchrotronSpectrumFromDist(prtl_dist, esync_bins, 1, 1)

    x_prtls = prtl_dist.EnergyBins().as_array()
    y_prtls = prtl_dist.F().as_array()

    x_sync = esync_bins.as_array()
    y_sync = esync2_dn_desync.as_array()

    assert check_slope(x_prtls, y_prtls, -p, 2, 800), "Particle slope is wrong"
    assert check_slope(x_sync, y_sync, -p / 2 + 3 / 2, 10, 1e4), "Sync slope is wrong"
    assert check_slope(
        x_sync, y_sync, 1 / 3 + 1, 3e-2, 2e-1, atol=0.1
    ), "Sync low-energy slope is wrong"


def test_sync_lin():
    p = 2.5
    dist_prtls = rg.TabulatedDistribution(
        rg.Linbins(1, 1000, 10000), rg.PlawGenerator(-p, 1, 1000)
    )

    bins_esyn = rg.Logbins(0.01, 1e6, 500)
    bins_esyn.unit = rg.EnergyUnits.mec2

    e_syn_2_f_syn = rg.SynchrotronSpectrumFromDist(dist_prtls, bins_esyn, 1, 1)

    x_prtls = dist_prtls.EnergyBins().as_array()
    y_prtls = dist_prtls.F().as_array()
    assert check_slope(x_prtls, y_prtls, -p, 2, 800), "Particle slope is wrong"

    x_sync = bins_esyn.as_array()
    y_sync = e_syn_2_f_syn.as_array()
    assert check_slope(x_sync, y_sync, -p / 2 + 3 / 2, 10, 1e4), "Sync slope is wrong"

    assert check_slope(
        x_sync, y_sync, 1 / 3 + 1, 3e-2, 2e-1, atol=0.1
    ), "Sync low-energy slope is wrong"
