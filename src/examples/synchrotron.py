import pathlib

fpath = pathlib.Path(__file__).parent.resolve()

import os
import sys

sys.path.append(os.path.join(fpath, "..", "..", "build"))

import matplotlib.pyplot as plt
import numpy as np
import ragnar as rg

rg.Initialize()


def fit(x, p, x0, xorig, yorig):
    y0 = yorig[np.argmin(np.abs(xorig - x0))]
    return (x / x0) ** p * y0


def check_slope(xs, ys, p, xmin, xmax, atol=1e-2):
    mask = (xs > xmin) & (xs < xmax)
    pfit, _ = np.polyfit(np.log10(xs[mask]), np.log10(ys[mask]), 1)
    print(
        "FITTING",
        ys,
        ys[mask],
        xs,
        xs[mask],
        np.polyfit(np.log10(xs[mask]), np.log10(ys[mask]), 1),
    )
    print(pfit, p)
    return np.isclose(p, pfit, atol=atol)


fig = plt.figure(figsize=(9, 3), dpi=300)
gs = fig.add_gridspec(1, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])


p = 2.23


def gen_spec(gfunc, gmin, gmax, gbins, efunc, emin, emax, ebins):
    prtl_dist = rg.TabulatedDistribution(
        gfunc(gmin, gmax, gbins), rg.PlawGenerator(-p, gmin, gmax)
    )
    esync_bins = efunc(emin, emax, ebins)
    esync_bins.unit = rg.EnergyUnits.mec2

    esync2_dn_desync = rg.SynchrotronSpectrumFromDist(prtl_dist, esync_bins, 1, 1)

    x_prtls = prtl_dist.EnergyBins().as_array()
    y_prtls = prtl_dist.F().as_array()

    x_sync = esync_bins.as_array()
    y_sync = esync2_dn_desync.as_array()
    return x_prtls, y_prtls, x_sync, y_sync


x_prtls, y_prtls, x_sync, y_sync = gen_spec(
    rg.Logbins, 1, 1000, 600, rg.Logbins, 0.01, 1e7, 200
)

ax1.plot(x_prtls, y_prtls)
ax2.plot(x_sync, y_sync)

# Testing linear/log bins for different regions of the spectrum
x_prtls, y_prtls, x_sync, y_sync = gen_spec(
    rg.Logbins, 1, 1000, 600, rg.Logbins, 1, 10, 200
)

ax2.plot(x_sync, y_sync)

x_prtls, y_prtls, x_sync, y_sync = gen_spec(
    rg.Logbins, 1, 1000, 600, rg.Linbins, 1, 10, 200
)

ax2.plot(x_sync, y_sync)

x_prtls, y_prtls, x_sync, y_sync = gen_spec(
    rg.Logbins, 1, 1000, 600, rg.Linbins, 100, 2000, 200
)

ax2.plot(x_sync, y_sync)

x_prtls, y_prtls, x_sync, y_sync = gen_spec(
    rg.Linbins, 1, 1000, 600, rg.Linbins, 100, 2000, 200
)

ax2.plot(x_sync, y_sync * 150)

ax1.set(xlabel="$\\gamma$", ylabel="$f(\\gamma)$")
ax2.set(xlabel="$E$ [$mc^2$]", ylabel="$E^2 f(E)$")

for ax in [ax1, ax2]:
    ax.set(xscale="log", yscale="log")

plt.savefig("sync-test.png")
