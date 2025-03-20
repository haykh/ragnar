import pathlib

fpath = pathlib.Path(__file__).parent.resolve()

import os
import sys

sys.path.append(os.path.join(fpath, "..", "..", "build"))

import matplotlib.pyplot as plt
import numpy as np
import ragnar as rg

rg.Initialize()


e_break = 1e-8

p = 1.5


def gen_spec(gfunc, gmin, gmax, gbins, efunc, emin, emax, ebins):
    dist_prtls = rg.TabulatedDistribution(
        gfunc(gmin, gmax, gbins), rg.PlawGenerator(-p, gmin, gmax)
    )

    dist_soft_photons = rg.TabulatedDistribution(
        rg.Logbins(1e-11, 1e-7, 200, rg.EnergyUnits.mec2),
        rg.DeltaGenerator(e_break, e_break / 10),
    )

    bins_eic = rg.Bins(efunc(emin, emax, ebins), rg.EnergyUnits.mec2)

    eic_2_f_ic = rg.ICSpectrum(dist_prtls, dist_soft_photons, bins_eic)

    x_prtls = dist_prtls.EnergyBins().as_array()
    y_prtls = dist_prtls.F().as_array()

    x_soft_photons, y_soft_photons = (
        dist_soft_photons.EnergyBins().as_array(),
        dist_soft_photons.F().as_array(),
    )

    x_ic, y_ic = bins_eic.as_array(), eic_2_f_ic.as_array()

    return x_prtls, y_prtls, x_soft_photons, y_soft_photons, x_ic, y_ic


fig = plt.figure(figsize=(14, 4), dpi=300)
gs = fig.add_gridspec(1, 3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])


x_prtls, y_prtls, x_soft_photons, y_soft_photons, x_ic, y_ic = gen_spec(
    rg.Logbins, 1e3, 1e7, 200, rg.Logbins, 2e3, 2e4, 200
)

ax1.plot(x_prtls, y_prtls)
ax2.plot(x_soft_photons, y_soft_photons)
ax3.plot(x_ic, y_ic)

x_prtls, y_prtls, x_soft_photons, y_soft_photons, x_ic, y_ic = gen_spec(
    rg.Linbins, 1e3, 1e7, 2000, rg.Logbins, 2e3, 2e4, 200
)

ax1.plot(x_prtls, y_prtls)
ax3.plot(x_ic, y_ic * 1.1e5)

x_prtls, y_prtls, x_soft_photons, y_soft_photons, x_ic, y_ic = gen_spec(
    rg.Logbins, 1e3, 1e7, 2000, rg.Linbins, 2e3, 2e4, 200
)

ax1.plot(x_prtls, y_prtls)
ax3.plot(x_ic, y_ic * 0.1)

for ax in [ax1, ax2, ax3]:
    ax.set(xscale="log", yscale="log")

plt.savefig("ic-test.png")
