import pathlib

fpath = pathlib.Path(__file__).parent.resolve()

import os
import sys

sys.path.append(os.path.join(fpath, "..", "..", "build"))

import matplotlib.pyplot as plt
import numpy as np
import ragnar as rg

rg.Initialize()

rng = np.random.default_rng(123)


def random_plaw(size, xmin, xmax, p, rng=np.random.default_rng(123)):
    return (
        (xmax ** (p + 1) - xmin ** (p + 1)) * rng.random(size) + xmin ** (p + 1)
    ) ** (1 / (p + 1))


nprtls = int(1e5)

rnd1 = 2 * (rng.random(nprtls) - 0.5)
rnd2 = 2 * np.pi * rng.random(nprtls)
b1s = np.sqrt(1 - rnd1**2) * np.cos(rnd2)
b2s = np.sqrt(1 - rnd1**2) * np.sin(rnd2)
b3s = rnd1

prtls = rg.Particles_3D("pairs")
prtls.fromArrays(
    {
        "U1": random_plaw(size=nprtls, xmin=1, xmax=100, p=-2, rng=rng),
        "B1": b1s,
        "B2": b2s,
        "B3": b3s,
    }
)

dist_prtls = prtls.energyDistribution(rg.Logbins(1, 1e3, 100))
dist_prtls = prtls.energyDistribution(rg.Logbins(1, 1e3, 100))


bins_e_syn = rg.Logbins(0.01, 1e5, 200, rg.EnergyUnits.mec2)
e_syn_2_f_syn = rg.SynchrotronSpectrum_3D(prtls, bins_e_syn, 1, 1, 1)

e_syn_2_f_syn_fromdist = rg.SynchrotronSpectrumFromDist(dist_prtls, bins_e_syn, 1, 1)

plt.plot(bins_e_syn.as_array(), 1.3 * e_syn_2_f_syn.as_array())
plt.plot(bins_e_syn.as_array(), e_syn_2_f_syn_fromdist.as_array())
plt.xscale("log")
plt.yscale("log")
plt.savefig("sync-prtls-test.png")
