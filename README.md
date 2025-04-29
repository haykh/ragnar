# Ragnar [ᚱᛅᚴᚾᛅᚱᛦ]
## Radiative post-processing code

Written in C++ with `Kokkos` portability and `python` API via `pybind11`. Supports loading data from PIC simulations for post-processing.

## Usage

Download the code with:

```sh
git clone --recursive https://github.com/haykh/ragnar.git
cd ragnar
```

You can build/install the package with pip using the following command:

```sh
pip install . --config-settings="cmake.args=<FLAGS>"
```

All the CMake flags can be passed in the `cmake.args` as shown above, e.g., `cmake.args=-DRAGNAR_USE_HDF5:BOOL=OFF;-DKokkos_ENABLE_OPENMP:BOOL=ON;...`. 

Otherwise, you can also compile the code to produce the shared library directly with CMake:

```sh
cmake -B build [-D RAGNAR_USE_HDF5:BOOL=OFF] [-D Kokkos_ENABLE_CUDA:BOOL=ON] ...
cmake --build build -j
```

Compilation produces a shared library file `.so` (located in `build/`) which can be imported directly in `python`. 

> If `Kokkos` is installed externally, no special `-DKokkos_***` flags are needed.

> If you don't intend to use `hdf5` for reading the simulation data, you may disable it by providing a flag `-DRAGNAR_USE_HDF5:BOOL=OFF` (it is set to `ON` by default).

Usage examples together with unit tests are located in `src/examples` and `src/tests`.

```python
import ragnar as rg
rg.Initialize(); # always remember to initialize Kokkos
```

### Generating/manipulating synthetic data

You can generate synthetic data using built-in functions:

```python
dist_prtls = rg.TabulatedDistribution(
    rg.Logbins(1, 100, 200), rg.PlawGenerator(-2, 1, 100)
)
```

After that, you may use this data further; for instance, to generate a synchrotron signal from a given distribution of particles:

```python
# first, generate the photon bins
bins_e_syn = rg.Logbins(0.01, 1e7, 200)
bins_e_syn.unit = rg.EnergyUnits.mec2

e_syn_2_f_syn = rg.SynchrotronSpectrumFromDist(dist_prtls, bins_e_syn, 1, 1)

# docstring can be accessed via `rg.SynchrotronSpectrumFromDist?`
```

Then you can plot the generated data:

```python
plt.plot(bins_e_syn.as_array(), e_syn_2_f_syn.as_array())
plt.xscale("log")
plt.yscale("log")
```

### Post-processing output data

All functions have docstrings which can be accessed via, e.g., `rg.SynchrotronSpectrum_3D?`.

While functions in `rg` can be accessed with any user-defined, interaction with the simulation data is done via the so-called plugins. Below is an example usage for `Tristan v2` plugin:

```python
plugin = rg.TristanV2_3D()
plugin.setPath("<PATH_TO_DATA>") # directory where the `output` is
plugin.setStep(55)

electrons = plugin.readParticles("e-", 1) # 1 -- is the species index (starting from 1)
positrons = plugin.readParticles("e+", 2)
protons = plugin.readParticles("p", 3)
```

`.readParticles` returns a `Particles` object, which is a special container to store all the read data (if compiled with GPU support, data is stored only on the GPU). With this object, one can, for instance, compute an energy distribution for the given species:

```python
gbins = rg.Logbins(1e-2, 1e3, 200) # define gamma * beta bins
e_dist = electrons.energyDistribution(gbins)

# plot d N / d (gamma * beta)
plt.plot(e_dist.EnergyBins().as_array(), e_dist.F().as_array())
plt.xscale("log")
plt.yscale("log")
```

### Unit tests

The code also has a set of unit tests that can be run after compilation using:

```sh
ctest --test-dir build
```

These tests are also ran using GitHub actions automatically on every push.

## Dependencies

All the dependencies (except for `pybind11`) can be built in-tree (except for the HDF5), however, it is recommended to install them externally to speed up compilation.

- [`Kokkos`](https://github.com/kokkos/kokkos)
- [`HighFive`](https://github.com/highfive-devs/highfive) : optional (used when `-D RAGNAR_USE_HDF5=ON`)
- [`pybind11`](https://github.com/pybind/pybind11) : no need to install

> `pybind11` is downloaded with the code when you clone with `--recursive`.

### Using `spack`

You may install all of the dependencies using the `spack` package manager:

```sh
spack env create ragnar
spack env activate ragnar
spack install --add kokkos [+cuda] [+wrapper] [cuda_arch=...]
spack install --add highfive
```
