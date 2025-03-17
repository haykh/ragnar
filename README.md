# Ragnar [ᚱᛅᚴᚾᛅᚱᛦ]
## Radiative post-processing code for PIC simulations

Written in C++ with `Kokkos` portability and `python` API via `pybind11`.

## Usage

Download the code with:

```sh
git clone --recursive https://github.com/haykh/ragnar.git
cd ragnar
```

Compile with:

```sh
cmake -B build [-D Kokkos_ENABLE_CUDA=ON] ...
cmake --build build -j
```
> If `Kokkos` is installed externally, no special flags are needed here.

Compilation produces a shared library file `.so` (located in `build/src/`) which can be imported directly in `python`. 

```python
import ragnar as rg
rg.Initialize(); # always remember to initialize Kokkos
```

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
gbins = rg.LogspaceView(1e-2, 1e3, 200) # define gamma * beta bins
e_count = electrons.energyDistribution(gbins)

# plot d N / d (gamma * beta)
plt.plot(gbins.as_vector(), e_count.as_vector())
plt.xscale("log")
plt.yscale("log")
```

> Since `e_count` is an `Array` object with data potentially stored on the GPU, you need to first cast it to a vector before plotting (which essentially returns a python list).

## Dependencies

All the dependencies (except for `pybind11`) can be built in-tree (except for the HDF5), however, it is recommended to install them externally to speed up compilation.

- [`Kokkos`](https://github.com/kokkos/kokkos)
- [`HighFive`](https://github.com/highfive-devs/highfive)
- [`toml11`](https://github.com/ToruNiina/toml11)
- [`pybind11`](https://github.com/pybind/pybind11) : no need to install

> `pybind11` is downloaded with the code when you clone with `--recursive`.

### Using `spack`

You may install all the dependencies using the `spack` package manager:

```sh
spack env create ragnar
spack env activate ragnar
spack install --add kokkos [+cuda] [+wrapper] [gpu_arch=...]
spack install --add toml11
spack install --add highfive
```
