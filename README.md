# Fusion Scientific Computations

This library contains computation mechanisms to support scientific computation.

For more information see the documentation at <https://alexrobomind.github.io/fusionsc>

## Regular Setup (pip)

FusionSC can be directly installed from pypi. This will also install a binary redistributable for the fusionsc executable.
```
pip install fusionsc
```

Alternatively, you can also install from the source (if you have suitable compilers for your python version installed)

```
git clone https://jugit.fz-juelich.de/a.knieps/fsc
cd fsc
pip install .
```

## Development setup

### Preparing & performing the build

To compile the standalone `fusionsc` executable, you need CMake and a suitable host compiler.
On Linux, openssl might additionaly be required.

Linux build (outputs to build/src/c++/tools/fusionsc):

```
git clone https://jugit.fz-juelich.de/a.knieps/fsc
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ../fsc
cmake --build . --target {targetName}
```

Windows build:

```
git clone https://jugit.fz-juelich.de/a.knieps/fsc
mkdir build
cd build
cmake ../fsc
cmake --build . --target {targetName} --config Release
```

The following targets are of particular relevance:

| Function         |  Target name        | Output location        |
| ---------------- | ------------------- |----------------------- |
| `fusionsc` tool  |  fsc-tool           | {build}/src/c++/tools  |
| python bindings  |  copy-pybindings    | {src}/python/fusionsc  |
| tests            |  tests              | {build}/src/c++        |
| Capnp File ID    |  capnp-id           | Console                |  

### Setting a development install

In order to have a development install compatible with your python version, you need a .pth file in your site-packages dir pointing to the src/python directory to locate
the fusionsc package. The util/dev-install.py script will set one up for you:

```
python util/dev-install.py
```