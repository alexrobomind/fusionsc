# Fusion Scientific Computations

This library contains computation mechanisms to support scientific computation.

## Setup (python)

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

### Preparing the build

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

If you want to setup a python development install, you need to also run the util/dev-install.py script (that will create a .pth file in your site-packages dir).

The following targets are available:

| Function         |  Target name        | Output location        |
| ---------------- | ------------------- |----------------------- |
| `fusionsc` tool  |  fsc-tool           | {build}/src/c++/tools  |
| python bindings  |  copy-pybindings    | {src}/python/fusionsc  |
| tests            |  tests              | {build}/src/c++        |

### Building the correct targets

If you want to run / build tests, you need to build the `tests` target.
