# Note: This file needs to be built from the root directory

# Build setup
ARG CACHE_DIR=/tmp/fusionsc-build-cache
ARG WITH_CACHE=--mount=type=bind,source=${CACHE_DIR},target=/build-cache

# Container for libc++
FROM alpine AS macproxy

# Baseline FusionSC dependencies
RUN apk add g++ cmake ninja python3-dev openssl-dev libc++-dev clang linux-headers libucontext-dev py3-numpy-dev

COPY . /src
WORKDIR /build
RUN CC=clang CXX=clang++ cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DFSC_DEP_PREF_VENDORED=Off -DCMAKE_CXX_FLAGS=-stdlib=libc++ /src
RUN ninja tests fsc-python-bindings

# Baseline build container
FROM debian AS base

# Baseline FusionSC dependencies
RUN apt-get -q update
RUN apt-get install -y g++ cmake ninja-build libssl-dev python3 ccache

FROM base AS build

COPY . /src
WORKDIR /build
ENV CCACHE_DIR=/build-cache
RUN ${WITH_CACHE} cmake \
  -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  \
  -DFSC_DEP_PREF_VENDORED=Off /src
RUN ${WITH_CACHE} ninja fsc-tool tests

FROM debian AS light
RUN apt-get -q update
RUN apt-get install -y libgomp libstdc++ openssl
COPY --from=build /build/src/c++/tools/fusionsc /usr/local/bin/fusionsc

# VMEC dependencies
FROM base AS full

RUN apt-get install -y \
  gfortran \
  libopenmpi-dev openmpi-bin \
  \
  libnetcdf-dev libnetcdff-dev \
  libhdf5-openmpi-dev hdf5-tools \
  \
  libblas-dev liblapack-dev libscalapack-openmpi-dev \
  netcdf-bin vim \
  make curl git

RUN git clone --depth 1 https://github.com/PrincetonUniversity/STELLOPT /stellopt
ENV STELLOPT_PATH=/stellopt

# Copy fusionsc machine file into container
COPY util/make_fusionsc.inc /stellopt/SHARE/make_fusionsc.inc
ENV MACHINE=fusionsc

# Build VMEC
WORKDIR /stellopt
RUN ./build_all -o release XVMEC2000

# Install VMEC into bin
RUN ln /stellopt/VMEC2000/Release/xvmec2000 /bin/xvmec2000

# Install FusionSC server
COPY --from=build /build/src/c++/tools/fusionsc /usr/local/bin/fusionsc
COPY --from=build /build/src/c++/tests /usr/local/bin/fusionsc-tests