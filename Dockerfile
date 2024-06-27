# Container for libc++
FROM alpine as macproxy

# Baseline FusionSC dependencies
RUN apk add g++ cmake ninja python3 openssl-dev libc++-dev clang linux-headers libucontext-dev

COPY . /src
WORKDIR /build
RUN CC=clang CXX=clang++ cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DFSC_DEP_PREF_VENDORED=Off -DCMAKE_CXX_FLAGS=-stdlib=libc++ /src
RUN ninja tests

# Baseline build container
FROM debian as base

# Baseline FusionSC dependencies
RUN apt-get -q update
RUN apt-get install -y g++ cmake ninja-build libssl-dev python3

FROM base as build

COPY . /src
WORKDIR /build
RUN cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DFSC_DEP_PREF_VENDORED=Off /src
RUN ninja fsc-tool

FROM debian as light
RUN apt-get -q update
RUN apt-get install -y libgomp libstdc++ openssl
COPY --from=build /build/src/c++/tools/fusionsc /usr/local/bin/fusionsc

# VMEC dependencies
FROM base as full

RUN apt-get install -y \
  gfortran \
  libopenmpi-dev openmpi-bin \
  \
  libnetcdf-dev libnetcdff-dev \
  libhdf5-openmpi-dev hdf5-tools \
  \
  libblas-dev liblapack-dev libscalapack-openmpi-dev \
  make curl git

RUN git clone --depth 1 https://github.com/PrincetonUniversity/STELLOPT /stellopt
ENV STELLOPT_PATH=/stellopt

# Copy fusionsc machine file into container
COPY util/make_fusionsc.inc /stellopt/SHARE/make_fusionsc.inc
ENV MACHINE=fusionsc

WORKDIR /stellopt
RUN ./build_all -o release XVMEC2000
COPY --from=build /build/src/c++/tools/fusionsc /usr/local/bin/fusionsc