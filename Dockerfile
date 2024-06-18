# Baseline build container
FROM debian as base

# Baseline FusionSC dependencies
RUN apt-get -q update
RUN apt-get install g++ cmake ninja libopenssl-dev python3 linux-headers

FROM build-core as build

COPY . /src
WORKDIR /build
RUN cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DFSC_DEP_PREF_VENDORED=Off /src
RUN ninja fsc-tool

FROM debian as light
RUN apt-get -q update
RUN apt-get install libgomp libstdc++6
COPY --from=build /build/src/c++/tools/fusionsc /usr/local/bin/fusionsc

# VMEC dependencies
FROM base as full

RUN apt-get install \
  gfortran \
  libopenmpi-dev openmpi-bin \
  \
  libnetcdf-dev libnetcdff-dev \
  libhdf5-openmpi-dev hdf5-tools \
  \
  libblas-dev liblapack-dev libscalapack-openmpi-dev \
  make curl git

RUN git clone https://github.com/PrincetonUniversity/STELLOPT /stellopt
ENV STELLOPT_PATH=/stellopt
ENV MACHINE=docker
WORKDIR /stellopt
RUN ./build_all -o release -j 8 XVMEC2000
COPY --from=build /build/src/c++/tools/fusionsc /usr/local/bin/fusionsc