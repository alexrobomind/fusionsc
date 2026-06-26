#Note: This needs to be run from the FusionSC build directory (above the "util" directory)

# Baseline build container
FROM debian AS base

# Baseline FusionSC dependencies
RUN apt-get -q update
RUN apt-get install -y g++ cmake ninja-build libssl-dev python3 ccache

# VMEC dependencies
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

# Switch into stellopt dir
WORKDIR /stellopt
RUN MACHINE=fusionsc ./build_all -o release XVMEC2000

# Some extra utilities
RUN apt-get install -y netcdf-bin vim