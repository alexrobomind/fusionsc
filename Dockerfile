FROM python:3.12 AS fusionsc-spack
WORKDIR /spack

# Install spack
RUN git clone https://github.com/spack/spack.git --depth 1 .
ENV PATH /spack/bin:$PATH
RUN spack bootstrap now

FROM fusionsc-spack AS fusionsc-deps
RUN spack install hdf5+hl+cxx libssh2 ninja cmake yaml-cpp catch2 eigen botan libssh2 sqlite zlib