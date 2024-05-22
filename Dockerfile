FROM python:3.12 AS fusionsc-spack
WORKDIR /spack

# Install spack
RUN git clone https://github.com/spack/spack.git --depth 1 .
RUN . /spack/share/spack/setup-env.sh