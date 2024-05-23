FROM alpine as build

RUN apk add g++ clang botan yaml-cpp catch2 eigen libssh2 sqlite zlib

COPY . /src
WORKDIR /build
RUN cmake --DCMAKE_BUILD_TYPE=Release -DCMAKE_PREF_VENDORED=Off /src