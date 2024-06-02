FROM alpine AS build

RUN apk add g++ cmake ninja openssl-dev python3 linux-headers

COPY . /src
WORKDIR /build
RUN cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DFSC_DEP_PREF_VENDORED=Off /src
RUN ninja fsc-tool

FROM alpine as vmec
RUN apk add g++ cmake ninja openssl-dev python3 linux-headers
RUN git clone https://github.com/PrincetonUniversity/STELLOPT /stellopt
ENV STELLOPT_PATH=/stellopt
ENV MACHINE=debian
WORKDIR /stellopt
RUN ./build_all -o release -j 8 XVMEC2000



FROM alpine as fusionsc
RUN apk add libstdc++ libgomp
COPY --from=build /build/src/c++/tools/fusionsc /usr/local/bin/fusionsc