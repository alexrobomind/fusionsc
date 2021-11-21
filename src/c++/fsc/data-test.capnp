@0xfb6666e6e3d75673;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("fsc::test");

# This file contains additional data structures to help with data testing

using import "data.capnp".DataRef;

struct DataHolder {
	data @0 : Data;
}

struct DataRefHolder(T) {
	ref @0 : DataRef(T);
}

interface A {}
interface B extends(A) {}
