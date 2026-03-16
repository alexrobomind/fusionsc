#pragma once

#include <fsc/magnetics.capnp.h>
#include "common.h"

namespace fsc {

//! Reads a geqdsk equilibrium file into an AxisymmetricEquilibrium object.
void parseGeqdsk(AxisymmetricEquilibrium::Builder out, kj::StringPtr geqdsk);

}