#pragma once

#include <fsc/magnetics.capnp.h>
#include "common.h"

namespace fsc {

void parseGeqdsk(AxisymmetricEquilibrium::Builder out, kj::StringPtr geqdsk);

}