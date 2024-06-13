#pragma once

#include <fsc/hfcam.capnp.h>

namespace fsc {

Own<HFCamProvider::Server> newHFCamProvider(Own<DeviceBase>);

}
