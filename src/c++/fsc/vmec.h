#pragma once

#include "common.h"
#include "jobs.h"

#include "kernels/device.h"

#include <fsc/vmec.capnp.h>
#include <fsc/jobs.capnp.h>

#include <kj/filesystem.h>

namespace fsc {

VmecDriver::Client createVmecDriver(Own<DeviceBase>&& dev, Own<JobLauncher>&& launcher);

kj::String generateVmecInput(VmecRequest::Reader request, kj::PathPtr mgridPath);
Promise<void> writeMGridFile(kj::PathPtr path, ComputedField::Reader cField);
void interpretOutputFile(kj::PathPtr path, VmecResult::Builder out);

}