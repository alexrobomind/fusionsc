#pragma once

#include "common.h"
#include "jobs.h"

#include <fsc/vmec.capnp.h>
#include <fsc/jobs.capnp.h>

#include <kj/filesystem.h>

namespace fsc {

VmecDriver::Client createVmecDriver(JobLauncher& launcher, kj::PathPtr workRoot);

kj::String generateVmecInput(VmecRequest::Reader request, kj::PathPtr mgridPath);
Promise<void> writeMGridFile(kj::PathPtr path, ComputedField::Reader cField);

}