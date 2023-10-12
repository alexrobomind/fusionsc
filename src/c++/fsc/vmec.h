#pragma once

#include "common.h"

#include <fsc/vmec.capnp.h>
#include <fsc/jobs.capnp.h>

#include <kj/filesystem.h>

namespace fsc {

VmecDriver::Client createVmecDriver(JobScheduler::Client scheduler, kj::PathPtr workRoot);

kj::String generateVmecInput(VmecRequest::Reader request, kj::PathPtr mgridPath);
Promise<void> writeMGridFile(kj::Path path, ComputedField::Reader cField);

}