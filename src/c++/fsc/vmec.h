#pragma once

#include <fsc/vmec.capnp.h>
#include <fsc/jobs.capnp.h>

#include <kj/filesystem.h>

namespace fsc {

VmecDriver::Client createVmecDriver(JobScheduler::Client scheduler, kj::PathPtr workRoot);

}