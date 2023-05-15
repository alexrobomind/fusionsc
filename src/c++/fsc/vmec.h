#pragma once

#include <fsc/vmec.capnp.h>
#include <fsc/jobs.capnp.h>

namespace fsc {

VmecDriver::Client createVmecDriver(JobScheduler::Client scheduler);

}