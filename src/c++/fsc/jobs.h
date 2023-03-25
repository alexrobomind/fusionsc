#pragma once

#include <fsc/jobs.capnp.h>

namespace fsc {

JobScheduler::Client newProcessScheduler();
JobScheduler::Client newSlurmScheduler();

};
