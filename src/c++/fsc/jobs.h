#pragma once

#include "common.h"

#include <fsc/jobs.capnp.h>
#include <kj/map.h>

namespace fsc {

JobScheduler::Client newProcessScheduler();
JobScheduler::Client newSlurmScheduler();

Job::Client runJob(JobScheduler::Client, kj::StringPtr cmd, kj::ArrayPtr<kj::StringPtr> args = {}, kj::StringPtr wd = nullptr);
Promise<kj::String> runToCompletion(Job::Client job);


namespace internal {
	kj::TreeMap<kj::String, kj::Array<kj::String>> testSControlParser(kj::StringPtr example);
}

};
