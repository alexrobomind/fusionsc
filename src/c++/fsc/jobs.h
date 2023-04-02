#pragma once

#include <fsc/jobs.capnp.h>

namespace fsc {

JobScheduler::Client newProcessScheduler();
JobScheduler::Client newSlurmScheduler();

Job::Client runJob(JobScheduler::Client, kj::StringPtr cmd, kj::ArrayPtr<kj::StringPtr> args = {}, kj::StringPtr wd = nullptr);
Promise<kj::String> runToCompletion(Job::Client job);

};
