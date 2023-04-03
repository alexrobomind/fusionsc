#include "jobs.h"

namespace fsc {

Job::Client runJob(JobScheduler::Client sched, kj::StringPtr cmd, kj::ArrayPtr<kj::StringPtr> args, kj::StringPtr workDir) {
	auto req = sched.runRequest();
	req.setWorkDir(workDir);
	req.setCommand(cmd);
	auto argsOut = req.initArguments(args.size());
	for(auto i : kj::indices(args))
		argsOut.set(i, args[i]);
	
	return req.send().getJob();
}

Promise<kj::String> runToCompletion(Job::Client job) {
	return job.whenResolved().then([job]() mutable {
		return job.whenCompletedRequest().send().ignoreResult()
		.then(
			// Success
			[job]() mutable {
				return job.attachRequest().sendForPipeline().getStdout().readAllStringRequest().send()
				.then([](RemoteInputStream::ReadAllStringResults::Reader results) mutable {
					return kj::heapString(results.getText());
				});
			},
			
			// Failure
			[job](kj::Exception&& e) mutable {
				return job.attachRequest().sendForPipeline().getStderr().readAllStringRequest().send()
				.then([](RemoteInputStream::ReadAllStringResults::Reader results) mutable -> kj::String {
					auto errorString = results.getText();
					
					kj::throwFatalException(KJ_EXCEPTION(FAILED, "Failed to execute", errorString));
					return kj::heapString("");
				});
			}
		);
	});
}

}
