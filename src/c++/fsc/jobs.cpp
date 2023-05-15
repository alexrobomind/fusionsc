#include "jobs.h"

namespace fsc {
	
// Class JobServerBase

Promise<void> JobServerBase::eval(EvalContext ctx) {
	// Attach to the job
	auto streams = thisCap().attachRequest().sendForPipeline();
	
	// Send provided data to stdin
	auto sendStdinRequest = streams.getStdin().writeRequest();
	sendStdinRequest.setData(ctx.getParams().getStdIn());
	
	// Interleave write- and read-requests so subprocess doesn't block
	auto writeStdinPromise = sendStdinRequest.send()
	.then([stream = streams.getStdin()]() mutable  {
		return stream.eofRequest().send().ignoreResult();
	});
	
	auto readStdoutPromise = streams.getStdout().readAllStringRequest().send()
	.then([ctx](RemoteInputStream::ReadAllStringResults::Reader response) mutable {
		ctx.getResults().setStdOut(response.getText());
	});
	
	auto readStderrPromise = streams.getStderr().readAllStringRequest().send()
	.then([ctx](RemoteInputStream::ReadAllStringResults::Reader response) mutable {
		ctx.getResults().setStdErr(response.getText());
	});
	
	auto processTermination = thisCap().whenCompletedRequest().send().ignoreResult();
	
	auto builder = kj::heapArrayBuilder<Promise<void>>(4);
	builder.add(mv(readStdoutPromise));
	builder.add(mv(readStderrPromise));
	builder.add(mv(writeStdinPromise));
	builder.add(mv(processTermination));
	
	return kj::joinPromises(builder.finish());
}

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
