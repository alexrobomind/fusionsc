#include "jobs.h"

namespace fsc {

namespace {

struct SlurmJob {
	JobScheduler::Client systemLauncher;
	unsigned int jobId;
	ForkedPromise<void> completionPromise;
	
	Promise<void> attach(AttachContext ctx) {
		return completionPromise.addBranch()
		.then([this, ctx]() {
			auto sattachRequest = systemLauncher.runRequest();
			sattachRequest.setCommand("sattach");
			sattachRequest.setArguments({kj::str(jobId, ".0")});
			
			auto sattachJob = sattachRequest.sendForPipeline().getJob(),
			return ctx.tailCall(sattachJob.attachRequest());
		});
	}
};

struct SlurmJobLauncher {
	JobScheduler::Client systemLauncher;
	
	Promise<void> run(RunContext ctx) override {
		auto params = ctx.getParams();
		
		auto runRequest = systemLauncher.runRequest();
		runRequest.setCommand("slurm");
		runRequest.setArguments({
			"--parsable",
			"--ntasks", kj::str(params.getNumTasks()),
			"--cpus-per-task", kj::str(params.getNumCpusPerTask())
		});
		
		auto job = runRequest.sendForPipeline().getJob();
		return runToCompletion(job)
		.then([ctx, job](kj::String stdoutText) {
			// The slurm output can hold a ';' to separate cluster name and id
			// Just retrieve the id part
			kj::String jobNoStr;
			KJ_IF_MAYBE(pPos, stdoutText.findFirst(';')) {
				jobNoStr = kj::heapString(stdoutText.slice(0, *pPos));
			} else {
				jobNoStr = kj::heapString(stdoutText);
			}
			
			unsigned int jobId = jobNoStr.parseAs<unsigned int>();
			ctx.setJob(kj::heap<SlurmJob>(systemLauncher, jobId));
		});
	}
}

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
	return job.whenCompletedRequest().send().ignoreResults()
	.then(
		// Success
		[ctx, params, job]() {
			return job.attachRequest().sendForPipeline().getStdout().readAllStringRequest().send()
			.then([](RemoteInputStream::ReadAllStringResults results) {
				return kj::heapString(results.getText());
			});
		},
		
		// Failure
		[ctx, params, job](kj::Exception&& e) {
			return job.attachRequest().sendForPipeline().getStderr().readAllStringRequest().send()
			.then([](RemoteInputStream::ReadAllStringResults results) {
				auto errorString = results.getText();
				
				kj::throwFatalException(
					KJ_EXCEPTION("Failed to execute", errorString);
				);
			});
		}
	);
}

}
