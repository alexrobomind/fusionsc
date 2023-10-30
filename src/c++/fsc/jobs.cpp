#include "jobs.h"
#include "local.h"

namespace fsc {

namespace {

struct BaseJobDir : public JobDir, kj::Refcounted {
	BaseJobDir(kj::Path p, Own<const kj::Directory> d, const kj::Directory& parent, kj::StringPtr name) :
		parent(parent.clone()),
		name(kj::heapString(name))
	{
		dir = mv(d);
		absPath = mv(p);
	}
	
	Own<JobDir> addRef() { return kj::addRef(*this); }
	
	~BaseJobDir() {
		ud.catchExceptionsIfUnwinding([this]() {
			parent -> remove(kj::Path(name));
		});
	}
	
	Own<const kj::Directory> parent;
	kj::StringPtr name;
	
	kj::UnwindDetector ud;
};

std::atomic<size_t> dirCounter = 0;

}

// class BaseDirProvider

BaseDirProvider::BaseDirProvider(kj::StringPtr dirName) {
	auto& fs = getActiveThread().filesystem();
	
	basePath = fs.getCurrentPath().eval(dirName);
	baseDir = fs.getRoot().openSubdir(basePath, kj::WriteMode::CREATE | kj::WriteMode::MODIFY | kj::WriteMode::CREATE_PARENT);
}

Own<JobDir> BaseDirProvider::createDir() {
	while(true) {
		auto nameCandidate = kj::str("job-", dirCounter++);
		
		if(baseDir -> exists(kj::Path(nameCandidate)))
			continue;
		
		auto dir = baseDir -> openSubdir(kj::Path(nameCandidate), kj::WriteMode::CREATE);
		return kj::refcounted<BaseJobDir>(basePath.append(nameCandidate), mv(dir), *baseDir, nameCandidate);
	}
}
	
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

Job::Client runJob(JobLauncher& sched, kj::StringPtr cmd, kj::ArrayPtr<kj::StringPtr> args, Maybe<kj::PathPtr> wd) {
	JobRequest req;
	req.command = kj::str(cmd);
	req.setArguments(args);
	req.workDir = wd.map([](kj::PathPtr x) { return x.clone(); });
	
	return sched.launch(mv(req));
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
