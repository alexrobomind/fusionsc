#pragma once

#include "common.h"

#include <fsc/jobs.capnp.h>
#include <kj/map.h>
#include <kj/filesystem.h>

namespace fsc {

struct JobDir {
	Own<const kj::Directory> dir;
	kj::Path absPath = nullptr;
	
	virtual Own<JobDir> addRef() = 0;
	inline virtual ~JobDir() {};
};

struct JobDirProvider {
	virtual Own<JobDir> createDir() = 0;
	inline virtual ~JobDirProvider() noexcept(false) {};
};

struct JobRequest {
	kj::String command;
	kj::Array<kj::String> arguments;
	
	Maybe<kj::Path> workDir;
	
	size_t numTasks = 1;
	size_t cpusPerTask = 1;
	
	inline void setArguments(kj::ArrayPtr<const kj::StringPtr> newArgs) {
		auto b = kj::heapArrayBuilder<kj::String>(newArgs.size());
		for(auto s : newArgs) b.add(kj::heapString(s));
		arguments = b.finish();
	}
};

struct JobLauncher : public JobDirProvider {
	virtual Job::Client launch(JobRequest req) = 0;
	
	virtual Own<JobLauncher> addRef() = 0;
	
	inline virtual ~JobLauncher() {};
};

Own<JobLauncher> newProcessScheduler(kj::StringPtr jobDir);
Own<JobLauncher> newSlurmScheduler(kj::StringPtr jobDir);

Job::Client runJob(JobLauncher&, kj::StringPtr cmd, kj::ArrayPtr<kj::StringPtr> args = {}, kj::PathPtr wd = nullptr);
Promise<kj::String> runToCompletion(Job::Client job);

// Implementation helpers

struct BaseDirProvider : public JobDirProvider {
	BaseDirProvider(kj::StringPtr basePath);
	
	Own<JobDir> createDir() override;

private:
	kj::Path basePath = nullptr;
	Own<const kj::Directory> baseDir;
};

struct JobServerBase : public Job::Server {
	Promise<void> eval(EvalContext) override;
};

namespace internal {
	kj::TreeMap<kj::String, kj::Array<kj::String>> testSControlParser(kj::StringPtr example);
}

};
