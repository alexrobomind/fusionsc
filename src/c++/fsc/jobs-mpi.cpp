#include "jobs.h"

namespace fsc {
	
namespace {

struct MpiLauncher : public JobLauncher, kj::Refcounted {
	Own<JobLauncher> backend;
	
	MpiLauncher(Own<JobLauncher>&& newBackend) :
		backend(mv(newBackend))
	{}
	
	Own<JobLauncher> addRef() override {
		return kj::addRef(*this);
	}
	
	Job::Client launch(JobRequest req) override {
		kj::Vector<kj::String> params;
		
		params.add(kj::str("-n"));
		params.add(kj::str(req.numTasks));
		params.add(kj::str("-c"));
		params.add(kj::str(req.cpusPerTask));
		
		req.numTasks = 1;
		req.cpusPerTask = 1;
		
		for(auto& str : req.arguments)
			params.add(kj::str(str));
		
		req.arguments = params.releaseAsArray();
		req.command = kj::str("mpiexec");
		
		return backend -> launch(mv(req));
	}
	
	virtual Own<JobDir> createDir() override {
		return backend -> createDir();
	}
};

}

kj::Own<JobLauncher> newMpiScheduler(Own<JobLauncher> backend) {
	return kj::refcounted<MpiLauncher>(mv(backend));
}

}