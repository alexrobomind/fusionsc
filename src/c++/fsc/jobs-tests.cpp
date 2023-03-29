#include <catch2/catch_test_macros.hpp>

#include "jobs.h"
#include "local.h"

namespace fsc {

TEST_CASE("job-echo") {
	auto l = newLibrary();
	l -> elevate();
	auto lt = l -> newThread();
	
	auto& ws = lt -> waitScope();
	
	kj::StringPtr ECHO_STRING = "Echo String p984598z!=()§$()\"kjasd'as";
	#if _WIN32
	kj::StringPtr cmd = "cmd";
	auto args = kj::heapArray<kj::StringPtr>({"/C", "echo", ECHO_STRING});
	#else
	kj::StringPtr cmd = "echo";
	auto args = kj::heapArray<kj::StringPtr>({ECHO_STRING});
	#endif
	
	// Start job
	JobScheduler::Client sched = newProcessScheduler();
	Job::Client job = runJob(sched, cmd, args);
	
	// Obtain job's stdout
	auto remoteStdout = job.attachRequest().sendForPipeline().getStdout();
	auto localStdout = lt -> streamConverter().fromRemote(remoteStdout).wait(ws);
	
	// Read all data from stdout
	auto stdoutData = localStdout -> readAllBytes().wait(ws);
	auto stdoutString = kj::heapString(stdoutData.asChars());
	
	KJ_DBG(stdoutString);
	KJ_DBG(ECHO_STRING);
}

}