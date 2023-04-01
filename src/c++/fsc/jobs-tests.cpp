#include <catch2/catch_test_macros.hpp>

#include "jobs.h"
#include "local.h"

namespace fsc {

TEST_CASE("job-echo") {
	auto l = newLibrary(true);
	auto lt = l -> newThread();
	
	auto& ws = lt -> waitScope();
	
	kj::StringPtr ECHO_STRING = "Echo String p9\\\"84598z!=()§$()\"kjasd'as";
	#if _WIN32
	kj::StringPtr cmd = "cmd";
	auto args = kj::heapArray<kj::StringPtr>({"/C", "echo", ECHO_STRING});
	#else
	kj::StringPtr cmd = "echo";
	auto args = kj::heapArray<kj::StringPtr>({"-n", ECHO_STRING});
	#endif
	
	// Start job
	JobScheduler::Client sched = newProcessScheduler();
	Job::Client job = runJob(sched, cmd, args);
	
	KJ_DBG("Job running");
	KJ_DBG("Attaching");
	
	// Obtain job's stdout
	auto remoteStdout = job.attachRequest().send().getStdout();
	
	KJ_DBG("Received remote output stream");
	remoteStdout.whenResolved().wait(ws);
	KJ_DBG("Stream resolved");
	
	auto localStdout = lt -> streamConverter().fromRemote(remoteStdout).wait(ws);
	
	KJ_DBG("Attached");
	
	// Read all data from stdout
	auto stdoutData = localStdout -> readAllBytes().wait(ws);
	auto stdoutString = kj::heapString(stdoutData.asChars());
		
	KJ_REQUIRE(stdoutString == ECHO_STRING);
}

}