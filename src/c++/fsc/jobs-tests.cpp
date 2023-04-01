#include <catch2/catch_test_macros.hpp>

#include "jobs.h"
#include "local.h"

namespace fsc {

TEST_CASE("job-echo") {
	auto l = newLibrary(true);
	auto lt = l -> newThread();
	
	auto& ws = lt -> waitScope();
	
	kj::StringPtr ECHO_STRING = "Echo String p9\\\"84598z!=()§$()\"kjasd'as";
	kj::StringPtr cmd = "cmake";
	auto args = kj::heapArray<kj::StringPtr>({"-E", "echo_append", ECHO_STRING});
	
	// Start job
	JobScheduler::Client sched = newProcessScheduler();
	Job::Client job = runJob(sched, cmd, args);
	
	KJ_DBG("Job running");
	KJ_DBG("Attaching");
	
	// Obtain job's stdout
	auto attach = job.attachRequest().sendForPipeline();
	// auto remoteStdout = job.attachRequest().send().getStdout();
	// auto remoteStderr = job.attachRequest().send().getStderr();
	auto remoteStdout = attach.getStdout();
	auto remoteStderr = attach.getStderr();
	
	KJ_DBG("Received remote output stream");
	remoteStdout.whenResolved().wait(ws);
	KJ_DBG("Stream resolved");
	
	auto localStdout = lt -> streamConverter().fromRemote(remoteStdout).wait(ws);
	auto localStderr = lt -> streamConverter().fromRemote(remoteStderr).wait(ws);
	
	KJ_DBG("Attached");
	
	// Read all data from stdout
	auto stdoutData = localStdout -> readAllBytes().wait(ws);
	auto stdoutString = kj::heapString(stdoutData.asChars());
	
	auto stderrData = localStderr -> readAllBytes().wait(ws);
	auto stderrString = kj::heapString(stderrData.asChars());
	KJ_DBG(stderrString);
	
	// Wait for process to terminate
	job.whenCompletedRequest().send().wait(ws);
	
	KJ_REQUIRE(stdoutString == ECHO_STRING);
}

}