#include <catch2/catch_test_macros.hpp>

#include "jobs.h"
#include "local.h"

namespace {

constexpr kj::StringPtr SCONTROL_EXAMPLE_OUT =
"JobId=11554971 JobName=python3\n"
"   UserId=knieps1(6938) GroupId=jusers(4854) MCS_label=N/A\n"
"   Priority=300388 Nice=0 Account=jiek42 QOS=normal\n"
"   JobState=RUNNING Reason=None Dependency=(null)\n"
"   Requeue=1 Restarts=0 BatchFlag=0 Reboot=0 ExitCode=0:0\n"
"   RunTime=00:43:22 TimeLimit=02:00:00 TimeMin=N/A\n"
"   SubmitTime=2023-04-03T10:54:17 EligibleTime=2023-04-03T10:54:17\n"
"   AccrueTime=2023-04-03T10:54:17\n"
"   StartTime=2023-04-03T10:55:34 EndTime=2023-04-03T12:55:39 Deadline=N/A\n"
"   SuspendTime=None SecsPreSuspend=0 LastSchedEval=2023-04-03T10:55:34 Scheduler=Backfill\n"
"   Partition=dc-cpu AllocNode:Sid=jrlogin08:18380\n"
"   ReqNodeList=(null) ExcNodeList=(null)\n"
"   NodeList=jrc0092\n"
"   BatchHost=jrc0092\n"
"   NumNodes=1 NumCPUs=256 NumTasks=128 CPUs/Task=1 ReqB:S:C:T=0:0:*:*\n"
"   TRES=cpu=256,node=1,billing=128,gres/mem512=0\n"
"   Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=*\n"
"   MinCPUsNode=1 MinMemoryNode=0 MinTmpDiskNode=0\n"
"   Features=(null) DelayBoot=00:00:00\n"
"   OverSubscribe=NO Contiguous=0 Licenses=home@just,project@just,scratch@just Network=(null)\n"
"   Command=python3\n"
"   WorkDir=/p/project/ciek-4/knieps/git-repos/\"a b c\"/main/programs/coils\n"
"   Power=\n"
"   TresPerNode=gres:mem512\n"_kj;

}

namespace fsc {

TEST_CASE("slurm-parser") {
	auto result = internal::testSControlParser(SCONTROL_EXAMPLE_OUT);
}

TEST_CASE("job-echo") {	
	auto l = newLibrary();
	auto lt = l -> newThread();
	
	auto& ws = lt -> waitScope();
	
	kj::StringPtr ECHO_STRING = "Echo String p9\\\"84598z!=()§$()\"kjasd'as";
	kj::StringPtr cmd = "cmake";
	auto args = kj::heapArray<kj::StringPtr>({"-E", "echo_append", ECHO_STRING});
	
	// Start job
	JobScheduler::Client sched = newProcessScheduler();
	Job::Client job = runJob(sched, cmd, args);
	
	// Obtain job's stdout
	auto attach = job.attachRequest().sendForPipeline();
	// auto remoteStdout = job.attachRequest().send().getStdout();
	// auto remoteStderr = job.attachRequest().send().getStderr();
	auto remoteStdout = attach.getStdout();
	auto remoteStderr = attach.getStderr();
	
	remoteStdout.whenResolved().wait(ws);
	
	auto localStdout = lt -> streamConverter().fromRemote(remoteStdout).wait(ws);
	auto localStderr = lt -> streamConverter().fromRemote(remoteStderr).wait(ws);
	
	// Read all data from stdout
	auto stdoutData = localStdout -> readAllBytes().wait(ws);
	auto stdoutString = kj::heapString(stdoutData.asChars());
	
	auto stderrData = localStderr -> readAllBytes().wait(ws);
	auto stderrString = kj::heapString(stderrData.asChars());
	
	// Wait for process to terminate
	job.whenCompletedRequest().send().wait(ws);
	
	KJ_REQUIRE(stdoutString == ECHO_STRING);
}

}