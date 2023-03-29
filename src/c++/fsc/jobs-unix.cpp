#ifndef _WIN32

#define _XOPEN_SOURCE 500

#include "jobs.h"
#include "common.h"
#include "data.h"
#include "local.h"

#include <kj/encoding.h>

#include <kj/async-unix.h>
#include <sys/wait.h>
#include <linux/wait.h> // Linux-specific extension to wait on process file descriptors
#include <sys/syscall.h>
#include <sys/types.h>
#include <signal.h>
#include <unistd.h>

namespace fsc {
	
namespace {
	
struct TrackedProcess {
	Maybe<int> pid;
	ForkedPromise<int> retCode;
};

struct UnixProcessJob : public Job::Server {
	Own<TrackedProcess> process;
	ForkedPromise<void> completionPromise;
	
	bool isDetached = false;
	Job::State state;
	
	Temporary<Job::AttachResponse> streams;
	
	UnixProcessJob(Own<TrackedProcess> process) :
		process(mv(process)),
		observer(prepareObserver(this -> process)),
		completionPromise(nullptr)
	{
		auto& executor = getActiveThread().library() -> steward();
		
		// Synchronously extract a branch from the remote promise
		// (This is neccessary since after the constructor "process" might be destroyed)
		Own<Promise<int>> remoteResult;
		executor.runSync([&remoteResult, this]() {
			remoteResult = k::heap<Promise<int>>(this -> process -> retCode.addBranch());
		});
		
		// Now asynchronously unwrap it
		completionPromise = executor.runAsync([result = mv(remoteResult)]() {
			// The destructor for "runAsync" lambdas runs on the calling thread
			// We need to destroy the promise in the thread that created it
			// Therefore, here we need to set result to nullptr in the function
			Promise<int> innerResult = mv(result);
			result = Own<Promise<int>>();
			
			return innerResult;
		})
		.then([this](int returnCode) {
			static_assert(false, "Needs post-processing with WIFSTATUS etc. etc.");
			
			state = Job::FAILED;
			KJ_REQUIRE(returnCode == 0, "Process failed");
			state = Job::COMPLETED;
		})
		.eagerlyEvaluate(nullptr)
		.fork();
	}
	
	~UnixProcessJob() {
		Promise<void> cleanup = getActiveThread().library() -> steward().executeAsync([process = mv(process)]() {			
			// Kill child
			KJ_IF_MAYBE(pPid, process -> pid) {
				kill(*pPid);
			}
			
			return process -> retCode.addBranch().ignoreResult().attach(mv(process));
		});
		
		getActiveThread().detach(mv(cleanup));
	}
	
	Promise<void> getState(GetStateContext ctx) override {
		ctx.initResults().setState(state);
		return READY_NOW;
	}
	
	Promise<void> cancel(CancelContext ctx) override {
		return 
		return READY_NOW;
	}
	
	Promise<void> detach(DetachContext ctx) override {
		KJ_FAIL_REQUIRE("Local process jobs can not be detached");
	}
	
	Promise<void> whenRunning(WhenRunningContext ctx) override {
		return READY_NOW;
	}
	
	Promise<void> whenCompleted(WhenCompletedContext ctx) override {
		return completionPromise.addBranch();
	}
	
	Promise<void> attach(AttachContext ctx) {
		ctx.setResults(streams);
		return READY_NOW;
	}
	
	
	// Unix API specific function
	static Own<SignalObserver> prepareObserver(ProcessHandle& handle) {
		return kj::heap<SignalObserver>(
			getActiveThread().ioContext().unixEventPort,
			handle,
			SignalObserver::OBSERVE_READ
		);
	}
	
	#endif
};

struct ProcessJobScheduler : public JobScheduler::Server {
	Promise<void> run(RunContext ctx) {
		// All signal-related things should run on the steward thread
		auto& executor = getActiveThread().library() -> steward();
		
		// Pipes for communication
		int stdinPipe[2];
		int stdoutPipe[2];
		int stderrPipe[2];
		
		pipe(stdinPipe);
		pipe(stdoutPipe);
		pipe(stderrPipe);
		
		JobRequest::Reader params = ctx.getParams();
		
		// Note: execv takes the arguments as char* instead of const char* because
		// of C limitations. They are guaranteed not to be modified.
		auto heapArgs = kj::heapArrayBuilder<char*>(params.getArguments().size() + 2);
		heapArgs.add(const_cast<char*>(params.getCommand().cStr()));
		for(auto arg : params.getArguments())
			heapArgs.add(const_cast<char*>(arg.cStr()));
		heapArgs.add(nullptr);
		
		char** args = heapArgs.begin();
		const char* path = params.getCommand().cStr();
		
		Own<TrackedProcess> proc;
		
		executor.executeSync([
			stdin = stdinPipe[0],
			stdout = stdoutPipe[1],
			stderr = stderrPipe[2],
			args, path, ctx,
			&proc
		]() {
			proc = kj::heap<TrackedProcess>();
			
			// Note: Since we use PID FDs, th cloned process does not need to send a
			// SIGCHLD signal upon termination, which frees us from having to mess with
			// signal handler potentially installed by other libraries.
			// int clonedFD;
			
			// Clone is nicer, but valgrind can't do it. So we use vfork() instead.
			pid_t pid;
			pid = vfork();
			// SYSCALL check is done below once we are sure we are not the child process
			// (if the check fails on the child for some reason
			
			if(pid == 0) {
				// Child process
				dup2(stdinPipe[0], 0);
				dup2(stdoutPipe[1], 1);
				dup2(stderrPipe[2], 2);
				
				close(stdinPipe[0]);
				close(stdoutPipe[1]);
				close(stderrPipe[2]);
				
				execv(path, args);
				exit(-1);
			}
		
			// Check for result now after vfork
			KJ_SYSCALL(pid);
			
			proc -> pid = pid;
			proc -> retCode = getActiveThread().ioContext().unixEventPort.onChildExit(proc -> pid);
		}
				
		auto job = kj::heap<ProcessJob>(mv(proc));
		
		job -> streams.setStdin(
			getActiveThread().streamConverter().toRemote(
				getActiveThread().ioContext().lowLevelProvider -> wrapOutputFd(stdinPipe[1], kj::LowLevelAsyncIoProvider::TAKE_OWNERSHIP)
			)
		);
		
		job -> streams.setStdout(
			getActiveThread().streamConverter().toRemote(
				getActiveThread().ioContext().lowLevelProvider -> wrapInputFd(stdoutPipe[0], kj::LowLevelAsyncIoProvider::TAKE_OWNERSHIP)
			)
		);
		
		job -> streams.setStderr(
			getActiveThread().streamConverter().toRemote(
				getActiveThread().ioContext().lowLevelProvider -> wrapInputFd(stderrPipe[0], kj::LowLevelAsyncIoProvider::TAKE_OWNERSHIP)
			)
		);
		
		ctx.initResults().setJob(mv(job));
		
		return READY_NOW;
		
		// Portable code
	}
};

struct SlurmJob {
	JobScheduler::Client localScheduler;
	uint64_t jobId;
};

}

// API

JobScheduler::Client newProcessScheduler() {
	return kj::heap<ProcessJobScheduler>();
}

Job::Client runJob(JobScheduler::Client sched, kj::StringPtr cmd, kj::ArrayPtr<kj::StringPtr> args, kj::StringPtr workDir) {
	auto req = sched.runRequest();
	req.setWorkDir(workDir);
	req.setCommand(cmd);
	auto argsOut = req.initArguments(args.size());
	for(auto i : kj::indices(args))
		argsOut.set(i, args[i]);
	
	return req.sendForPipeline().getJob();
}

}

#endif