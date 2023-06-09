#ifndef _WIN32

#define _XOPEN_SOURCE 500

#include "jobs.h"
#include "common.h"
#include "data.h"
#include "local.h"

#include <kj/encoding.h>

#include <kj/async-unix.h>
#include <sys/wait.h>
// #include <linux/wait.h> // Linux-specific extension to wait on process file descriptors
#include <sys/syscall.h>
#include <sys/types.h>
#include <signal.h>
#include <unistd.h>

namespace fsc {
	
namespace {
	
struct TrackedProcess {
	Maybe<int> pid;
	Promise<int> retCode = nullptr;
};

struct UnixProcessJob : public JobServerBase {
	Own<TrackedProcess> process;
	ForkedPromise<void> completionPromise;
	
	bool isDetached = false;
	Job::State state = Job::State::RUNNING;
	
	Temporary<Job::AttachResponse> streams;
	
	UnixProcessJob(Own<TrackedProcess> argProcess) :
		process(mv(argProcess)),
		completionPromise(nullptr)
	{
		auto& executor = getActiveThread().library() -> steward();
		
		completionPromise = executor.executeAsync([&process = *(this->process)]() mutable {
			return mv(process.retCode);
		})
		.then([this](int waitCode) {
			state = Job::State::FAILED;
			
			KJ_DBG(waitCode);
			
			if(WIFEXITED(waitCode)) {
				int returnCode = WEXITSTATUS(waitCode);
				
				KJ_REQUIRE(returnCode == 0, "Process returned non-zero exit code");
				state = Job::State::COMPLETED;
				return;
			} else if(WIFSIGNALED(waitCode)) {
				KJ_FAIL_REQUIRE("Process was killed with signal ", WTERMSIG(waitCode));
			} else {
				KJ_FAIL_REQUIRE("Internal error: Process did not exit when expected to");
			}
		})
		.eagerlyEvaluate([this](kj::Exception&& e) {
			if(state == Job::State::RUNNING)
				state = Job::State::FAILED;
			kj::throwFatalException(mv(e));
		})
		.fork();
	}
	
	~UnixProcessJob() {
		Promise<void> cleanup = getActiveThread().library() -> steward().executeAsync([&process = *process]() mutable {			
			// Kill child
			KJ_IF_MAYBE(pPid, process.pid) {
				kill(*pPid, SIGKILL);
			}
		})
		.then([this, completion = completionPromise.addBranch()]() mutable {
			return mv(completion);
		})
		.attach(mv(process));
		
		getActiveThread().detach(mv(cleanup));
	}
	
	Promise<void> getState(GetStateContext ctx) override {
		ctx.initResults().setState(state);
		return READY_NOW;
	}
	
	Promise<void> cancel(CancelContext ctx) override {
		auto& executor = getActiveThread().library() -> steward();
		
		return executor.executeAsync([this]() {
			KJ_IF_MAYBE(pPid, process -> pid) {
				kill(*pPid, SIGKILL);
			}
		});
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
	
	Promise<void> attach(AttachContext ctx) override {
		ctx.setResults(streams);
		return READY_NOW;
	}
};

struct UnixProcessJobScheduler : public JobScheduler::Server {
	Promise<void> run(RunContext ctx) override {
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
			stderr = stderrPipe[1],
			args, path, ctx,
			&proc
		]() {
			proc = kj::heap<TrackedProcess>();
			
			// Clone is nicer, but valgrind can't do it. So we use fork() instead.
			pid_t pid;
			pid = fork();
			
			if(pid == 0) {
				// Child process
				dup2(stdin, 0);
				dup2(stdout, 1);
				dup2(stderr, 2);
				
				close(stdin);
				close(stdout);
				close(stderr);
				
				execvp(path, args);
				exit(-1);
			}
			
			close(stdin);
			close(stdout);
			close(stderr);
			
			proc -> pid = pid;
			proc -> retCode = getActiveThread().ioContext().unixEventPort.onChildExit(proc -> pid);
		});
				
		auto job = kj::heap<UnixProcessJob>(mv(proc));
		
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
	KJ_REQUIRE(getActiveThread().library() -> isElevated(), "Process jobs can only be run through elevated FSC instances.");
	return kj::heap<UnixProcessJobScheduler>();
}

}

#endif