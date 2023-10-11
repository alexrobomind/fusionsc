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

struct UnixProcessJob : public JobServerBase {
	ForkedPromise<void> completionPromise;
	
	bool isDetached = false;
	Job::State state = Job::State::RUNNING;
	
	Temporary<Job::AttachResponse> streams;
	
	pid_t pid;
	Own<kj::AsyncInputStream> dummyInput;
	
	char readBuffer[1];
	
	UnixProcessJob(pid_t pid, Own<kj::AsyncInputStream>&& pDummyStream) :
		pid(pid), dummyInput(mv(pDummyStream)),
		completionPromise(nullptr)
	{
		auto& executor = getActiveThread().library() -> steward();
		
		completionPromise = dummyInput -> tryRead(readBuffer, 1, 1)
		.then([this](size_t bytesRead) {
			return performWait();
		})
		.eagerlyEvaluate([this](kj::Exception&& e) {
			if(state == Job::State::RUNNING)
				state = Job::State::FAILED;
			kj::throwFatalException(mv(e));
		})
		.fork();
	}
	
	~UnixProcessJob() {
		if(pid != 0)
			kill(pid, SIGKILL);
	}
	
	Promise<void> performWait() {
		int status;
		pid_t waitResult = waitpid(pid, &status, WNOHANG);
		
		if(waitResult == 0) {
			// State change not yet completed, try again a bit later
			return getActiveThread().timer().afterDelay(10 * kj::MILLISECONDS)
			.then([this]() { return performWait(); });
		}
		
		if(waitResult == -1) {
			// waitpid failed
			// check reason
			if(errno == ECHILD) {
				// Someone probably reaped the process before we got to wait on it
				// (perhaps in a sigchld handler). Because of this, we can not determine
				// the exit code.
				KJ_FAIL_REQUIRE("Child process got reaped before its exit code could be determined. Note that this does not imply failure or success of the subprocess, just that the exit code could not be saved");
			} else if(errno == EINTR) {
				// Signal handler interrupted the wait. Try again
				return kj::evalLater([this]() { return performWait(); });
			}
			KJ_FAIL_REQUIRE("Error retrieving exit code: waitpid() failed for unknown reason", errno);
		}
		
		if(WIFEXITED(status)) {
			int exitCode = WEXITSTATUS(status);
			
			KJ_REQUIRE(exitCode == 0, "Process returned non-zero exit code");
			state = Job::State::COMPLETED;
			pid = 0;
			return READY_NOW;
		} else if(WIFSIGNALED(status)) {
			pid = 0;
			KJ_FAIL_REQUIRE("Process was killed with signal ", WTERMSIG(status));
		} else {
			// State change was not an exit. Look for more changes.
			return kj::evalLater([this]() { return performWait(); });
		}
	}
	
	Promise<void> getState(GetStateContext ctx) override {
		ctx.initResults().setState(state);
		return READY_NOW;
	}
	
	Promise<void> cancel(CancelContext ctx) override {
		if(pid != 0) {
			kill(pid, SIGKILL);
		}
		
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
		int dummyPipe[2]; // Dummy pipe to register closing of the process
		
		pipe(stdinPipe);
		pipe(stdoutPipe);
		pipe(stderrPipe);
		pipe(dummyPipe);
		
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
		const char* workDir = params.getWorkDir().cStr();
		
		auto stdin = stdinPipe[0];
		auto stdout = stdoutPipe[1];
		auto stderr = stderrPipe[1];
		auto dummy = dummyPipe[1];
				
		// Clone is nicer, but valgrind can't do it. So we use fork() instead.
		pid_t pid;
		pid = fork();
		
		if(pid == 0) {
			// Child process
			dup2(stdinPipe[0], 0);
			dup2(stdoutPipe[1], 1);
			dup2(stderrPipe[1], 2);
			
			constexpr int RARELY_USED_FD = 17;
			dup2(dummyPipe[1], RARELY_USED_FD);
			
			close(stdinPipe[0]);
			close(stdinPipe[1]);
			close(stdoutPipe[0]);
			close(stdoutPipe[1]);
			close(stderrPipe[0]);
			close(stderrPipe[1]);
			close(dummyPipe[0]);
			close(dummyPipe[1]);
			
			KJ_SYSCALL(chdir(workDir));
			
			execvp(path, args);
			exit(-1);
		}
			
		close(stdinPipe[0]);
		close(stdoutPipe[1]);
		close(stderrPipe[1]);
		close(dummyPipe[1]);
		
		auto dummyInput = getActiveThread().ioContext().lowLevelProvider -> wrapInputFd(
			dummyPipe[0], kj::LowLevelAsyncIoProvider::TAKE_OWNERSHIP
		);
		
		auto job = kj::heap<UnixProcessJob>(pid, mv(dummyInput));
		
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
	return kj::heap<UnixProcessJobScheduler>();
}

}

#endif