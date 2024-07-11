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
	
	Own<kj::AsyncInputStream> myStdout;
	Own<kj::AsyncInputStream> myStderr;
	Own<MultiplexedOutputStream> myStdin;
	
	pid_t pid;
	Own<kj::AsyncInputStream> dummyInput;
	
	char readBuffer[1];
	
	UnixProcessJob(pid_t pid, Own<kj::AsyncInputStream>&& pDummyStream) :
		pid(pid), dummyInput(mv(pDummyStream)),
		completionPromise(nullptr)
	{		
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
		auto fork = [this](Own<kj::AsyncInputStream>& is) {
			kj::Tee tee = kj::newTee(mv(is));
			is = mv(tee.branches[0]);
			
			return getActiveThread().streamConverter().toRemote(mv(tee.branches[1]));
		};
		
		auto res = ctx.initResults();
		
		res.setStdout(fork(myStdout));
		res.setStderr(fork(myStderr));
		
		res.setStdin(getActiveThread().streamConverter().toRemote(myStdin -> addRef()));
		
		return READY_NOW;
	}
};

struct UnixProcessJobScheduler : public JobLauncher, kj::Refcounted, BaseDirProvider {
	using BaseDirProvider::BaseDirProvider;
	
	Own<JobDir> createDir() override {
		return BaseDirProvider::createDir();
	}
	
	Own<JobLauncher> addRef() override {
		return kj::addRef(*this);
	}
	
	Job::Client launch(JobRequest req) override {
		KJ_REQUIRE(req.numTasks == 1, "Can not launch multi-task jobs on the system launcher");
		
		// Pipes for communication
		int stdinPipe[2];
		int stdoutPipe[2];
		int stderrPipe[2];
		int dummyPipe[2]; // Dummy pipe to register closing of the process
		
		pipe(stdinPipe);
		pipe(stdoutPipe);
		pipe(stderrPipe);
		pipe(dummyPipe);
		
		// Note: execv takes the arguments as char* instead of const char* because
		// of C limitations. They are guaranteed not to be modified.
		auto heapArgs = kj::heapArrayBuilder<char*>(req.arguments.size() + 2);
		heapArgs.add(const_cast<char*>(req.command.cStr()));
		for(kj::StringPtr arg : req.arguments)
			heapArgs.add(const_cast<char*>(arg.cStr()));
		heapArgs.add(nullptr);
		
		char** args = heapArgs.begin();
		const char* path = req.command.cStr();
		
		kj::String workDir;
		KJ_IF_MAYBE(pWd, req.workDir) {
			workDir = pWd -> toString(true);
		};
		
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
			
			if(workDir != nullptr) {
				KJ_SYSCALL(chdir(workDir.cStr()), workDir);
			}
			
			KJ_SYSCALL(execvp(path, args), path, args);
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
		
		job -> myStdin = multiplex(getActiveThread().ioContext().lowLevelProvider -> wrapOutputFd(stdinPipe[1], kj::LowLevelAsyncIoProvider::TAKE_OWNERSHIP));
		job -> myStdout = buffer(getActiveThread().ioContext().lowLevelProvider -> wrapInputFd(stdoutPipe[0], kj::LowLevelAsyncIoProvider::TAKE_OWNERSHIP));
		job -> myStderr = buffer(getActiveThread().ioContext().lowLevelProvider -> wrapInputFd(stderrPipe[0], kj::LowLevelAsyncIoProvider::TAKE_OWNERSHIP));
		
		return job;
		
		// Portable code
	}
};

}

// API

Own<JobLauncher> newProcessScheduler(kj::StringPtr jobDir) {
	return kj::refcounted<UnixProcessJobScheduler>(jobDir);
}

}

#endif