#ifdef _WIN32

#include "jobs.h"
#include "common.h"
#include "data.h"
#include "local.h"

#include <kj/encoding.h>

#include <kj/async-win32.h>

namespace fsc {
	
namespace {


// For Windows, we can use the kj SignalObserver API to wait on processes once it is finished.
using ProcessHandle = kj::AutoCloseHandle;

struct Win32ProcessJob : public Job::Server {
	ProcessHandle process;
	
	kj::Canceler canceler;
	ForkedPromise<int> completionPromise;
	
	bool isDetached = false;
	Job::State state;
	
	Temporary<Job::AttachResponse> streams;
	
	Win32ProcessJob(ProcessHandle&& process) :
		process(mv(process)),
		observer(prepareObserver(this -> process)),
		completionPromise(nullptr)
	{
		startTrackingTask();
	}
	
	~Win32ProcessJob() {
		if(!isDetached)
			cancelTask();
	}
	
	void startTrackingTask() {
		completionPromise = kj::evalLater([this]() { return waitForTermination(); })
		.eagerlyEvaluate([this](kj::Exception e) {
			state = Job::State::FAILED;
			kj::throwFatalException(mv(e));
		})
		.fork();
	}
	
	Promise<void> waitForTermination() {
		#if _WIN32
		
		// Win32 version
		DWORD exitCode = 0;
		KJ_WIN32(GetExitCodeProcess(process, &exitCode));
		
		if(exitCode == STILL_ACTIVE) {
			// Currently Cap'n'proto does not support waiting on signals
			return getActiveThread().timer().afterDelay(100 * kj::MILLISECONDS)
			// return observer -> onSignaled()
			.then([this]() { return waitForTermination(); });
		}
		
		KJ_REQUIRE(exitCode == 0, "Process finished with non-zero exit code");
		
		return READY_NOW;
		
		#else
		
		// UNIX version
		siginfo_t info;
		info.si_pid = 0;
		KJ_SYSCALL(waitid((idtype_t) P_PIDFD, (int) process, &info, WEXITED | WNOHANG));
		
		if(info.si_pid != 0) {
			KJ_REQUIRE(info.si_pid == (int) process);
			KJ_REQUIRE(info.si_pid == CLD_EXITED, "Process was killed");
			auto childStatusCode = info.si_status;
			KJ_REQUIRE(childStatusCode == 0, "Process finished with non-zero exit code");
			return READY_NOW;
		}
		
		return observer -> whenBecomesReadable()
		.then([this]() { waitForTermination(); });
		
		#endif
	}
	
	void cancelTask() {
		canceler.cancel("Canceled");
		
		#if _WIN32
		KJ_WIN32(TerminateProcess(process, 1));
		#else
		KJ_SYSCALL(pidfd_send_signal_wrapper((int) process, SIGTERM));
		#endif
	}
	
	Promise<void> getState(GetStateContext ctx) override {
		ctx.initResults().setState(state);
		return READY_NOW;
	}
	
	Promise<void> cancel(CancelContext ctx) override {
		cancelTask();
		return READY_NOW;
	}
	
	Promise<void> detach(DetachContext ctx) override {
		isDetached = true;
		return READY_NOW;
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
	
	#if _WIN32
	
	// Win32 API specific function
	static Own<SignalObserver> prepareObserver(ProcessHandle& handle) {
		// return getActiveThread().ioContext().win32EventPort.observeSignalState(handle);
		return kj::heap<SignalObserver>();
	}
	
	#else
	
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

struct WinCArgumentEscaper {
	kj::Vector<char> output;
	
	size_t consumedBackslashes = 0;
	
	void consume(char x) {
		if(x == '\\') {
			++consumedBackslashes;
			return;
		}
		
		if(x == '"') {
			for(auto i : kj::range(0, 2 * consumedBackslashes)) {
				output.add('\\');
			}
			consumedBackslashes = 0;
			
			output.add('\\');
			output.add('"');
		} else {
			for(auto i : kj::range(0, consumedBackslashes)) {
				output.add('\\');
			}
			consumedBackslashes = 0;
			
			output.add(x);
		}
	}
	
	kj::String finish() {
		for(auto i : kj::range(0, 2 * consumedBackslashes)) {
			output.add('\\');
		}
		return kj::heapString(output.releaseAsArray());
	}
	
	static kj::String escape(kj::StringPtr str) {
		WinCArgumentEscaper escaper;
		for(auto c : str)
			escaper.consume(c);
		return escaper.finish();
	}
	
	static kj::String escapeCommandLine(kj::ArrayPtr<kj::StringPtr> args) {
		auto tree = kj::strTree("\"", args[0], "\"");
		
		for(auto arg : args.slice(1, args.size())) {
			tree = kj::strTree(mv(tree), " ", "\"", escape(arg), "\"");
		}
		
		return kj::str(mv(tree));
	}
};

struct ProcessJobScheduler : public JobScheduler::Server {
	Promise<void> run(RunContext ctx) {
		JobRequest::Reader params = ctx.getParams();
		
		#if _WIN32
		
		// Windows launch code
		
		kj::Vector<kj::StringPtr> cmdLine;
		cmdLine.add(params.getCommand());
		for(auto arg : params.getArguments())
			cmdLine.add(arg);
		
		kj::String cmdString = WinCArgumentEscaper::escapeCommandLine(cmdLine.releaseAsArray());
		
		STARTUPINFOW stInfo;
		memset(&stInfo, 0, sizeof(STARTUPINFOW));
		stInfo.cb = sizeof(STARTUPINFOW);
		
		PROCESS_INFORMATION procInfo;
		
		KJ_WIN32(CreateProcessW(
			nullptr, // lpApplicationName
			kj::encodeWideString(cmdString, true).begin(), // lpCommandLine
			nullptr, // lpSecurityAttributes
			nullptr, // lpThreadAttributes
			false,   // bInheritHandles
			0, // dwCreationFlags
			nullptr, // lpEnvironment
			params.getWorkDir().size() != 0 ? kj::encodeWideString(params.getWorkDir(), true).begin() : nullptr, // lpCurrentDirectory
			&stInfo, // lpStartupInfo
			&procInfo
		));
		
		ProcessHandle procHandle(procInfo.hProcess);
		CloseHandle(procInfo.hThread);
		
		// Note: Stream tracking not yet available
		
		auto job = kj::heap<ProcessJob>(mv(procHandle));
		ctx.initResults().setJob(mv(job));
		
		return READY_NOW;
		
		#else
			
		// Linux launch code
		
		// Pipes for communication
		int stdinPipe[2];
		int stdoutPipe[2];
		int stderrPipe[2];
		
		pipe(stdinPipe);
		pipe(stdoutPipe);
		pipe(stderrPipe);
		
		const size_t STACK_SIZE = 1024 * 1024; // Reserve 1MB of stack for fork action
		auto stackSpace = kj::heapArray<kj::byte>(STACK_SIZE);
		
		// Note: execv takes the arguments as char* instead of const char* because
		// of C limitations. They are guaranteed not to be modified.
		auto heapArgs = kj::heapArrayBuilder<char*>(params.getArguments().size() + 2);
		heapArgs.add(const_cast<char*>(params.getCommand().cStr()));
		for(auto arg : params.getArguments())
			heapArgs.add(const_cast<char*>(arg.cStr()));
		heapArgs.add(nullptr);
		
		char** args = heapArgs.begin();
		const char* path = params.getCommand().cStr();
		
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
		
		KJ_SYSCALL(pid);
		
		// Parent process
		int pidFD;
		KJ_SYSCALL(pidFD = pidfd_open_wrapper(pid));
		ProcessHandle procHandle(pidFD);
		
		auto job = kj::heap<ProcessJob>(mv(procHandle));
		
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
		
		#endif
		
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