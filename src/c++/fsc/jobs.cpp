#include "jobs.h"
#include "common.h"
#include "local.h"

#include <kj/encoding.h>

#if _WIN32
#include <kj/async-win32.h>
#else
#include <sys/wait.h>
#endif

namespace fsc {
	
namespace {

#if _WIN32

// For Windows, we can use the kj SignalObserver API to wait on processes.
using ProcessHandle = kj::AutoCloseHandle;
using SignalObserver = kj::Win32EventPort::SignalObserver;

#else
	
// On Linux, we use pidfds to track the process. Handling SIGCHLD is a pain
// due to the need to handle it safely across threads. Additionally, we do
// not interfere with other infrastructure set up this way.
using ProcessHandle = kj::AutoCloseFd;
using SignalObserver = kj::UnixEventPort::FdObserver;

#endif

struct ProcessJob : public Job::Server {
	ProcessHandle process;
	Own<SignalObserver> observer;
	
	kj::Canceler canceler;
	ForkedPromise<void> completionPromise;
	
	bool isDetached = false;
	Job::State state;
	
	ProcessJob(ProcessHandle&& process) :
		process(mv(process)),
		observer(prepareObserver(this -> process)),
		completionPromise(nullptr)
	{
		startTrackingTask();
	}
	
	~ProcessJob() {
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
			return observer -> onSignaled()
			.then([this]() { return waitForTermination(); });
		}
		
		KJ_REQUIRE(exitCode == 0, "Process finished with non-zero exit code");
		
		#else
		
		// UNIX version
		siginfo_t info;
		info.si_pid = 0;
		KJ_SYSCALL(waitid(P_PIDFD, handle, WEXITED | WNOHANG));
		
		if(info.si_pid != 0) {
			KJ_REQUIRE(info.si_pid == (int) handle);
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
		KJ_WIN32(TerminateProcess(process, 1));
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
	
	
	#if _WIN32
	
	// Win32 API specific function
	static Own<SignalObserver> prepareObserver(ProcessHandle& handle) {
		return getActiveThread().ioContext().win32EventPort.observeSignalState(handle);
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
		
		auto job = kj::heap<ProcessJob>(mv(procHandle));
		ctx.initResults().setJob(mv(job));
		
		return READY_NOW;
		
		#endif
	}
};

}

}