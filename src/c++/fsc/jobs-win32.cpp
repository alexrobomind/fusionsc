#ifdef _WIN32

#include "jobs.h"
#include "common.h"
#include "data.h"
#include "local.h"

#include <kj/encoding.h>

#include <kj/async-win32.h>

namespace fsc {
	
namespace {

struct HandleWriter {
	kj::AutoCloseHandle pipeHandle;
	
	HandleWriter(HANDLE hdl) : pipeHandle(hdl) {}
	
	void operator()(kj::AsyncIoProvider& provider, kj::AsyncIoStream& stream, kj::WaitScope& waitScope) {
		const size_t BUFFER_SIZE = 1024;
		auto buffer = kj::heapArray<kj::byte>(BUFFER_SIZE);
		
		size_t bytes;
		DWORD bytesWritten = 0;
		
		auto readAction = [&, this]() {
			bytes = stream.read((void*) buffer.begin(), 1, buffer.size()).wait(waitScope);
		};
		auto writeAction = [&, this]() {
			KJ_WIN32(WriteFile(pipeHandle, (void*) buffer.begin(), bytes, &bytesWritten, nullptr), "Failed to write data to output stream");
		};
		
		while(true) {
			KJ_IF_MAYBE(pException, kj::runCatchingExceptions(readAction)) {
				// Failed to read, EOF
				return;
			}
			
			size_t offset = 0;
			while(offset < bytes) {
				KJ_IF_MAYBE(pException, kj::runCatchingExceptions(writeAction)) {
					return;
				}
				offset += bytesWritten;
			}
		}
	}
};

struct HandleReader {
	kj::AutoCloseHandle pipeHandle;
	
	HandleReader(HANDLE hdl) : pipeHandle(hdl) {}
	
	void operator()(kj::AsyncIoProvider& provider, kj::AsyncIoStream& stream, kj::WaitScope& waitScope) {
		const size_t BUFFER_SIZE = 1024;
		auto buffer = kj::heapArray<kj::byte>(BUFFER_SIZE);
		
		DWORD numBytesRead;
			
		auto readAction = [&, this]() {
			KJ_WIN32(ReadFile(pipeHandle, (void*) buffer.begin(), buffer.size(), &numBytesRead, nullptr), "Failed to read data from input stream");
		};
		auto writeAction = [&, this]() {
			stream.write((void*) buffer.begin(), numBytesRead).wait(waitScope);
		};
		
		while(true) {
			KJ_IF_MAYBE(pException, kj::runCatchingExceptions(readAction)) {
				return;
			}
			
			if(numBytesRead == 0)
				return;
			
			KJ_IF_MAYBE(pException, kj::runCatchingExceptions(writeAction)) {
				return;
			}
		}
	}
};

Own<kj::AsyncOutputStream> newHandleWriter(HANDLE hdl) {
	auto pipeThread = getActiveThread().ioContext().provider -> newPipeThread(HandleWriter(hdl));
	
	return pipeThread.pipe.attach(mv(pipeThread.thread));
}

Own<kj::AsyncInputStream> newHandleReader(HANDLE hdl) {
	auto pipeThread = getActiveThread().ioContext().provider -> newPipeThread(HandleReader(hdl));
	
	return pipeThread.pipe.attach(mv(pipeThread.thread));
}

struct Win32ProcessJob : public Job::Server {
	kj::AutoCloseHandle process;
	
	kj::Canceler canceler;
	ForkedPromise<void> completionPromise;
	
	bool isDetached = false;
	Job::State state = Job::State::RUNNING;
	
	Temporary<Job::AttachResponse> streams;
	
	Win32ProcessJob(HANDLE process) :
		process(process),
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
			KJ_DBG("Task failure", e);
			if(state == Job::State::RUNNING)
				state = Job::State::FAILED;
			
			state = Job::State::FAILED;
			kj::throwFatalException(mv(e));
		})
		.fork();
	}
	
	Promise<void> waitForTermination() {
		DWORD exitCode = 0;
		KJ_WIN32(GetExitCodeProcess(process, &exitCode));
		
		KJ_DBG(exitCode);
		
		if(exitCode == STILL_ACTIVE) {
			// Currently Cap'n'proto does not support waiting on signals
			return getActiveThread().timer().afterDelay(200 * kj::MILLISECONDS)
			// return observer -> onSignaled()
			.then([this]() { return waitForTermination(); });
		}
		
		KJ_REQUIRE(exitCode == 0, "Process finished with non-zero exit code");
		state = Job::State::COMPLETED;
		
		return READY_NOW;
	}
	
	void cancelTask() {
		canceler.cancel("Canceled");
		TerminateProcess(process, 1);
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

struct Win32ProcessJobScheduler : public JobScheduler::Server {
	Promise<void> run(RunContext ctx) {
		JobRequest::Reader params = ctx.getParams();
		
		kj::Vector<kj::StringPtr> cmdLine;
		cmdLine.add(params.getCommand());
		for(auto arg : params.getArguments())
			cmdLine.add(arg);
		
		kj::String cmdString = WinCArgumentEscaper::escapeCommandLine(cmdLine.releaseAsArray());
		
		SECURITY_ATTRIBUTES pipeAttributes;
		memset(&pipeAttributes, 0, sizeof(SECURITY_ATTRIBUTES));
		pipeAttributes.nLength = sizeof(SECURITY_ATTRIBUTES);
		pipeAttributes.bInheritHandle = true;
		
		HANDLE inStream[2];
		HANDLE outStream[2];
		HANDLE errStream[2];
		KJ_WIN32(CreatePipe(inStream, inStream + 1, &pipeAttributes, 0), "Failed to create stdin pipe");
		KJ_WIN32(CreatePipe(outStream, outStream + 1, &pipeAttributes, 0), "Failed to create stdout pipe");
		KJ_WIN32(CreatePipe(errStream, errStream + 1, &pipeAttributes, 0), "Failed to create stderr pipe");
		
		auto stdinStream = newHandleWriter(inStream[1]);
		auto stdoutStream = newHandleReader(outStream[0]);
		auto stderrStream = newHandleReader(errStream[0]);
		
		KJ_DEFER({
			CloseHandle(inStream[0]);
			CloseHandle(outStream[1]);
			CloseHandle(errStream[1]);
		});
		
		STARTUPINFOW stInfo;
		memset(&stInfo, 0, sizeof(STARTUPINFOW));
		stInfo.cb = sizeof(STARTUPINFOW);
		stInfo.dwFlags = STARTF_USESTDHANDLES;
		stInfo.hStdInput = inStream[0];
		stInfo.hStdOutput = outStream[1];
		stInfo.hStdError  = errStream[1];
		
		KJ_DBG(cmdString);
		
		PROCESS_INFORMATION procInfo;
		
		KJ_WIN32(CreateProcessW(
			nullptr, // lpApplicationName
			kj::encodeWideString(cmdString, true).begin(), // lpCommandLine
			nullptr, // lpSecurityAttributes
			nullptr, // lpThreadAttributes
			true,   // bInheritHandles
			0, // dwCreationFlags
			nullptr, // lpEnvironment
			params.getWorkDir().size() != 0 ? kj::encodeWideString(params.getWorkDir(), true).begin() : nullptr, // lpCurrentDirectory
			&stInfo, // lpStartupInfo
			&procInfo
		));
				
		CloseHandle(procInfo.hThread);
		
		// Note: Stream tracking not yet available
		
		auto job = kj::heap<Win32ProcessJob>(procInfo.hProcess);
		
		job -> streams.setStdin(getActiveThread().streamConverter().toRemote(mv(stdinStream)));
		job -> streams.setStdout(getActiveThread().streamConverter().toRemote(mv(stdoutStream)));
		job -> streams.setStderr(getActiveThread().streamConverter().toRemote(mv(stderrStream)));
		
		ctx.initResults().setJob(mv(job));
		
		return READY_NOW;
	}
};

}

// API

JobScheduler::Client newProcessScheduler() {
	KJ_REQUIRE(getActiveThread().library() -> isElevated(), "Process jobs can only be run through elevated FSC instances.");
	return kj::heap<Win32ProcessJobScheduler>();
}

}

#endif