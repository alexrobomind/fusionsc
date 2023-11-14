#include "jobs.h"
#include "common.h"
#include "local.h"

#include <kj/parse/common.h>
#include <kj/parse/char.h>
#include <kj/map.h>

namespace p = kj::parse;

namespace fsc {

namespace {

//! Parser that can read the output of an scontrol show command
struct SControlParser {
	using Result = kj::TreeMap<kj::String, kj::Array<kj::String>>;
	
	template<typename Result>
	static Array<Result> merge(Result head, kj::Array<kj::Tuple<Result>> tail) {
		auto builder = kj::heapArrayBuilder<Result>(tail.size() + 1);
		builder.add(mv(head));
		for(auto& e : tail)
			builder.add(mv(kj::get<0>(e)));
		
		return builder.finish();
	}
	
	template<typename Result, typename T, typename Sep>
	static auto sepBy(T t, Sep sep) {
		auto parser = p::sequence(
			cp(t), p::many(p::sequence(p::discard(mv(sep)), cp(t)))
		);
		
		return p::transform(mv(parser), &SControlParser::merge<Result>);
	}
		
	template<typename Input>
	Maybe<Result> operator()(Input& input) {
		auto unquotedChars = p::whitespaceChar.orAny("\"/").invert();
		
		auto valueSegment = p::oneOf(
			p::doubleQuotedString,
			p::charsToString(p::many(cp(unquotedChars)))
		);
		
		auto valueExpr = sepBy<kj::String>(cp(valueSegment), p::exactChar<'/'>());
			
		auto keyValueExpr = p::sequence(
			p::charsToString(p::oneOrMore(p::anyOfChars("=").invert())),
			p::discard(p::exactChar<'='>()),
			cp(valueExpr)
		);
		
		auto scontrolGrammar = p::sequence(
			p::discardWhitespace,
			cp(keyValueExpr),
			p::many(p::sequence(
				p::discardWhitespace,
				cp(keyValueExpr)
			)),
			p::discardWhitespace,
			p::endOfInput
		);
		
		auto maybeResult = scontrolGrammar(input);
		FSC_MAYBE_OR_RETURN(pRes, maybeResult, nullptr);
		
		Result result;
		
		// First result
		result.insert(mv(kj::get<0>(*pRes)), mv(kj::get<1>(*pRes)));
		for(auto& tuple : kj::get<2>(*pRes)) {
			result.insert(mv(kj::get<0>(tuple)), mv(kj::get<1>(tuple)));
		}
		
		return mv(result);
	}
};

kj::StringTree escapeForBash(kj::StringPtr input) {
	auto result = kj::strTree("'");
	
	while(true) {
		KJ_IF_MAYBE(pIdx, input.findFirst('\'')) {
			result = kj::strTree(mv(result), kj::heapString(input.slice(0, *pIdx)), "'\\''");
			input = input.slice(*pIdx + 1);
		} else {
			break;
		}
	}
	
	result = kj::strTree(mv(result), "'");
	return result;
}

kj::String wrapCommand(kj::StringPtr command, ArrayPtr<kj::String> arguments) {
	auto result = kj::strTree(escapeForBash(command));
	for(kj::StringPtr arg : arguments) {
		result = kj::strTree(mv(result), " ", escapeForBash(arg));
	}
	return result.flatten();
}

struct SlurmJob : public Job::Server {
	Own<JobLauncher> systemLauncher;
	unsigned int jobId;
	ForkedPromise<void> completionPromise;
	
	Job::State state = Job::State::PENDING;
	
	SlurmJob(Own<JobLauncher> systemLauncher, unsigned int jobId) :
		systemLauncher(mv(systemLauncher)), jobId(jobId),
		completionPromise(track().eagerlyEvaluate(nullptr).fork())
	{}
	
	Promise<void> track() {
		return queryState()
		.catch_([this](kj::Exception&& e) {
			// If job state query fails, assume that state didn't change
			// but log error to console
			KJ_LOG(ERROR, "Failed to query job state", e);
			return state;
		})
		.then([this](Job::State state) mutable -> Promise<void> {
			KJ_REQUIRE(state != Job::State::FAILED, "Job failed");
			
			if(state == Job::State::COMPLETED)
				return READY_NOW;
			
			//TODO: Change this to a larger time interval
			const auto QUERY_DELAY = 1 * kj::SECONDS;
			return getActiveThread().timer().afterDelay(QUERY_DELAY)
			.then([this]() mutable { return track(); });
		});
	}
	
	Promise<Job::State> queryState() {
		JobRequest scontrolRequest;
		scontrolRequest.command = kj::str("scontrol");
		scontrolRequest.setArguments({"show", "job", kj::str(jobId)});
		
		return systemLauncher -> launch(mv(scontrolRequest)).evalRequest().send()
		.then([](Job::EvalResults::Reader results) {
			p::IteratorInput<char, const char*> input(results.getStdOut().begin(), results.getStdOut().end());
			
			Maybe<SControlParser::Result> maybeParseResult = SControlParser()(input);
			FSC_REQUIRE_MAYBE(pParseResult, maybeParseResult, "Failed to parse scontrol response", results.getStdOut());
			
			auto maybeResult = pParseResult -> find("JobState");
			FSC_REQUIRE_MAYBE(pResult, maybeResult, "scontrol response does not contain JobState entry");
			
			KJ_REQUIRE(pResult -> size() == 1, "Invalid job state entry");
			kj::StringPtr state = (*pResult)[0];
			
			if(state == "RUNNING")
				return Job::State::RUNNING;
			
			if(state == "PENDING")
				return Job::State::PENDING;
			
			if(state == "FAILED")
				return Job::State::FAILED;
			
			if(state == "COMPLETED")
				return Job::State::COMPLETED;
			
			KJ_FAIL_REQUIRE("Unknown job state", state);
		});
	}
	
	Promise<void> whenRunning() {
		if(state != Job::State::PENDING)
			return READY_NOW;
		
		return getActiveThread().timer().afterDelay(1 * kj::SECONDS)
		.then([this]() mutable { return whenRunning(); });
	}
		
	
	Promise<void> attach(AttachContext ctx) override {
		return whenRunning()
		.then([this, ctx]() mutable {
			JobRequest sattachRequest;
			sattachRequest.command = kj::str("sattach");
			sattachRequest.setArguments({kj::str(jobId, ".0")});
			
			auto sattachJob = systemLauncher -> launch(mv(sattachRequest));
			return ctx.tailCall(sattachJob.attachRequest());
		});
	}
	
	Promise<void> whenCompleted(WhenCompletedContext ctx) override {
		return completionPromise.addBranch();
	}
	
	Promise<void> whenRunning(WhenRunningContext ctx) override {
		return whenRunning();
	}
	
	Promise<void> cancel(CancelContext ctx) override {
		JobRequest scancelRequest;
		scancelRequest.command = kj::str("scancel");
		scancelRequest.setArguments({kj::str(jobId)});
		
		return systemLauncher -> launch(mv(scancelRequest)).whenCompletedRequest().send().ignoreResult();
	}
};

struct SlurmJobLauncher : public JobLauncher, kj::Refcounted {
	Own<JobLauncher> systemLauncher;
	
	SlurmJobLauncher(Own<JobLauncher> sysLauncher) :
		systemLauncher(mv(sysLauncher))
	{}
	
	Job::Client launch(JobRequest req) override {
		JobRequest sbatchRequest;
		sbatchRequest.command = kj::str("sbatch");
		sbatchRequest.setArguments({
			"--parsable",
			"--ntasks", kj::str(req.numTasks),
			"--cpus-per-task", kj::str(req.cpusPerTask),
			"--wrap", wrapCommand(req.command, req.arguments),
		});
		sbatchRequest.workDir = mv(req.workDir);
		
		return runToCompletion(systemLauncher -> launch(mv(sbatchRequest)))
		.then([req = mv(req), sl = systemLauncher -> addRef()](kj::String stdoutText) mutable -> Job::Client {
			// The slurm output can hold a ';' to separate cluster name and id
			// Just retrieve the id part
			kj::String jobNoStr;
			KJ_IF_MAYBE(pPos, stdoutText.findFirst(';')) {
				jobNoStr = kj::heapString(stdoutText.slice(0, *pPos));
			} else {
				jobNoStr = kj::heapString(stdoutText);
			}
			
			unsigned int jobId = jobNoStr.parseAs<unsigned int>();
			return kj::heap<SlurmJob>(mv(sl), jobId);
		});
	}
	
	Own<JobDir> createDir() override {
		return systemLauncher -> createDir();
	}
	
	Own<JobLauncher> addRef() override {
		return kj::addRef(*this);
	}
};

}

Own<JobLauncher> newSlurmScheduler(Own<JobLauncher> backend) {
	return kj::refcounted<SlurmJobLauncher>(mv(backend));
}
	

namespace internal {
	kj::TreeMap<kj::String, kj::Array<kj::String>> testSControlParser(kj::StringPtr example) {
		// KJ_DBG(example);
		// example = "a=b c=d/e"_kj;
		p::IteratorInput<char, const char*> input(example.begin(), example.end());
		
		auto result = SControlParser()(input);
		
		FSC_REQUIRE_MAYBE(pResult, result, "Parsing failed");
		
		return mv(*pResult);
	}
}

}
