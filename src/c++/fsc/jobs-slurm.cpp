#include "jobs.h"
#include "common.h"

#include <kj/parse/common.h>
#include <kj/parse/char.h>
#include <kj/map.h>

namespace p = kj::parse;

namespace fsc {

namespace {

struct SControlParser {
	using Result = kj::TreeMap<kj::String, kj::Array<kj::String>>;
	
	template<typename Result, typename T, typename Sep>
	static auto sepBy(T t, Sep sep) {
		const auto parser = p::sequence(
			t, p::many(p::sequence(p::discard(sep), t))
		);
		
		auto merger = [](Result head, kj::Array<kj::Tuple<Result>> tail) -> kj::Array<Result> {
			auto builder = kj::heapArrayBuilder<Result>(tail.size() + 1);
			builder.add(mv(head));
			for(auto& e : tail)
				builder.add(mv(kj::get<0>(e)));
			
			return builder.finish();
		};
		
		return p::transform(parser, merger);
	}
		
	template<typename Input>
	Maybe<Result> operator()(Input& input) {
		const auto unquotedChars = p::whitespaceChar.orAny("\"/").invert();
		
		const auto valueSegment = p::oneOf(
			p::charsToString(p::oneOrMore(unquotedChars)),
			p::doubleQuotedString
		);
		
		const auto valueExpr = sepBy<kj::String>(valueSegment, p::exactChar<'/'>());
			
		const auto keyValueExpr = p::sequence(
			p::charsToString(p::oneOrMore(p::alphaNumeric)),
			p::discard(p::exactChar<'='>()),
			valueExpr
		);
		
		const auto scontrolGrammar = p::sequence(
			p::discardWhitespace,
			keyValueExpr,
			p::many(p::sequence(
				p::discardWhitespace,
				keyValueExpr
			)),
			p::endOfInput
		);
		
		FSC_MAYBE_OR_RETURN(pRes, scontrolGrammar(input), nullptr);
		
		Result result;
		
		// First result
		result.insert(mv(kj::get<0>(*pRes)), mv(kj::get<1>(*pRes)));
		for(auto& tuple : kj::get<2>(*pRes)) {
			result.insert(mv(kj::get<0>(tuple)), mv(kj::get<1>(tuple)));
		}
		
		return mv(result);
	}
};

struct SlurmJob : public Job::Server {
	JobScheduler::Client systemLauncher;
	unsigned int jobId;
	ForkedPromise<void> completionPromise;
	
	Job::State state = Job::State::PENDING;
	
	SlurmJob(JobScheduler::Client systemLauncher, unsigned int jobId) :
		systemLauncher(mv(systemLauncher)), jobId(jobId),
		completionPromise(track().fork())
	{
		
	}
	
	Promise<void> track() {
		KJ_UNIMPLEMENTED();
	}
	
	Promise<void> attach(AttachContext ctx) {
		return completionPromise.addBranch()
		.then([this, ctx]() mutable {
			auto sattachRequest = systemLauncher.runRequest();
			sattachRequest.setCommand("sattach");
			sattachRequest.setArguments({kj::str(jobId, ".0")});
			
			auto sattachJob = sattachRequest.sendForPipeline().getJob();
			return ctx.tailCall(sattachJob.attachRequest());
		});
	}
};

struct SlurmJobLauncher : public JobScheduler::Server {
	JobScheduler::Client systemLauncher;
	
	Promise<void> run(RunContext ctx) override {
		auto params = ctx.getParams();
		
		auto runRequest = systemLauncher.runRequest();
		runRequest.setCommand("slurm");
		runRequest.setArguments({
			"--parsable",
			"--ntasks", kj::str(params.getNumTasks()),
			"--cpus-per-task", kj::str(params.getNumCpusPerTask())
		});
		
		auto job = runRequest.sendForPipeline().getJob();
		return runToCompletion(job)
		.then([this, ctx, job](kj::String stdoutText) mutable {
			// The slurm output can hold a ';' to separate cluster name and id
			// Just retrieve the id part
			kj::String jobNoStr;
			KJ_IF_MAYBE(pPos, stdoutText.findFirst(';')) {
				jobNoStr = kj::heapString(stdoutText.slice(0, *pPos));
			} else {
				jobNoStr = kj::heapString(stdoutText);
			}
			
			unsigned int jobId = jobNoStr.parseAs<unsigned int>();
			ctx.initResults().setJob(kj::heap<SlurmJob>(systemLauncher, jobId));
		});
	}
};

}

namespace internal {
	kj::TreeMap<kj::String, kj::Array<kj::String>> testSControlParser(kj::StringPtr example) {
		p::IteratorInput<char, const char*> input(example.begin(), example.end());
		
		FSC_REQUIRE_MAYBE(pResult, SControlParser()(input), "Parsing failed");
		
		return mv(*pResult);
	}
}

}
