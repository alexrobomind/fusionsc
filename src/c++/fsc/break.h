#pragma once

#include "common.h"

namespace fsc {

struct BreakHandler {
	BreakHandler();
	~BreakHandler();
	
	Promise<void> onBreak();
	
	template<typename T>
	Promise<T> wrap(Promise<T> in) {
		Promise<T> failurePath = onBreak()
		.then([]() -> Promise<T> {
			return KJ_EXCEPTION(FAILED, "Interrupted");
		});
		
		return in.exclusiveJoin(mv(failurePath));
	}
};

}