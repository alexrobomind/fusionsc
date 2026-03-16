#pragma once

#include "common.h"

namespace fsc {

/**
 * \brief Helper to abort computations with CTRL + C.
 *
 * Usage of this is slightly complicated. A single BreakHandler object should
 * be created on stack before the FusionSC library itself is used (implementation
 * constraint). However, its methods may only be used after library startup.
 */
struct BreakHandler {
	BreakHandler();
	~BreakHandler();
	
	//! Create a promise that resolves upon the next Ctrl + C event.
	Promise<void> onBreak();
	
	//! Wraps the promise so that it fails on the next Ctrl + C event.
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