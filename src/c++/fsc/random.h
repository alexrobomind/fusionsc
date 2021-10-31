#pragma once

#include "common.h"

#include <botan/auto_rng.h>

namespace fsc {

/**
 * A cryptographically secure (pseudo-)random number generator.
 * Currently implemented as a Botan-provided auto-seeding RNG, but
 * might change without notice.
 */
class CSPRNG {
public:
	inline void randomize(kj::ArrayPtr<byte> target);
	
private:
	Botan::AutoSeeded_RNG rng;
};

// === Implementation ===

void CSPRNG::randomize(kj::ArrayPtr<byte> target) {
	rng.randomize(target.begin(), target.size());
}

} // namespace fsc