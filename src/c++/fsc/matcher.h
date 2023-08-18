#pragma once

#include "common.h"

#include <fsc/matcher.capnp.h>

namespace fsc {
	Own<Matcher::Server> newMatcher();
}