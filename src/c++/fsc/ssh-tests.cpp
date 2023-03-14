#include <catch2/catch_test_macros.hpp>

#include <kj/array.h>

#include "ssh.h"
#include "local.h"

using namespace fsc;

TEST_CASE("in-process-server") {
	