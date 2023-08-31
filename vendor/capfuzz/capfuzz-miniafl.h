#pragma once

// Fallback for running the fuzz inputs without AFL compiler
// Just reads the input from stdin

#ifndef __AFL_COMPILER

#include <cstdlib>

namespace capfuzz { namespace miniafl { namespace {

constexpr size_t FUZZBUF_SIZE = 8192;

unsigned char fuzzbuf[FUZZBUF_SIZE];
size_t fuzzInputSize = 2 * FUZZBUF_SIZE;

bool readFuzzbuf() {
	if(fuzzInputSize <= FUZZBUF_SIZE)
		return false;
	
	size_t i = 0;
	for(i = 0; i < FUZZBUF_SIZE; ++i) {
		auto item = std::getchar();
		
		if(item == EOF)
			break;
		
		fuzzbuf[i] = (unsigned char) item;
	}
	fuzzInputSize = i;
	
	return true;
}

}}}

#define __AFL_FUZZ_INIT()
#define __AFL_LOOP(x) ::capfuzz::miniafl::readFuzzbuf()
#define __AFL_FUZZ_TESTCASE_BUF ::capfuzz::miniafl::fuzzbuf
#define __AFL_FUZZ_TESTCASE_LEN ::capfuzz::miniafl::fuzzInputSize

#endif