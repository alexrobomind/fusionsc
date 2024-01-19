#include <catch2/catch_test_macros.hpp>

#include "streams.h"

TEST_CASE("istream") {
	kj::StringPtr input = "Hello";
	kj::ArrayInputStream is(input.asArray().asBytes());
	
	kj::Own<std::istream> wrapped = fsc::asStdStream(is);
	
	std::string line;
	std::getline(*wrapped, line);
	
	kj::StringPtr asPtr(line);
	
	KJ_DBG(asPtr);
	KJ_REQUIRE(asPtr == input);
}

TEST_CASE("ostream") {
	kj::StringPtr input = "Hello";
	kj::VectorOutputStream os;
	
	kj::Own<std::ostream> wrapped = fsc::asStdStream(os);
	*wrapped << input.cStr() << std::flush;
		
	auto arr = os.getArray();

	KJ_DBG(arr);
	KJ_REQUIRE(input.asArray() == arr);
}