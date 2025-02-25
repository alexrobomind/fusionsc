#include "efit.h"

#include <kj/parse/char.h>
#include <kj/parse/common.h>

namespace p = kj::parse;

namespace fsc {

namespace {

template<typename T>
double readFloat(T& input) {
	auto maybeParsed = p::sequence(
		p::discard(p::optional(p::whitespace)),
		p::optional(p::exactChar<'-'>()), // kj::parse::number can parse floating point, but not negative numbers. Lol.
		p::number
	)(input);
	FSC_REQUIRE_MAYBE(pParsed, maybeParsed, "Failed to parse float number where expected");
	
	double result = kj::get<1>(*pParsed);
	KJ_IF_MAYBE(pNeg, kj::get<0>(*pParsed)) {
		result = -result;
	}
	return result;
}


template<typename T>
kj::Array<double> readFloats(T& input, size_t count) {
	auto out = kj::heapArrayBuilder<double>(count);
	
	for(auto i : kj::range(0, count)) {
		double result = readFloat(input);
		out.add(result);
	}
	
	return out.finish();
}

OneOf<uint64_t, char> parseNumber(const kj::Array<char>& digits) {
	uint64_t result = 0;
	for(char c : digits) {
		result *= 10;
		result += c - '0';
	}
	return result;
}

}

void parseGeqdsk(AxisymmetricEquilibrium::Builder out, kj::StringPtr geqdsk) {
	// Apparently, the format for the header line seems to be:
	//  - Lots of random garbage
	//  - A meaningless number
	//  - nR
	//  - nZ
	// with numbers separated by arbitrary numbers of whitespace characters
	//
	// The following code is designed to work around this by first splitting
	// off the header line, then greedily consuming the combinations
	// (integer, then as much whitespace as possible) and
	// (one character, then as much whitespace as possible)
	// until it hits the end of line, finally checking whether the last two
	// tokens from that were integer + perhaps some whitespace
	
	// Locate end of first line
	const char* headerEnd = geqdsk.begin();
	while(headerEnd < geqdsk.end() && *headerEnd != '\n')
		++headerEnd;
	
	p::IteratorInput<char, const char*> headerInput(geqdsk.begin(), headerEnd);
	p::IteratorInput<char, const char*> input(headerEnd, geqdsk.end());
	
	auto headerParser = p::many(p::sequence(
		p::oneOf(
			p::transform(p::oneOrMore(p::digit), &parseNumber),
			p::transform(p::anyOfChars("").invert(), [](char val) { return OneOf<uint64_t, char>(val); })
		),
		p::discardWhitespace
	));
	
	auto maybeHeader = headerParser(headerInput);
	FSC_REQUIRE_MAYBE(pHeader, maybeHeader, "Failed to parse header of GEQDSK file", input.getBest() - geqdsk.begin(), input.getPosition() - geqdsk.begin());
		
	KJ_REQUIRE(pHeader -> size() >= 2, "Header must contain at least 2 elements");
	auto headerTail = pHeader -> end() - 2;
	
	KJ_REQUIRE((headerTail) -> is<uint64_t>(), "Second-to-last element of header must be integer");
	KJ_REQUIRE((headerTail + 1) -> is<uint64_t>(), "Last element of header must be integer");
	
	size_t nR = headerTail -> get<uint64_t>();
	size_t nZ = (headerTail + 1) -> get<uint64_t>();
	
	double rDim = readFloat(input);
	double zDim = readFloat(input);
	double rCentral = readFloat(input);
	double rLeft = readFloat(input);
	double zMid = readFloat(input);
	
	double rMAxis = readFloat(input);
	double zMAxis = readFloat(input);
	double fluxMag = readFloat(input);
	double fluxBoundary = readFloat(input);
	double bCentral = readFloat(input);
	
	readFloats(input, 5); // The next line only contains the current and 4 pieces of redundant information
	readFloats(input, 5); // Same as above
	
	// Normalized toroidal field
	auto toroidalField = readFloats(input, nR);
	
	// Pressure
	auto pressure = readFloats(input, nR); // pressure
	
	readFloats(input, nR); // d/dPsi 0.5 fPol**2
	readFloats(input, nR); // d/dPsi pressure);
	
	// Poloidal flux
	auto psi = readFloats(input, nR * nZ);
	
	// q profile
	auto qProfile = readFloats(input, nR); // q(Psi)
	
	// We ignore the rest of the file, as we don't care about the boundary
	
	// Store ranges
	out.setRMin(rLeft);
	out.setRMax(rLeft + rDim);
	out.setZMin(zMid - 0.5 * zDim);
	out.setZMax(zMid + 0.5 * zDim);
	out.setFluxAxis(fluxMag);
	out.setFluxBoundary(fluxBoundary);
	
	out.setNormalizedToroidalField(toroidalField);
	out.setPressureProfile(pressure);
	out.setQProfile(qProfile);
	
	auto pFlux = out.initPoloidalFlux();
	pFlux.setShape({nZ, nR});
	pFlux.setData(psi);
}

}