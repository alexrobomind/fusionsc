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

}

void parseGeqdsk(AxisymmetricEquilibrium::Builder out, kj::StringPtr geqdsk) {
	auto headerParser = p::sequence(
		p::discard(p::sequence(
			// Part of the line we don't care about
			p::discardWhitespace,
			p::exactString("EFID"),
			p::discardWhitespace,
			p::integer, p::exactChar<'/'>(), p::integer, p::exactChar<'/'>(), p::integer,
			p::discardWhitespace,
			p::exactChar<'#'>(), p::discardWhitespace,
			p::integer, p::discardWhitespace,
			p::many(p::alphaNumeric),
			p::discardWhitespace,
			p::integer, // Dummy number
			p::discardWhitespace
		)),
		p::integer, // nR
		p::discardWhitespace,
		p::integer // nZ
	);
	
	p::IteratorInput<char, const char*> input(geqdsk.begin(), geqdsk.end());
	
	auto maybeHeader = headerParser(input);
	FSC_REQUIRE_MAYBE(pHeader, maybeHeader, "Failed to parse header of GEQDSK file", input.getBest() - geqdsk.begin(), input.getPosition() - geqdsk.begin());
	
	size_t nR = kj::get<0>(*pHeader);
	size_t nZ = kj::get<1>(*pHeader);
	
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
	
	readFloats(input, nR); // pressure
	
	readFloats(input, nR); // d/dPsi 0.5 fPol**2
	readFloats(input, nR); // d/dPsi pressure);
	
	// Poloidal flux
	auto psi = readFloats(input, nR * nZ);
	readFloats(input, nR);; // q(Psi)
	
	// We ignore the rest of the file, as we don't care about the boundary
	
	// Store ranges
	out.setRMin(rLeft);
	out.setRMax(rLeft + rDim);
	out.setZMin(zMid - 0.5 * zDim);
	out.setZMax(zMid + 0.5 * zDim);
	out.setFluxAxis(fluxMag);
	out.setFluxBoundary(fluxBoundary);
	
	out.setNormalizedToroidalField(toroidalField);
	
	auto pFlux = out.initPoloidalFlux();
	pFlux.setShape({nZ, nR});
	pFlux.setData(psi);
}

}