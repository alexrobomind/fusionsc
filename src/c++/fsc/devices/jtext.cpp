#include "jtext.h"

#include "../common.h"
#include "../magnetics.h"
#include "../geometry.h"

#include <kj/parse/common.h>
#include <kj/parse/char.h>

#include <fsc/devices/jtext.capnp.h>

namespace p = kj::parse;

namespace fsc { namespace devices { namespace jtext {

namespace {

kj::Array<double> parseDatFile(kj::StringPtr datFile) {
	auto transformer = [](auto maybeMinus, double val) -> double {		
		KJ_IF_MAYBE(pDontcare, maybeMinus) {
			return -val;
		} else {
			return val;
		}
	};
	
	auto maybeNegFloat = p::transform(
		p::sequence(
			p::discardWhitespace,
			p::optional(p::discard(p::exactChar<'-'>())),
			p::number
		),
	
		transformer
	);
	auto numbers = p::many(cp(maybeNegFloat));
	
	p::IteratorInput<char, const char*> input(datFile.begin(), datFile.end());
	auto maybeResult = numbers(input);
	FSC_REQUIRE_MAYBE(pResult, maybeResult, "Failed to parse dat file");
	
	return mv(*pResult);
}

void geoFromDatFile(kj::StringPtr datFile, Geometry::WrapToroidally::Builder output) {
	auto data = parseDatFile(datFile);
	
	size_t nPoints = data.size() / 2;
	auto r = output.initR(nPoints);
	auto z = output.initZ(nPoints);
	for(auto i : kj::range(0, nPoints)) {
		r.set(i, data[2 * i + 0]);
		z.set(i, data[2 * i + 1]);
	}
}

void coilFromDatFile(kj::StringPtr datFile, Filament::Builder output) {
	auto data = parseDatFile(datFile);
	
	size_t nPoints = data.size() / 3;
	Tensor<double, 2> result(3, nPoints);
	for(auto i : kj::range(0, nPoints)) {
		result(0, i) = data[3 * i + 0];
		result(1, i) = data[3 * i + 1];
		result(2, i) = data[3 * i + 2];
	}
	
	writeTensor(result, output.initInline());
}

struct JtextGeoResolver : public GeometryResolverBase {
	virtual Promise<void> processGeometry(Geometry::Reader input, Geometry::Builder output, ResolveGeometryContext context) override {
		if(!input.isJtext())
			return GeometryResolverBase::processGeometry(input, output, context);
		
		auto jtext = input.getJtext();
		switch(jtext.which()) {
			case Geometry::Jtext::TOP_LIMITER: {
				double dr = jtext.getTopLimiter() - 0.24;
				output.setTransformed(TOP_LIMITER.get());
				output.getTransformed().getShifted().setShift({0, 0, dr});
				return READY_NOW;
			}
			case Geometry::Jtext::BOTTOM_LIMITER: {
				double dr = jtext.getBottomLimiter() - 0.24;
				
				output.setTransformed(BOTTOM_LIMITER.get());
				output.getTransformed().getShifted().setShift({0, 0, -dr});
				return READY_NOW;
			}
			case Geometry::Jtext::LFS_LIMITER: {
				constexpr double ang = pi / 180 * 337.5;
				const double s = std::sin(ang);
				const double c = std::cos(ang);
				
				double dr = jtext.getLfsLimiter() - 0.24;
				
				output.setTransformed(LFS_LIMITER.get());
				output.getTransformed().getShifted().setShift({c * dr, s * dr, 0});
				
				return READY_NOW;
			}
			case Geometry::Jtext::HFS_LIMITER: {
				output.setNested(HFS_LIMITER.get());
				
				return READY_NOW;
			}
			case Geometry::Jtext::FIRST_WALL: {
				output.setNested(FIRST_WALL.get());
				
				return READY_NOW;
			}
			case Geometry::Jtext::TARGET: {
				auto wt = output.initWrapToroidally();
				geoFromDatFile(TARGET.get(), wt);
				
				auto pr = wt.initPhiRange();
				pr.getPhiStart().setDeg(125);
				pr.getPhiEnd().setDeg(145);
				
				wt.setNPhi(20);
				
				auto tags = output.initTags(1);
				tags[0].setName("component");
				tags[0].getValue().setText("target");
				
				return READY_NOW;
			}
		}
		
		KJ_FAIL_REQUIRE("Unknown J-TEXT node type");
	}
};

struct JtextFieldResolver : public FieldResolverBase {	
	Promise<void> processFilament(Filament     ::Reader input, Filament     ::Builder output, ResolveFieldContext context) override {
		if(!input.isJtext())
			return FieldResolverBase::processFilament(input, output, context);
		
		auto jtext = input.getJtext();
		
		uint8_t islandCoil = jtext.getIslandCoil();
		KJ_REQUIRE(islandCoil >= 1);
		KJ_REQUIRE(islandCoil <= 6);
		
		coilFromDatFile(ISLAND_COILS.get()[islandCoil - 1], output);
		return READY_NOW;
	}
};

}

GeometryResolver::Client newGeometryResolver() {
	return kj::heap<JtextGeoResolver>();
}

FieldResolver::Client newFieldResolver() {
	return kj::heap<JtextFieldResolver>();
}

kj::StringPtr exampleGeqdsk() {
	return EXAMPLE_GFILE.get();
}

ToroidalGrid::Reader defaultGrid() {
	return DEFAULT_GRID.get();
}

CartesianGrid::Reader defaultGeometryGrid() {
	return DEFAULT_GEO_GRID.get();
}

}}

}