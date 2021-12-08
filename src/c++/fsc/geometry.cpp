#include "geometry.h"

namespace fsc {

Promise<void> GeometryResolverBase::processGeometry(Geometry::Reader input, Geometry::Builder output, ResolveContext context) {
	output.setTags(input.getTags());
	
	switch(input.which()) {
		case Geometry::COMBINED: {
			auto combinedIn = input.getCombined();
			auto n = combinedIn.size();
			auto combinedOut = output.initCombined(n);
			
			kj::ArrayBuilder<Promise<void>> subTasks(n);
			for(decltype(n) i = 0; i < n; ++i) {
				subTasks.add(processGeometry(combinedIn[i], combinedOut[i], context));
			}
			
			return kj::joinPromises(subTasks.finish());
		}
		case Geometry::TRANSFORMED: {
			auto transformIn = input.getTransformed();
			auto transformOut = output.initTransformed();
			
			return processTransform(transformIn, transformOut, context);
		}
		case Geometry::REF: {
			if(!context.getParams().getFollowRefs()) {
				output.setRef(input.getRef());
				return READY_NOW;
			}
						
			Temporary<Geometry> tmp;
			return lt->dataServce().download(input.getRef())
			.then([tmp, context, this](LocalDataRef<Geometry> local) mutable {
				return processGeometry(local.get(), tmp, context);
			}.then([tmp, output, this]() mutable {
				output.setRef(lt->dataService().publish(lt->randomID(), tmp));
			}.attach(mv(tmp), thisCap());
		}
		case Geometry::MESH: {
			output.setMesh(input.getMesh());
			return READY_NOW;
		}
		default:
			return READY_NOW;	
	}
}

Promise<void> GeometryResolverBase::processTransform(Transformed<Geometry>::Reader input, Transformed<Geometry>::Reader output, ResolveContext context) {
	switch(input.which()) {
		case Transformed<Geometry>::LEAF: {
			return processGeometry(input.getLeaf(), output.initLeaf(), context);
		}
		case Transformed<Geometry>::SHIFTED: {
			auto shiftIn = input.getShifted();
			auto shiftOut = output.initShifted();
			
			shiftIn.setShift(shiftOut.getShift());
			return processTransform(shiftIn.getNode(), shiftOut.initNode(), context);
		}
		case Transformed<Geometry>::TURNED: {
			auto turnedIn  = input.getTurned();
			auto turnedOut = output.initTurned();
			
			turnedOut.setAngle(turnedIn.getAngle());
			turnedOut.setAxis(turnedIn.getAxis());
			turnedOut.setCenter(turnedIn.getCenter());
			return processTransform(turnedIn.getNode(), turnedOut.initNode(), context);
		}
		default:
			return KJ_FAIL_REQUIRE("Unknown transform node encountered", input.which());
	}
}

Promise<void> GeometryResolverBase::resolve(ResolveContext context) {
	auto input = context.getParams().getGeometry();
	auto output = context.getResults().initGeometry();
	
	return processGeometry(input, output, context);
}

}