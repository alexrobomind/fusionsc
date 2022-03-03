#include "geometry.h"
#include "data.h"

#include "tensor.h"

namespace fsc {
	
GeometryResolverBase::GeometryResolverBase(LibraryThread& lt) : lt(lt->addRef()) {}

Promise<void> GeometryResolverBase::processGeometry(Geometry::Reader input, Geometry::Builder output, ResolveContext context) {
	output.setTags(input.getTags());
	
	switch(input.which()) {
		case Geometry::COMBINED: {
			auto combinedIn = input.getCombined();
			auto n = combinedIn.size();
			auto combinedOut = output.initCombined(n);
			
			auto subTasks = kj::heapArrayBuilder<Promise<void>>(n);
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
			return lt->dataService().download(input.getRef())
			.then([tmp = Geometry::Builder(tmp), context, this](LocalDataRef<Geometry> local) mutable {
				return processGeometry(local.get(), tmp, context);
			}).then([tmp = tmp.asReader(), output, this]() mutable {
				output.setRef(lt->dataService().publish(lt->randomID(), tmp));
			}).attach(mv(tmp), thisCap());
		}
		case Geometry::MESH: {
			output.setMesh(input.getMesh());
			return READY_NOW;
		}
		default:
			return READY_NOW;	
	}
}

Promise<void> GeometryResolverBase::processTransform(Transformed<Geometry>::Reader input, Transformed<Geometry>::Builder output, ResolveContext context) {
	switch(input.which()) {
		case Transformed<Geometry>::LEAF: {
			return processGeometry(input.getLeaf(), output.initLeaf(), context);
		}
		case Transformed<Geometry>::SHIFTED: {
			auto shiftIn = input.getShifted();
			auto shiftOut = output.initShifted();
			
			shiftOut.setShift(shiftIn.getShift());
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
			KJ_FAIL_REQUIRE("Unknown transform node encountered", input.which());
	}
}

Promise<void> GeometryResolverBase::resolve(ResolveContext context) {
	auto input = context.getParams().getGeometry();
	auto output = context.initResults();
	
	return processGeometry(input, output, context);
}

// Class GeometryLibImpl

namespace {
	Promise<void> collectTagNames(Geometry::Reader input, kj::HashSet<kj::String>& output);
	Promise<void> collectTagNames(Transformed<Geometry>::Reader input, kj::HashSet<kj::String>& output);
	
	Promise<void> collectTagNames(Geometry::Reader input, kj::HashSet<kj::String>& output) {
		for(auto tag : input.getTags()) {
			const kj::StringPtr tagName = tag.getName();
			
			if(!output.contains(tagName))
				output.insert(kj::heapString(tagName));
		}
		
		switch(input.which()) {
			case Geometry::COMBINED:
				auto promises = kj::heapArrayBuilder<Promise<void>>(input.getCombined().size());
				
				for(auto child : input.getCombined()) {
					promises.add(collectTagNames(child, output));
					
				return joinPromises(promises.finish());
			case Geometry::TRANSFORMED:
				return collectTagNames(input.getTransformed(), output);
			case Geometry::REF:
				return lt->dataService().download(input.getRef())
				.then([input, &output](LocalDataRef<Geometry> geo) {
					return collectTagNames(geo.get(), output);
				});
			case Geometry::MESH:
				return READY_NOW;
			default:
				KJ_FAIL_REQUIRE("Unknown geometry node type encountered during merge operation. Likely an unresolved node", input.which());
		}
	}
	
	Promise<void> collectTagNames(Transformed<Geometry>::Reader input, kj::HashSet<kj::String>& output) {
		switch(input.which()) {
			case Transformed<Geometry>::LEAF:
				return collectTagNames(input.getLeaf(), output);
			case Transformed<Geometry>::SHIFTED:
				return collectTagNames(input.getShifted().getNode(), output);
			case Transformed<Geometry>::TURNED:
				return collectTagNames(input.getTurned().getNode(), output);
			default:
				KJ_FAIL_REQUIRE("Unknown transform type", input.which());
		}
	}
	
	using Mat4d = Eigen::Matrix<double, 4, 4>;
	
	Promise<void> mergeGeometries(Geometry::Reader input, kj::HashSet<kj::String>& tagTable, const Temporary<capnp::List<TagValue>>& tagScope, Mat4d transform, MergedGeometry::Builder output);
	Promise<void> mergeGeometries(Transformed<Geometry>::Reader input, kj::HashSet<kj::String>& tagTable, const Temporary<capnp::List<TagValue>>& tagScope, Mat4d transform, MergedGeometry::Builder output);
	
	Promise<void> mergeGeometries(Geometry::Reader input, kj::HashSet<kj::String>& tagTable, const Temporary<capnp::List<TagValue>>& tagScope, Mat4d transform, MergedGeometry::Builder output) {
		Temporary<capnp::List<TagValue>> tagValues(tagScope);
		
		for(auto tag : input.getTags()) {
			const kj::StringPtr tagName = tag.getName();
			
			KJ_IF_MAYBE(rowPtr, tagTable.find(tag.getName())) {
				size_t tagIdx = rowPtr - tagTable.begin();
				tagValues.set(tagIdx, tag.getValue());
			} else {
				KJ_FAIL_REQUIRE("Internal error, tag not found in tag table", tag.getName());
			}
		}
		
		switch(input.which()) {
			case Geometry::COMBINED:
				auto promises = kj::heapArrayBuilder<Promise<void>>(input.getCombined().size());
				
				for(auto child : input.getCombined()) {
					promises.add(mergeGeometries(input, tagTable, tagValues, transform, output));
					
				return joinPromises(promises.finish());
			case Geometry::TRANSFORMED:
				return mergeGeometries(input.getTransformed(), tagTable, tagValues, transform, output);
			case Geometry::REF:
				return lt->dataService().download(input.getRef())
				.then([input, &tagTable, tagValues, transform, output](LocalDataRef<Geometry> geo) {
					return mergeGeometries(geo.get(), tagTable, tagValues, transform, output);
				});
				
			case Geometry::MESH:
				KJ_UNIMPLEMENTED("Missing mesh merger");
			default:
				KJ_FAIL_REQUIRE("Unknown geometry node type encountered during merge operation. Likely an unresolved node", input.which());
		}
	}
	
	Promise<void> mergeGeometries(Transformed<Geometry>::Reader input, kj::HashSet<kj::String>& tagTable, const Temporary<capnp::List<TagValue>>& tagScope, Mat4d transform, MergedGeometry::Builder output) {
		switch(input.which()) {
			case Transformed<Geometry>::LEAF:
				return mergeGeometries(input.getLeaf(), tagTable, tagScope, transform, output);
			case Transformed<Geometry>::SHIFTED:
				auto shift = input.getShifted().getShift();
				KJ_REQUIRE(shift.size() == 3);
				
				for(int i = 0; i < 3; ++i) {
					transform(i, 3) += shift[i];
				}
				
				return mergeGeometries(input.getShifted().getNode(), tagTable, tagScope, transform, output);
			case Transformed<Geometry>::TURNED:
				KJ_UNIMPLEMENTED("Missing rotation matrix application");
				
				return mergeGeometries(input.getShifted().getNode(), tagTable, tagScope, transform, output);
			default:
				KJ_FAIL_REQUIRE("Unknown transform type", input.which());
		}
		
	}
}

Promise<void> GeometryLibImpl::merge(MergeContext context) {
	auto tagNameTable = kj::heap<kj::hashSet<kj::String>>();
	
	auto promise = collectTagNames(context.getParams(), *tagNameTable);
	
	KJ_UNIMPLEMENTED("Merge routine missing");
	
	return promise.attach(mv(tagNameTable));
}

}