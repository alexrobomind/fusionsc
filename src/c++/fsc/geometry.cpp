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
	
	/**
	  * Calculates a rotation matrix around an axis and angle
	*/
	Mat4d rotationAxisAngle(Vec3d center, Vec3d axis, double angle) {
		double x = axis[0];
		double y = axis[1];
		double z = axis[2];
		double c = std::cos(angle);
		double s = std::sin(angle);
		
		// http://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
		Mat4d turn {
			{ c + x * x * (1 - c)    ,    x * y * (1 - c) + z * s,    x * z * (1 - c) + y * s, 0},
			{ x * y * (1 - c) + z * s,    c + y * y * (1 - c)    ,    y * z * (1 - c) + x * s, 0},
			{ x * z * (1 - c) - y * s,    y * z * (1 - c) - x * s,    c + z * z * (1 - c)    , 0},
			{            0           ,                0          ,                0          , 1}
		};
		
		Mat4d shift1 {
			{ 1, 0, 0, -center(0) },
			{ 0, 1, 0, -center(1) },
			{ 0, 0, 1, -center(2) },
			{ 0, 0, 0, 1 }
		};
		
		Mat4d shift2 {
			{ 1, 0, 0, center(0) },
			{ 0, 1, 0, center(1) },
			{ 0, 0, 1, center(2) },
			{ 0, 0, 0, 1 }
		};
		
		return shift2 * turn * shift1;
	}
	
	struct GeometryAccumulator {
		kj::Vector<Temporary<MergedGeometry::Entry>> entries;
		
		inline void finish(MergedGeometry::Builder output) {
			auto outEntries = output.initEntries(entries.size());
			for(size_t i = 0; i < entries.size(); ++i) {
				entries.set(i, outEntries[i]);
			}
		}
	};
	
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
	
	Promise<void> mergeGeometries(Geometry::Reader input, kj::HashSet<kj::String>& tagTable, const Temporary<capnp::List<TagValue>>& tagScope, Mat4d transform, MergedGeometry::Builder output, kj::Vector<Temporary<);
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
				return lt->dataServes().download(input.getMesh())
				.then([input, &tagTable, tagValues, transform, output](LocalDataRef<Mesh> inputMesh) {
					auto vertexShape = inputMesh.getVertices().getShape();
					KJ_REQUIRE(vertexShape.size() == 2);
					KJ_REQUIRE(vertexShape[1] == 3);
					
					Temporary<MergedGeometry::Entry> newEntry;
					
					newEntry.setTags(tagValues);
					
					// Copy mesh and make in-place adjustments
					auto mesh = newEntry.setMesh(inputMesh);
					auto vertexData = inputMesh.getVertices().getData();
					KJ_REQUIRE(vertexData.size() == vertexShape[0] * vertexShape[1]);
					
					for(size_t i_vert = 0; i_vert < vertexShape[0], ++i_vert) {
						Vec4f vertex { 
							vertexData[i_vert * 3 + 0],
							vertexData[i_vert * 3 + 1],
							vertexData[i_vert * 3 + 2],
							1
						};
						
						Vec4f newVertex = transform * vertex;
						
						for(size_t i = 0; i < 3; ++i)
							vertexData[i_vert * 3 + i] = newVertex[i];
					}
					
					output.entries.pushBack(mv(newEntry));
				});
				auto mesh = geometry.get
				
				newEntry.initTags(
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
				auto turned = input.getTurned();
				auto inAxis = turned.getAxis();
				auto inCenter = turned.getCenter();
				double angle = turned.getAngle();
				
				KJ_REQUIRE(inAxis.size() == 3);
				KJ_REQUIRE(inCenter.size() == 3);
				
				Vec3d axis   { inAxis[0], inAxis[1], inAxis[2] };
				Vec3d center { inCenter[0], inCenter[1], inCenter[2] };
				
				auto rotation = rotationAxisAngle(center, axis, angle);
				
				return mergeGeometries(input.getShifted().getNode(), tagTable, tagScope, transform * rotation, output);
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