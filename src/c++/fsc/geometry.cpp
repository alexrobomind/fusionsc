#include "geometry.h"

#include <kj/map.h>

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
	
}

GeometryLibImpl::GeometryLibImpl(LibraryThread& lt) : lt(lt->addRef()) {}

Promise<void> GeometryLibImpl::collectTagNames(Geometry::Reader input, kj::HashSet<kj::String>& output) {
	for(auto tag : input.getTags()) {
		const kj::StringPtr tagName = tag.getName();
		
		if(!output.contains(tagName))
			output.insert(kj::heapString(tagName));
	}
	
	switch(input.which()) {
		case Geometry::COMBINED: {
			auto promises = kj::heapArrayBuilder<Promise<void>>(input.getCombined().size());
			
			for(auto child : input.getCombined()) {
				promises.add(collectTagNames(child, output));
			}
				
			return joinPromises(promises.finish());
		}
		case Geometry::TRANSFORMED:
			return collectTagNames(input.getTransformed(), output);
		case Geometry::REF:
			return lt->dataService().download(input.getRef())
			.then([input, &output, this](LocalDataRef<Geometry> geo) {
				return collectTagNames(geo.get(), output);
			});
		case Geometry::MESH:
			return READY_NOW;
		default:
			KJ_FAIL_REQUIRE("Unknown geometry node type encountered during merge operation. Likely an unresolved node", input.which());
	}
}

Promise<void> GeometryLibImpl::collectTagNames(Transformed<Geometry>::Reader input, kj::HashSet<kj::String>& output) {
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

Promise<void> GeometryLibImpl::mergeGeometries(Geometry::Reader input, kj::HashSet<kj::String>& tagTable, const capnp::List<TagValue>::Reader tagScope, Mat4d transform, GeometryAccumulator& output) {
	Temporary<capnp::List<TagValue>> tagValues(tagScope);
	
	for(auto tag : input.getTags()) {
		const kj::StringPtr tagName = tag.getName();
		
		KJ_IF_MAYBE(rowPtr, tagTable.find(tag.getName())) {
			size_t tagIdx = rowPtr - tagTable.begin();
			tagValues.setWithCaveats(tagIdx, tag.getValue());
		} else {
			KJ_FAIL_REQUIRE("Internal error, tag not found in tag table", tag.getName());
		}
	}
	
	switch(input.which()) {
		case Geometry::COMBINED: {
			auto promises = kj::heapArrayBuilder<Promise<void>>(input.getCombined().size());
			
			for(auto child : input.getCombined()) {
				promises.add(mergeGeometries(input, tagTable, tagValues, transform, output));
			}
				
			return joinPromises(promises.finish());
		}
		case Geometry::TRANSFORMED:
			return mergeGeometries(input.getTransformed(), tagTable, tagValues, transform, output);
		case Geometry::REF:
			return lt->dataService().download(input.getRef())
			.then([input, &tagTable, tagValues = mv(tagValues), transform, &output, this](LocalDataRef<Geometry> geo) {
				return mergeGeometries(geo.get(), tagTable, tagValues, transform, output);
			});
			
		case Geometry::MESH:
			return lt->dataService().download(input.getMesh())
			.then([input, &tagTable, tagValues = mv(tagValues), transform, &output](LocalDataRef<Mesh> inputMeshRef) {
				auto inputMesh = inputMeshRef.get();
				auto vertexShape = inputMesh.getVertices().getShape();
				KJ_REQUIRE(vertexShape.size() == 2);
				KJ_REQUIRE(vertexShape[1] == 3);
				
				Temporary<MergedGeometry::Entry> newEntry;
				
				newEntry.setTags(tagValues);
				
				// Copy mesh and make in-place adjustments
				newEntry.setMesh(inputMesh);
				auto mesh = newEntry.getMesh();
				auto vertexData = inputMesh.getVertices().getData();
				KJ_REQUIRE(vertexData.size() == vertexShape[0] * vertexShape[1]);
				
				auto outVertices = mesh.getVertices();
				auto outVertexData = outVertices.initData(vertexData.size());
				
				outVertices.setShape(inputMesh.getVertices().getShape());
				
				for(size_t i_vert = 0; i_vert < vertexShape[0]; ++i_vert) {
					Vec4d vertex { 
						vertexData[i_vert * 3 + 0],
						vertexData[i_vert * 3 + 1],
						vertexData[i_vert * 3 + 2],
						1
					};
					
					Vec4d newVertex = transform * vertex;
					
					for(size_t i = 0; i < 3; ++i)
						outVertexData.set(i_vert * 3 + i, newVertex[i]);
				}
				
				output.entries.add(mv(newEntry));
			});
		default:
			KJ_FAIL_REQUIRE("Unknown geometry node type encountered during merge operation. Likely an unresolved node", input.which());
	}
}

Promise<void> GeometryLibImpl::mergeGeometries(Transformed<Geometry>::Reader input, kj::HashSet<kj::String>& tagTable, const capnp::List<TagValue>::Reader tagScope, Mat4d transform, GeometryAccumulator& output) {
	switch(input.which()) {
		case Transformed<Geometry>::LEAF:
			return mergeGeometries(input.getLeaf(), tagTable, tagScope, transform, output);
		case Transformed<Geometry>::SHIFTED: {
			auto shift = input.getShifted().getShift();
			KJ_REQUIRE(shift.size() == 3);
			
			for(int i = 0; i < 3; ++i) {
				transform(i, 3) += shift[i];
			}
			
			return mergeGeometries(input.getShifted().getNode(), tagTable, tagScope, transform, output);
		}
		case Transformed<Geometry>::TURNED: {
			auto turned = input.getTurned();
			auto inAxis = turned.getAxis();
			auto inCenter = turned.getCenter();
			double angle = turned.getAngle();
			
			KJ_REQUIRE(inAxis.size() == 3);
			KJ_REQUIRE(inCenter.size() == 3);
			
			Vec3d axis   { inAxis[0], inAxis[1], inAxis[2] };
			Vec3d center { inCenter[0], inCenter[1], inCenter[2] };
			
			auto rotation = rotationAxisAngle(center, axis, angle);
			
			return mergeGeometries(turned.getNode(), tagTable, tagScope, transform * rotation, output);
		}
		default:
			KJ_FAIL_REQUIRE("Unknown transform type", input.which());
	}
	
}

Promise<void> GeometryLibImpl::index(IndexContext context) {
	// First we need to download the geometry we want to index
	return lt->dataService().download(context.getParams().getGeoRef())
	.then([this, context](LocalDataRef<MergedGeometry> inputRef) mutable {
		// Create output temporary and read information about input
		Temporary<IndexedGeometry> output;
		MergedGeometry::Reader input = inputRef.get();
			
		auto grid = context.getParams().getGrid();
		// auto gridSize = grid.getSize();
		Vec3u gridSize { grid.getNX(), grid.getNY(), grid.getNZ() };
		
		size_t totalSize = gridSize[0] * gridSize[1] * gridSize[2];
		
		// Allocate temporary un-aligned storage for grid refs
		kj::Vector<kj::Vector<Temporary<IndexedGeometry::ElementRef>>> tmpRefs(totalSize);
		
		// Iterate through all components of geometry
		for(size_t iEntry = 0; iEntry < input.getEntries().size(); ++iEntry) {
			// Retrieve mesh
			auto entry = input.getEntries()[iEntry];
			auto mesh = entry.getMesh();
			auto pointData = mesh.getVertices().getData();
			
			// Iterate over all polygons in mesh to assign them to grid boxes
			// ... Step 1: Count
			size_t nPolygons;
			switch(mesh.which()) {
				case Mesh::POLY_MESH:
					nPolygons = mesh.getPolyMesh().size() - 1;
					break;
				case Mesh::TRI_MESH:
					nPolygons = mesh.getIndices().size() / 3;
					break;
				default:
					KJ_FAIL_REQUIRE("Unknown mesh type", mesh.which());
			}
			
			// ... Step 2: Iterate
			for(size_t iPoly = 0; iPoly < nPolygons; ++iPoly) {
				// Locate polygon in index buffer
				size_t iStart; size_t iEnd;
				
				switch(mesh.which()) {
					case Mesh::POLY_MESH: {
						auto pm = mesh.getPolyMesh();
						iStart = pm[iPoly];
						iEnd   = pm[iPoly + 1];
						break;
					}
					case Mesh::TRI_MESH:
						iStart = 3 * iPoly;
						iEnd   = 3 * (iPoly + 1);
						break;
					default:
						KJ_FAIL_REQUIRE("Unknown mesh type", mesh.which());
				}
			
				// Find bounding box for polygon
				double inf = std::numeric_limits<double>::infinity();
				
				Vec3d max { -inf, -inf, -inf };
				Vec3d min { inf, inf, inf };
				
				for(size_t iPoint = iStart; iPoint < iEnd; ++iPoint) {
					Vec3d point;
					for(size_t i = 0; i < 3; ++i)
						point[i] = pointData[3 * iPoint + i];
					
					max = max.cwiseMax(point);
					min = min.cwiseMin(point);
				}
			
				// Locate bounding box points in grid
				Vec3u minCell = locationInGrid(min, grid);
				Vec3u maxCell = locationInGrid(max, grid);
				
				// For all points in-between ...
				for(size_t iX = minCell[0]; iX <= maxCell[0]; ++iX) {
				for(size_t iY = minCell[1]; iY <= maxCell[1]; ++iY) {
				for(size_t iZ = minCell[2]; iZ <= maxCell[2]; ++iZ) {
					// ... add a new ref in the corresponding cell
					size_t globalIdx = (iX * gridSize[1] + iY) * gridSize[2] + iZ;
					
					Temporary<IndexedGeometry::ElementRef>& newRef = tmpRefs[globalIdx].add();
					newRef.setMeshIndex(iEntry);
					newRef.setElementIndex(iPoly);
				}}}
			}
		}
		
		// Set up output data. This creates a packed representation of the index
		
		// Set up output (including shape)
		auto shapedRefs = output.initData();
		auto shapedRefsShape = shapedRefs.initShape(3);
		for(size_t i = 0; i < 3; ++i)
			shapedRefsShape.set(i, gridSize[i]);
		
		auto refs = shapedRefs.initData(totalSize);
		
		// Copy data into output
		for(size_t i = 0; i < totalSize; ++i) {
			auto& in = tmpRefs[i];
			auto out = refs.init(i, in.size());
			
			for(size_t iInner = 0; iInner < in.size(); ++iInner)
				out.setWithCaveats(iInner, in[iInner]);			
		}
		
		// Set up back-references
		output.setGrid(grid);
		output.setBase(context.getParams().getGeoRef());
		
		// Publish output into data store and return reference
		// Derive the ID from the parameters struct
		DataRef<IndexedGeometry>::Client outputRef = lt->dataService().publish(context.getParams(), output.asReader());
		context.getResults().setRef(outputRef);
	});
}

Promise<void> GeometryLibImpl::merge(MergeContext context) {
	// Prepare scratch pad structures that will hold intermediate data
	auto tagNameTable = kj::heap<kj::HashSet<kj::String>>();
	auto geomAccum = kj::heap<GeometryAccumulator>();
	
	// First collect all possible tag names into a table
	auto promise = collectTagNames(context.getParams(), *tagNameTable)
	
	// Then call "mergeGeometries" on the root node with an identity transform and empty tag scope
	//   This will collect temporary built meshes in geomAccum
	.then([context, &geomAccum = *geomAccum, &tagNameTable = *tagNameTable, this]() mutable {
		Temporary<capnp::List<TagValue>> tagScope(tagNameTable.size());
		
		Mat4d idTransform {
			{1, 0, 0, 0},
			{0, 1, 0, 0},
			{0, 0, 1, 0},
			{0, 0, 0, 1}
		};
		
		KJ_LOG(WARNING, "Beginning merge operation");
		return mergeGeometries(context.getParams(), tagNameTable, tagScope, idTransform, geomAccum);
	})
	
	// Finally, copy the data from the accumulator into the output
	.then([context, &geomAccum = *geomAccum, &tagNameTable = *tagNameTable, this]() mutable {
		// Copy data over
		KJ_LOG(WARNING, "Allocating output");
		Temporary<MergedGeometry> output;
		KJ_LOG(WARNING, "Copying geometry");
		geomAccum.finish(output);
		
		// Copy tag names from the tag table
		auto outTagNames = output.initTagNames(tagNameTable.size());
		for(size_t i = 0; i < tagNameTable.size(); ++i)
			outTagNames.set(i, *(tagNameTable.begin() + i));
		
		// Publish the merged geometry into the data store
		// Derive the ID from parameters
		KJ_LOG(WARNING, "Publishing output");
		context.getResults().setRef(
			lt->dataService().publish(context.getParams(), output.asReader()).attach(mv(output))
		);
		KJ_LOG(WARNING, "Finished");
	});
	
	return promise.attach(mv(tagNameTable), mv(geomAccum));
}

// Grid location methods

Vec3u locationInGrid(Vec3d point, CartesianGrid::Reader grid) {
	Vec3d min { grid.getXMin(), grid.getYMin(), grid.getZMin() };
	Vec3d max { grid.getXMax(), grid.getYMax(), grid.getZMax() };
	Vec3u size { grid.getNX(), grid.getNY(), grid.getNZ() };
	
	return locationInGrid(point, min, max, size);
}

}