#include "geometry.h"

#include <kj/map.h>

namespace fsc {
	
Promise<void> GeometryResolverBase::processGeometry(Geometry::Reader input, Geometry::Builder output, ResolveGeometryContext context) {
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
			return getActiveThread().dataService().download(input.getRef())
			.then([tmp = Geometry::Builder(tmp), context, this](LocalDataRef<Geometry> local) mutable {
				return processGeometry(local.get(), tmp, context);
			}).then([tmp = tmp.asReader(), output]() mutable {
				output.setRef(getActiveThread().dataService().publish(getActiveThread().randomID(), tmp));
			}).attach(mv(tmp), thisCap());
		}
		case Geometry::NESTED: {
			return processGeometry(input.getNested(), output, context);
		}
		case Geometry::MESH: {
			output.setMesh(input.getMesh());
			return READY_NOW;
		}
		case Geometry::MERGED: {
			output.setMerged(input.getMerged());
			return READY_NOW;
		}
		case Geometry::INDEXED: {
			output.setIndexed(input.getIndexed());
			return READY_NOW;
		}
		default: {
			output.setNested(input);
			return READY_NOW;
		}
	}
}

Promise<void> GeometryResolverBase::processTransform(Transformed<Geometry>::Reader input, Transformed<Geometry>::Builder output, ResolveGeometryContext context) {
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

Promise<void> GeometryResolverBase::resolveGeometry(ResolveGeometryContext context) {
	auto input = context.getParams().getGeometry();
	auto output = context.initResults();
	
	return processGeometry(input, output, context);
}

// Class GeometryLibImpl

struct GeometryLibImpl : public GeometryLib::Server {	
	Promise<void> merge(MergeContext context) override;
	Promise<void> index(IndexContext context) override;
	Promise<void> planarCut(PlanarCutContext context) override;
	
private:
	struct GeometryAccumulator {
		kj::Vector<Temporary<MergedGeometry::Entry>> entries;
		
		inline void finish(MergedGeometry::Builder output) {
			auto outEntries = output.initEntries(entries.size());
			for(size_t i = 0; i < entries.size(); ++i) {
				outEntries.setWithCaveats(i, entries[i]);
			}
		}			
	};
	
	Promise<void> mergeGeometries(Geometry::Reader input, kj::HashSet<kj::String>& tagTable, const capnp::List<TagValue>::Reader tagScope, Maybe<Mat4d> transform, GeometryAccumulator& output);
	Promise<void> mergeGeometries(Transformed<Geometry>::Reader input, kj::HashSet<kj::String>& tagTable, const capnp::List<TagValue>::Reader tagScope, Maybe<Mat4d> transform, GeometryAccumulator& output);
	
	Promise<void> collectTagNames(Geometry::Reader input, kj::HashSet<kj::String>& output);
	Promise<void> collectTagNames(Transformed<Geometry>::Reader input, kj::HashSet<kj::String>& output);
};

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
			return getActiveThread().dataService().download(input.getRef())
			.then([input, &output, this](LocalDataRef<Geometry> geo) {
				return collectTagNames(geo.get(), output);
			});
		case Geometry::NESTED: {
			return collectTagNames(input.getNested(), output);
		}
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

Promise<void> GeometryLibImpl::mergeGeometries(Geometry::Reader input, kj::HashSet<kj::String>& tagTable, const capnp::List<TagValue>::Reader tagScope, Maybe<Mat4d> transform, GeometryAccumulator& output) {
	// KJ_CONTEXT("Merge Operation", input);
	
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
				promises.add(mergeGeometries(child, tagTable, tagValues, transform, output));
			}
				
			return joinPromises(promises.finish());
		}
		case Geometry::TRANSFORMED:
			return mergeGeometries(input.getTransformed(), tagTable, tagValues, transform, output);
		case Geometry::REF:
			return getActiveThread().dataService().download(input.getRef())
			.then([input, &tagTable, tagValues = mv(tagValues), transform, &output, this](LocalDataRef<Geometry> geo) {
				return mergeGeometries(geo.get(), tagTable, tagValues, transform, output);
			});
		case Geometry::NESTED:
			return mergeGeometries(input.getNested(), tagTable, tagValues, transform, output);
			
		case Geometry::MESH:
			return getActiveThread().dataService().download(input.getMesh())
			.then([input, &tagTable, tagValues = mv(tagValues), transform, &output](LocalDataRef<Mesh> inputMeshRef) {
				auto inputMesh = inputMeshRef.get();
				auto vertexShape = inputMesh.getVertices().getShape();
				KJ_REQUIRE(vertexShape.size() == 2);
				KJ_REQUIRE(vertexShape[1] == 3);
				
				Temporary<MergedGeometry::Entry> newEntry;
				
				newEntry.setTags(tagValues);
				
				// Copy mesh and make in-place adjustments
				newEntry.setMesh(inputMesh);
				
				KJ_IF_MAYBE(pTransform, transform) {
					auto mesh = newEntry.getMesh();
					auto vertexData = inputMesh.getVertices().getData();
					KJ_REQUIRE(vertexData.size() == vertexShape[0] * vertexShape[1]);
					
					auto outVertices = mesh.getVertices();
					// auto outVertexData = outVertices.initData(vertexData.size());
					auto outVertexData = outVertices.getData();
					
					// outVertices.setShape(inputMesh.getVertices().getShape());
				
					for(size_t i_vert = 0; i_vert < vertexShape[0]; ++i_vert) {
						Vec4d vertex { 
							vertexData[i_vert * 3 + 0],
							vertexData[i_vert * 3 + 1],
							vertexData[i_vert * 3 + 2],
							1
						};
						
						Vec4d newVertex = (*pTransform) * vertex;
						
						for(size_t i = 0; i < 3; ++i)
							outVertexData.set(i_vert * 3 + i, newVertex[i]);
					}
				} else {
					// No adjustments needed if transform not specified
				}
				
				output.entries.add(mv(newEntry));
			});
		default:
			KJ_FAIL_REQUIRE("Unknown geometry node type encountered during merge operation. Likely an unresolved node", input.which());
	}
}

Promise<void> GeometryLibImpl::mergeGeometries(Transformed<Geometry>::Reader input, kj::HashSet<kj::String>& tagTable, const capnp::List<TagValue>::Reader tagScope, Maybe<Mat4d> transform, GeometryAccumulator& output) {
	switch(input.which()) {
		case Transformed<Geometry>::LEAF:
			return mergeGeometries(input.getLeaf(), tagTable, tagScope, transform, output);
		case Transformed<Geometry>::SHIFTED: {
			auto shift = input.getShifted().getShift();
			KJ_REQUIRE(shift.size() == 3);
			
			KJ_IF_MAYBE(pTransform, transform) {
				for(int i = 0; i < 3; ++i) {
					(*pTransform)(i, 3) += shift[i];
				}
			} else {
				transform = Mat4d {
					{1, 0, 0, shift[0]},
					{0, 1, 0, shift[1]},
					{0, 0, 1, shift[2]},
					{0, 0, 0, 1}
				};
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
			
			KJ_IF_MAYBE(pTransform, transform) {
				return mergeGeometries(turned.getNode(), tagTable, tagScope, (Mat4d)((*pTransform) * rotation), output);
			} else {
				return mergeGeometries(turned.getNode(), tagTable, tagScope, rotation, output);
			}
		}
		default:
			KJ_FAIL_REQUIRE("Unknown transform type", input.which());
	}
	
}

Promise<void> GeometryLibImpl::index(IndexContext context) {
	// First we need to download the geometry we want to index
	return getActiveThread().dataService().download(context.getParams().getGeoRef())
	.then([this, context](LocalDataRef<MergedGeometry> inputRef) mutable {
		// Create output temporary and read information about input
		auto output = context.getResults().initIndexed();
		MergedGeometry::Reader input = inputRef.get();
			
		auto grid = context.getParams().getGrid();
		// auto gridSize = grid.getSize();
		Vec3u gridSize { grid.getNX(), grid.getNY(), grid.getNZ() };
		
		size_t totalSize = gridSize[0] * gridSize[1] * gridSize[2];
		
		// Allocate temporary un-aligned storage for grid refs
		kj::Vector<kj::Vector<Temporary<IndexedGeometry::ElementRef>>> tmpRefs(totalSize);
		tmpRefs.resize(totalSize);
		
		// Iterate through all components of geometry
		for(size_t iEntry = 0; iEntry < input.getEntries().size(); ++iEntry) {
			// Retrieve mesh
			auto entry = input.getEntries()[iEntry];
			auto mesh = entry.getMesh();
			auto pointData = mesh.getVertices().getData();
			auto indices = mesh.getIndices();
			
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
					uint32_t index = indices[iPoint];
					Vec3d point;
					for(size_t i = 0; i < 3; ++i)
						point[i] = pointData[3 * index + i];
					
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
		Temporary<IndexedGeometry::IndexData> indexData;
		auto shapedRefs = indexData.initGridContents();
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
		
		LocalDataRef<IndexedGeometry::IndexData> indexDataRef = getActiveThread().dataService().publish(getActiveThread().randomID(), indexData.asReader());
		
		output.setGrid(grid);
		output.setBase(context.getParams().getGeoRef());
		output.setData(mv(indexDataRef));
	});
}

Promise<void> GeometryLibImpl::merge(MergeContext context) {
	// Prepare scratch pad structures that will hold intermediate data
	auto tagNameTable = heapHeld<kj::HashSet<kj::String>>();
	auto geomAccum = heapHeld<GeometryAccumulator>();
	
	// First collect all possible tag names into a table
	auto promise = collectTagNames(context.getParams(), *tagNameTable)
	
	// Then call "mergeGeometries" on the root node with an identity transform and empty tag scope
	//   This will collect temporary built meshes in geomAccum
	.then([context, geomAccum, tagNameTable, this]() mutable {
		Temporary<capnp::List<TagValue>> tagScope(tagNameTable->size());
		
		KJ_LOG(WARNING, "Beginning merge operation");
		return mergeGeometries(context.getParams(), *tagNameTable, tagScope, nullptr, *geomAccum);
	})
	
	// Finally, copy the data from the accumulator into the output
	.then([context, geomAccum, tagNameTable, this]() mutable {
		KJ_LOG(WARNING, "Merge complete");
		// Copy data over
		Temporary<MergedGeometry> output;
		geomAccum->finish(output);
		
		// Copy tag names from the tag table
		auto outTagNames = output.initTagNames(tagNameTable->size());
		for(size_t i = 0; i < tagNameTable->size(); ++i)
			outTagNames.set(i, *(tagNameTable->begin() + i));
		
		// Publish the merged geometry into the data store
		// Derive the ID from parameters (is OK as cap table is empty)
		context.getResults().setRef(
			getActiveThread().dataService().publish(context.getParams(), output.asReader()).attach(mv(output))
		);
	});
	
	return promise.attach(tagNameTable.x(), geomAccum.x());
}

Promise<void> GeometryLibImpl::planarCut(PlanarCutContext context) {
	auto geoRef = context.getParams().getGeoRef();
	return getActiveThread().dataService().download(geoRef)
	.then([context](LocalDataRef<MergedGeometry> geoRef) mutable {		
		auto geo = geoRef.get();
		auto plane = context.getParams().getPlane();
		auto orientation = plane.getOrientation();
		
		// Build plane equation
		Vec3d normal;
		
		switch(orientation.which()) {
			case Plane::Orientation::PHI: {
				double phi = orientation.getPhi();
				normal[0] = -std::sin(phi);
				normal[1] = std::cos(phi);
				normal[2] = 0;
				break;
			}
			
			case Plane::Orientation::NORMAL: {
				auto inNormal = orientation.getNormal();
				KJ_REQUIRE(inNormal.size() == 3, "Can only process 3D planes");
				
				for(int i = 0; i < 3; ++i)
					normal[i] = inNormal[i];
				
				break;
			}
			
			default:
				KJ_FAIL_REQUIRE("Unknown plane orientation type", orientation.which());
		}
		
		Vec3d center(0, 0, 0);
		if(plane.hasCenter()) {
			auto inCenter = plane.getCenter();
			KJ_REQUIRE(inCenter.size() == 3, "Can only process 3D planes");
			
			for(int i = 0; i < 3; ++i)
				center[i] = inCenter[i];
		}
		
		double d = -center.dot(normal);
		
		kj::Vector<kj::Tuple<Vec3d, Vec3d>> lines;
		
		for(auto entry : geo.getEntries()) {			
			auto mesh = entry.getMesh();
			auto indices = mesh.getIndices();
			auto vertexData = mesh.getVertices().getData();
			
			auto meshPoint = [&](uint32_t idx) {
				idx = indices[idx];
				Vec3d result;
				for(int i = 0; i < 3; ++i) {
					result[i] = vertexData[3 * idx + i];
				}
				
				return result;
			};
			
			auto processLine = [&](Vec3d p1, Vec3d p2) -> Maybe<Vec3d> {
				double d1 = normal.dot(p1) + d;
				double d2 = normal.dot(p2) + d;
				
				if(d1 * d2 > 0)
					return nullptr; // No intersection
				
				if(d1 == d2) {
					lines.add(tuple(p1, p2)); // Line intersects coplanar with plane
					return nullptr;
				}
				
				double l = (0 - d1) / (d2 - d1);
				Vec3d result = l * p2 + (1 - l) * p1;
				return result;
			};
				
			auto processTriangle = [&](uint32_t i1, uint32_t i2, uint32_t i3) {
				auto p1 = meshPoint(i1);
				auto p2 = meshPoint(i2);
				auto p3 = meshPoint(i3);
				
				Maybe<Vec3d> linePlaneHits[3];
				linePlaneHits[0] = processLine(p1, p2);
				linePlaneHits[1] = processLine(p2, p3);
				linePlaneHits[2] = processLine(p3, p1);
				
				uint8_t nHit = 0;
				Vec3d pair[2];
				
				for(int i = 0; i < 3; ++i) {
					KJ_IF_MAYBE(pHit, linePlaneHits[i]) {
						KJ_REQUIRE(nHit <= 1, "3 mid-point hits should not be possible");
						pair[nHit++] = *pHit;
					}
				}
				
				KJ_REQUIRE(nHit == 0 || nHit == 2, "Invalid mid-point count (should be 0 or 2)", nHit);
				
				if(nHit == 2)
					lines.add(tuple(pair[0], pair[1]));
			};
			
			switch(mesh.which()) {
				case Mesh::TRI_MESH:
					for(auto iTri : kj::zeroTo(indices.size() / 3)) {
						processTriangle(3 * iTri + 0, 3 * iTri + 1, 3 * iTri + 2);
					}
					break;
				
				case Mesh::POLY_MESH: {
					auto polys = mesh.getPolyMesh();
					if(polys.size() < 2)
						continue;
					
					uint32_t nPolys = polys.size() - 1;
					for(auto iPoly : kj::zeroTo(nPolys)) {
						auto start = polys[iPoly];
						auto end = polys[iPoly + 1];
						
						for(auto i2 : kj::range(start + 1, end - 1)) {
							processTriangle(start, i2, i2 + 1);
						}
					}
					break;
				}
				
				default:
					KJ_FAIL_REQUIRE("Unknown mesh type", mesh.which());
			};
		}
		
		auto outEdges = context.initResults().initEdges();
		{
			auto shape = outEdges.initShape(3);
			shape.set(0, lines.size());
			shape.set(1, 2);
			shape.set(2, 3);
		
			auto data = outEdges.initData(2 * 3 * lines.size());
			for(auto iPair : kj::indices(lines)) {
				auto& pair = lines[iPair];
				
				for(auto iDim : kj::zeroTo(3)) {
					data.set(6 * iPair + iDim + 0, kj::get<0>(pair)[iDim]);
					data.set(6 * iPair + iDim + 3, kj::get<1>(pair)[iDim]);
				}
			}
		}
	});
}

GeometryLib::Client newGeometryLib() {
	return GeometryLib::Client(kj::heap<GeometryLibImpl>());
};

// Grid location methods

Vec3u locationInGrid(Vec3d point, CartesianGrid::Reader grid) {
	Vec3d min { grid.getXMin(), grid.getYMin(), grid.getZMin() };
	Vec3d max { grid.getXMax(), grid.getYMax(), grid.getZMax() };
	Vec3u size { grid.getNX(), grid.getNY(), grid.getNZ() };
	
	return locationInGrid(point, min, max, size);
}

}