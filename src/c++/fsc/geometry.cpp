#include "geometry.h"
#include "poly.h"
#include "data.h"
#include "tensor.h"

#include <random>

#include <kj/map.h>

namespace fsc {

namespace {
	double angle(Angle::Reader in) {
		switch(in.which()) {
			case Angle::RAD: return in.getRad();
			case Angle::DEG: return pi / 180 * in.getDeg();
		}
		KJ_FAIL_REQUIRE("Unknown angle type");
	}
	
	// Limit to 128 million grid cells
	constexpr size_t MAX_GRID_SIZE = 1024 * 1024 * 128;
}

bool isBuiltin(Geometry::Reader in) {
	switch(in.which()) {
		case Geometry::COMBINED:
		case Geometry::TRANSFORMED:
		case Geometry::REF:
		case Geometry::NESTED:
		case Geometry::MESH:
		case Geometry::MERGED:
		case Geometry::INDEXED:
			return true;
		
		default:
			return false;
	}
}
	
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
				output.setRef(getActiveThread().dataService().publish(tmp));
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
		case Transformed<Geometry>::SCALED: {
			auto scaledIn  = input.getScaled();
			auto scaledOut = output.initScaled();
			
			scaledOut.setScale(scaledIn.getScale());
			return processTransform(scaledIn.getNode(), scaledOut.initNode(), context);
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
	Promise<void> reduce(ReduceContext context) override;
	Promise<void> weightedSample(WeightedSampleContext context) override;
	
private:
	struct GeometryAccumulator {
		kj::Vector<Temporary<MergedGeometry::Entry>> entries;
		
		inline void finish(MergedGeometry::Builder output) {
			auto outEntries = output.initEntries(entries.size());
			for(size_t i = 0; i < entries.size(); ++i) {
				outEntries.setWithCaveats(i, entries[i]);
				entries[i] = nullptr;
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
			{ c + x * x * (1 - c)    ,    x * y * (1 - c) - z * s,    x * z * (1 - c) + y * s, 0},
			{ x * y * (1 - c) + z * s,    c + y * y * (1 - c)    ,    y * z * (1 - c) - x * s, 0},
			{ x * z * (1 - c) - y * s,    y * z * (1 - c) + x * s,    c + z * z * (1 - c)    , 0},
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
	
	auto handleMerged = [&output](DataRef<MergedGeometry>::Client ref) {
		return getActiveThread().dataService().download(ref)
		.then([&output](LocalDataRef<MergedGeometry> localRef) {
			auto merged = localRef.get();
			
			for(auto tagName : merged.getTagNames()) {
				output.insert(kj::heapString(tagName));
			}
		});
	};
	
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
		case Geometry::MERGED:
			return handleMerged(input.getMerged());
		case Geometry::INDEXED:
			return handleMerged(input.getIndexed().getBase());
		case Geometry::WRAP_TOROIDALLY:
			return READY_NOW;
			
		default:
			KJ_FAIL_REQUIRE("Unknown geometry node type encountered during merge operation. Likely an unresolved node", input);
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
		case Transformed<Geometry>::SCALED:
			return collectTagNames(input.getScaled().getNode(), output);
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
	
	auto handleMesh = [&tagTable, transform, &output ](Mesh::Reader inputMesh, capnp::List<TagValue>::Reader tagValues) {
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
	};
	
	auto handleMerged = [&output, &tagTable, handleMesh](DataRef<MergedGeometry>::Client ref, capnp::List<TagValue>::Reader tagScope) {
		return getActiveThread().dataService().download(ref)
		.then([&output, &tagTable, tagScope, handleMesh = mv(handleMesh)](LocalDataRef<MergedGeometry> localRef) {
			auto merged = localRef.get();
			
			for(auto entry : merged.getEntries()) {
				Temporary<capnp::List<TagValue>> tagValues(tagScope);
				
				auto tagNames = merged.getTagNames();
				auto eTagVals = entry.getTags();
				
				for(auto iTag : kj::indices(merged.getTagNames())) {
					auto tagName = tagNames[iTag];
					auto tagVal  = eTagVals[iTag];
					
					if(tagVal.isNotSet())
						continue;
		
					KJ_IF_MAYBE(rowPtr, tagTable.find(tagName)) {
						size_t tagIdx = rowPtr - tagTable.begin();
						tagValues.setWithCaveats(tagIdx, tagVal);
					} else {
						KJ_FAIL_REQUIRE("Internal error, tag not found in tag table", tagName);
					}
				}
				
				handleMesh(entry.getMesh(), tagValues);
			}
		});
	};
	
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
			.then([&tagTable, tagValues = mv(tagValues), &output, handleMesh](LocalDataRef<Mesh> inputMeshRef) {
				handleMesh(inputMeshRef.get(), tagValues);
			});
		
		case Geometry::MERGED:
			return handleMerged(input.getMerged(), tagValues.asReader()).attach(mv(tagValues));
		
		case Geometry::INDEXED:
			return handleMerged(input.getIndexed().getBase(), tagValues.asReader()).attach(mv(tagValues));
		
		case Geometry::WRAP_TOROIDALLY: {
			auto wt = input.getWrapToroidally();
			
			double phiStart = 0;
			double phiEnd = 2 * pi;
			
			uint32_t nPhi = wt.getNPhi();
			KJ_REQUIRE(nPhi > 1);
			
			bool close = false;
			if(wt.isPhiRange()) {
				auto pr = wt.getPhiRange();
				phiStart = angle(pr.getPhiStart());
				phiEnd = angle(pr.getPhiEnd());
				close = pr.getClose();
			}
			
			auto r = wt.getR();
			auto z = wt.getZ();
			KJ_REQUIRE(z.size() == r.size());
			
			uint32_t nVerts = r.size();
			KJ_REQUIRE(nVerts > 1);
			
			uint32_t nLines = nVerts - 1;
			
			// Create vertices
			Tensor<double, 3> vertices(3, nVerts, nPhi + 1);
			const double dPhi = (phiEnd - phiStart) / nPhi;
			for(auto iPhi : kj::range(0, nPhi + 1)) {
				double phi = iPhi * dPhi + phiStart;
				
				for(auto iVert : kj::indices(r)) {					
					vertices(0, iVert, iPhi) = r[iVert] * cos(phi);
					vertices(1, iVert, iPhi) = r[iVert] * sin(phi);
					vertices(2, iVert, iPhi) = z[iVert];
				}
			}
			KJ_LOG(INFO, "Vertices generated");
			
			// Create triangles
			Tensor<uint32_t, 4> triangles(3, 2, nLines, nPhi);
			for(auto iPhi : kj::range(0, nPhi)) {
				for(auto iLine : kj::range(0, nLines)) {
					uint32_t v1 = iPhi * nVerts + iLine;
					uint32_t v2 = v1 + 1;
					uint32_t v3 = v2 + nVerts;
					uint32_t v4 = v3 - 1;
					
					triangles(0, 0, iLine, iPhi) = v1;
					triangles(1, 0, iLine, iPhi) = v2;
					triangles(2, 0, iLine, iPhi) = v3;
					
					triangles(0, 1, iLine, iPhi) = v3;
					triangles(1, 1, iLine, iPhi) = v4;
					triangles(2, 1, iLine, iPhi) = v1;
				}
			}
			KJ_LOG(INFO, "Triangles generated");
			
			using A = Eigen::array<int64_t, 2>;
			
			Tensor<double, 2> flatVerts = vertices.reshape(A({3, nVerts * (nPhi + 1)}));
			KJ_LOG(INFO, "Verts reshaped");
			
			{
				Temporary<Mesh> tmpMesh;
				writeTensor(flatVerts, tmpMesh.getVertices());
				
				auto indices = tmpMesh.initIndices(3 * 2 * nLines * nPhi);
				for(auto i : kj::indices(indices))
					indices.set(i, triangles.data()[i]);
				
				tmpMesh.setTriMesh();
				handleMesh(tmpMesh, tagValues);
			}
			
			// Generate end caps
			if(close) {
				Tensor<double, 2> rz(nVerts, 2);
				for(auto i : kj::range(0, nVerts)) {
					rz(i, 0) = r[i];
					rz(i, 1) = z[i];
				}
				
				Tensor<uint32_t, 2> triangulation = triangulate(rz);
				uint32_t numTriangles = triangulation.dimension(0);
				
				auto phis = kj::heapArray<double>({phiStart, phiEnd});
				for(double phi : phis) {
					Temporary<Mesh> tmpMesh;
					Tensor<double, 2> vertices(3, nVerts);
					for(auto iVert : kj::indices(r)) {					
						vertices(0, iVert) = r[iVert] * cos(phi);
						vertices(1, iVert) = r[iVert] * sin(phi);
						vertices(2, iVert) = z[iVert];
					}
					writeTensor(vertices, tmpMesh.getVertices());
				
					auto indices = tmpMesh.initIndices(3 * numTriangles);
					for(auto i : kj::range(0, numTriangles)) {						
						indices.set(3 * i + 0, triangulation(i, 0));
						indices.set(3 * i + 1, triangulation(i, 1));
						indices.set(3 * i + 2, triangulation(i, 2));
					}
				
					tmpMesh.setTriMesh();
					handleMesh(tmpMesh, tagValues);
				}
			}
			return READY_NOW;
		}
			
		default:
			KJ_FAIL_REQUIRE("Unknown geometry node type encountered during merge operation. Likely an unresolved node", input.which());
	}
}

Promise<void> GeometryLibImpl::mergeGeometries(Transformed<Geometry>::Reader input, kj::HashSet<kj::String>& tagTable, const capnp::List<TagValue>::Reader tagScope, Maybe<Mat4d> transform, GeometryAccumulator& output) {
	Mat4d transformBase;
	Transformed<Geometry>::Reader node;
	
	switch(input.which()) {
		case Transformed<Geometry>::LEAF:
			return mergeGeometries(input.getLeaf(), tagTable, tagScope, transform, output);
			
		case Transformed<Geometry>::SHIFTED: {
			node = input.getShifted().getNode();
			auto shift = input.getShifted().getShift();
			KJ_REQUIRE(shift.size() == 3);
			
			transformBase = Mat4d {
				{1, 0, 0, shift[0]},
				{0, 1, 0, shift[1]},
				{0, 0, 1, shift[2]},
				{0, 0, 0, 1}
			};
			break;
		}
		case Transformed<Geometry>::TURNED: {
			auto turned = input.getTurned();
			auto inAxis = turned.getAxis();
			auto inCenter = turned.getCenter();
			double ang = angle(turned.getAngle());
			
			KJ_REQUIRE(inAxis.size() == 3);
			KJ_REQUIRE(inCenter.size() == 3);
			
			Vec3d axis   { inAxis[0], inAxis[1], inAxis[2] };
			Vec3d center { inCenter[0], inCenter[1], inCenter[2] };
			
			transformBase = rotationAxisAngle(center, axis, ang);
			node = turned.getNode();
			break;
		}
		case Transformed<Geometry>::SCALED: {
			auto scaled = input.getScaled();
			auto scaleBy = scaled.getScale();
			
			KJ_REQUIRE(scaleBy.size() == 3);
			
			transformBase = Mat4d {
				{scaleBy[0], 0, 0, 0},
				{0, scaleBy[1], 0, 0},
				{0, 0, scaleBy[2], 0},
				{0, 0, 0, 1}
			};
			node = scaled.getNode();
			break;
		}
			
		default:
			KJ_FAIL_REQUIRE("Unknown transform type", input.which());
	}
			
	KJ_IF_MAYBE(pTransform, transform) {
		return mergeGeometries(node, tagTable, tagScope, (Mat4d)((*pTransform) * transformBase), output);
	} else {
		return mergeGeometries(node, tagTable, tagScope, transformBase, output);
	}
}

Promise<void> GeometryLibImpl::index(IndexContext context) {
	Geometry::Reader geometry = context.getParams().getGeometry();
	while(geometry.isNested()) {
		geometry = geometry.getNested();
	}
	
	DataRef<MergedGeometry>::Client merged = nullptr;
		
	switch(geometry.which()) {
		case Geometry::INDEXED: {
			auto indexed = geometry.getIndexed();
			if(ID::fromReader(indexed.getGrid()) == ID::fromReader(context.getParams().getGrid())) {
				context.initResults().setIndexed(indexed);
				return READY_NOW;
			}
			
			merged = indexed.getBase();
			break;
		}
		case Geometry::MERGED: {
			merged = geometry.getMerged();
			
			break;
		}
		default: {
			auto mergeRequest = thisCap().mergeRequest();
			mergeRequest.setNested(context.getParams().getGeometry());
			merged = mergeRequest.send().getRef();
			
			break;
		}
	}	
	
	// First we need to download the geometry we want to index
	return getActiveThread().dataService().download(merged)
	.then([this, context](LocalDataRef<MergedGeometry> inputRef) mutable {
		KJ_LOG(INFO, "Beginning indexing operation");
		
		// Create output temporary and read information about input
		auto output = context.getResults().initIndexed();
		MergedGeometry::Reader input = inputRef.get();
			
		auto grid = context.getParams().getGrid();
		// auto gridSize = grid.getSize();
		Vec3u gridSize { grid.getNX(), grid.getNY(), grid.getNZ() };
		
		size_t totalSize = gridSize[0] * gridSize[1] * gridSize[2];
		KJ_REQUIRE(totalSize <= MAX_GRID_SIZE, "Maximum indexing grid size exceeded");
		
		struct ElRefStruct {
			uint64_t meshIdx;
			uint64_t elementIdx;
		};
		
		// Allocate temporary un-aligned storage for grid refs
		kj::Vector<kj::Vector<ElRefStruct>> tmpRefs(totalSize);
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
					
					ElRefStruct& newRef = tmpRefs[globalIdx].add();
					newRef.meshIdx = iEntry;
					newRef.elementIdx = iPoly;
				}}}
			}
			
			// KJ_LOG(INFO, "Processed mesh", iEntry, input.getEntries().size());
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
			
			for(size_t iInner = 0; iInner < in.size(); ++iInner) {
				auto outRef = out[iInner];
				outRef.setMeshIndex(in[iInner].meshIdx);
				outRef.setElementIndex(in[iInner].elementIdx);
			}
			
			in.clear();
		}
		
		LocalDataRef<IndexedGeometry::IndexData> indexDataRef = getActiveThread().dataService().publish(indexData.asReader());
		indexData = nullptr;
		
		output.setGrid(grid);
		output.setBase(inputRef);
		output.setData(mv(indexDataRef));
	});
}

Promise<void> GeometryLibImpl::merge(MergeContext context) {
	// Check if already merged
	Geometry::Reader geometry = context.getParams();
	while(geometry.isNested()) {
		geometry = geometry.getNested();
	}
		
	switch(geometry.which()) {
		case Geometry::INDEXED: {
			context.getResults().setRef(geometry.getIndexed().getBase());
			return READY_NOW;
		}
		case Geometry::MERGED: {
			context.getResults().setRef(geometry.getMerged());
			return READY_NOW;
		}
		default: {			
			break;
		}
	}	
		
	// Prepare scratch pad structures that will hold intermediate data
	auto tagNameTable = heapHeld<kj::HashSet<kj::String>>();
	auto geomAccum = heapHeld<GeometryAccumulator>();
	
	// First collect all possible tag names into a table
	auto promise = collectTagNames(context.getParams(), *tagNameTable)
	
	// Then call "mergeGeometries" on the root node with an identity transform and empty tag scope
	//   This will collect temporary built meshes in geomAccum
	.then([context, geomAccum, tagNameTable, this]() mutable {
		Temporary<capnp::List<TagValue>> tagScope(tagNameTable->size());
		
		KJ_LOG(INFO, "Beginning merge operation");
		return mergeGeometries(context.getParams(), *tagNameTable, tagScope, nullptr, *geomAccum);
	})
	
	// Finally, copy the data from the accumulator into the output
	.then([context, geomAccum, tagNameTable, this]() mutable {
		KJ_LOG(INFO, "Merge complete");
		// Copy data over
		Temporary<MergedGeometry> output;
		geomAccum->finish(output);
		
		// Copy tag names from the tag table
		auto outTagNames = output.initTagNames(tagNameTable->size());
		for(size_t i = 0; i < tagNameTable->size(); ++i)
			outTagNames.set(i, *(tagNameTable->begin() + i));
		
		// Publish the merged geometry into the data store
		context.getResults().setRef(
			getActiveThread().dataService().publish(output.asReader())
		);
	});
	
	return promise.attach(tagNameTable.x(), geomAccum.x());
}

Promise<void> GeometryLibImpl::reduce(ReduceContext context) {
	auto params = context.getParams();
	
	// Check if already merged
	Geometry::Reader geometry = params.getGeometry();
	while(geometry.isNested()) {
		geometry = geometry.getNested();
	}
	
	DataRef<MergedGeometry>::Client ref = nullptr;
	switch(geometry.which()) {
		case Geometry::INDEXED: {
			ref = geometry.getIndexed().getBase();
		}
		case Geometry::MERGED: {
			ref = geometry.getMerged();
		}
		default: {
			auto mergeRequest = thisCap().mergeRequest();
			mergeRequest.setNested(geometry);
			ref = mergeRequest.sendForPipeline().getRef();
		}
	}
	
	return getActiveThread().dataService().download(ref)
	.then([context, params](LocalDataRef<MergedGeometry> localRef) mutable {
		KJ_LOG(INFO, "Beginning reduce operation");
		auto geometry = localRef.get();
		
		auto entries = geometry.getEntries();
		kj::Vector<Temporary<Mesh>> meshesOut;
		
		uint32_t iEntry = 0;
		
		while(iEntry < entries.size()) {
			kj::Vector<Mesh::Reader> meshes;
			
			uint32_t nVertTot = 0;
			uint32_t nIndTot = 0;
			
			while(iEntry < entries.size()) {
				auto mesh = entries[iEntry].getMesh();
				
				uint32_t nVert = mesh.getVertices().getShape()[0];
				uint32_t nInd = mesh.getIndices().size();
				
				if(meshes.size() > 0) {
					if(nVertTot + nVert > params.getMaxVertices())
						break;
					
					if(nIndTot + nInd > params.getMaxIndices())
						break;
				}
				
				meshes.add(mesh);
				nVertTot += nVert;
				nIndTot  += nInd;
				++iEntry;
			}
			
			Temporary<Mesh> newMesh;
			newMesh.initVertices().setShape({nVertTot, 3});
			auto vertData = newMesh.getVertices().initData(3 * nVertTot);
			auto indData = newMesh.initIndices(nIndTot);
			{
				uint32_t iVert = 0;
				uint32_t iInd = 0;
				
				for(auto mesh : meshes) {
					auto indIn = mesh.getIndices();
					auto vertIn = mesh.getVertices().getData();
					
					// Copy indices with appropriate shift
					for(auto i : kj::indices(mesh.getIndices())) {
						indData.set(i + iInd, indIn[i] + iVert);
					}
					
					// Copy vertices
					for(auto i : kj::indices(vertIn)) {
						vertData.set(i + 3 * iVert, vertIn[i]);
					}
					
					uint32_t nInd = indIn.size();
					uint32_t nVert = vertIn.size() / 3;
					
					iInd += nInd;
					iVert += nVert;
				}
			}
			
			// Copy polygon information
			bool allTri = true;
			for(auto mesh : meshes) {
				if(!mesh.isTriMesh()) {
					allTri = false;
					break;
				}
			}
			
			if(allTri) {
				newMesh.setTriMesh();
			} else {
				// Meh, we need to copy all the polygon data
				uint32_t polyCount = 0;
				for(auto mesh : meshes) {
					switch(mesh.which()) {
						case Mesh::TRI_MESH:
							polyCount += mesh.getIndices().size() / 3;
							break;
						case Mesh::POLY_MESH:
							polyCount += mesh.getPolyMesh().size();
							break;
						default:
							KJ_FAIL_REQUIRE("Unknown mesh type encountered in reduce operation");
					}
				}
				
				auto polys = newMesh.initPolyMesh(polyCount);
				uint32_t iPoly = 0;
				for(auto mesh : meshes) {
					if(mesh.isTriMesh()) {
						for(auto i : kj::range(0, mesh.getIndices().size() / 3)) {
							polys.set(iPoly++, 3);
						}
					} else {
						for(uint32_t poly : mesh.getPolyMesh()) {
							polys.set(iPoly++, poly);
						}
					}
				}
			}
			
			meshesOut.add(mv(newMesh));
		}
		
		KJ_LOG(INFO, "Reduction complete, publishing results", entries.size(), meshesOut.size());
		
		Temporary<MergedGeometry> merged;
		auto outEntries = merged.initEntries(meshesOut.size());
		
		for(auto i : kj::indices(outEntries)) {
			outEntries[i].setMesh(meshesOut[i]);
			meshesOut[i] = nullptr;
		}
		
		context.initResults().setRef(getActiveThread().dataService().publish(mv(merged)));
		KJ_LOG(INFO, "Reduction published");
	});
}

Promise<void> GeometryLibImpl::weightedSample(WeightedSampleContext context) {
	auto mergeRequest = thisCap().mergeRequest();
	mergeRequest.setNested(context.getParams().getGeometry());
	
	auto geoRef = mergeRequest.send().getRef();
	return getActiveThread().dataService().download(geoRef)
	.then([context](LocalDataRef<MergedGeometry> geoRef) mutable {
		auto geo = geoRef.get();
		
		struct IndexKey {
			int ix; int iy; int iz;
			
			IndexKey(Vec3d x, double scale) :
				ix(x(0) / scale), iy(x(1) / scale), iz(x(2) / scale)
			{}
			
			bool operator==(const IndexKey& o) {
				if(ix != o.ix) return false;
				if(iy != o.iy) return false;
				if(iz != o.iz) return false;
				return true;
			}
			
			bool operator<(const IndexKey& other) {
				if(ix < other.ix) return true;
				if(iy < other.iy) return true;
				return iz < other.iz;
			}
			
			unsigned int hashCode() {
				return kj::hashCode(ix, iy, iz);
			}
		};
		
		struct CellData {
			double totalArea;
			Vec3d weightedCenter;
						
			CellData(double area, Vec3d center) :
				totalArea(area), weightedCenter(center)
			{}
			
			CellData& operator+=(const CellData& o) {
				totalArea += o.totalArea;
				weightedCenter += o.weightedCenter;
				return *this;
			}
		};
		
		using MapType = kj::TreeMap<IndexKey, CellData>;
		MapType data;
		
		double scale = context.getParams().getScale();
		
		uint_fast32_t rngSeed;
		getActiveThread().rng().randomize(kj::ArrayPtr<byte>(reinterpret_cast<byte*>(&rngSeed), sizeof(uint_fast32_t)));
		
		std::mt19937 rng(rngSeed);
		std::uniform_real_distribution<double> pDist(0, 1);
		
		auto processTriangle = [&](Vec3d x1, Vec3d x2, Vec3d x3) {
			double d12 = (x2 - x1).norm();
			double d13 = (x3 - x1).norm();
			double d23 = (x3 - x2).norm();
			
			double length = std::max({d12, d23, d13});
			double area = (x2 - x1).cross(x3 - x1).norm();
			
			size_t nPoints = std::max(length / scale, area / (scale * scale));
			nPoints = std::max(nPoints, (size_t) 1);
			
			for(auto i : kj::range(0, nPoints)) {
				double tx = pDist(rng);
				double ty = pDist(rng);
				
				if(ty > tx)
					std::swap(tx, ty);
				
				Vec3d x = x1 + (x2 - x1) * tx + (x3 - x2) * ty;
				
				IndexKey key(x, scale);
				CellData value(area / nPoints, area / nPoints * x);
				
				data.upsert(
					key, value,
					[](CellData& existing, CellData&& update) {
						existing += update;
					}
				);
			}
		};
		
		for(auto e : geo.getEntries()) {
			auto mesh = e.getMesh();
			
			auto vertexData = mesh.getVertices().getData();
			auto indices = mesh.getIndices();
			
			size_t nPoints = vertexData.size() / 3;
		
			auto loadPoint = [&](size_t offset) {
				size_t i = indices[offset];
				
				double x = vertexData[3 * i + 0];
				double y = vertexData[3 * i + 1];
				double z = vertexData[3 * i + 2];
				
				return Vec3d(x, y, z);
			};
			
			if(mesh.isTriMesh()) {
				for(auto i : kj::range(0, indices.size() / 3)) {
					processTriangle(
						loadPoint(3 * i + 0),
						loadPoint(3 * i + 1),
						loadPoint(3 * i + 2)
					);
				}
			} else if(mesh.isPolyMesh()) {
				auto polys = mesh.getPolyMesh();
				KJ_REQUIRE(polys.size() > 0);
				
				for(auto iPoly : kj::range(0, polys.size() - 1)) {
					size_t polyStart = polys[iPoly];
					size_t polyEnd = polys[iPoly + 1];
					
					if(polyEnd - polyStart < 3)
						continue;
					
					Vec3d x1 = loadPoint(polyStart);
					Vec3d x2 = loadPoint(polyEnd);
					
					for(auto i3 : kj::range(polyStart + 2, polyEnd)) {
						Vec3d x3 = loadPoint(i3);
						processTriangle(x1, x2, x3);
					}
				}
			} else {
				KJ_FAIL_REQUIRE("Unknown mesh type");
			}
		}
		
		auto result = context.initResults();
		
		Eigen::Tensor<double, 2> centers(data.size(), 3);
		auto areas = result.initAreas(data.size());
		
		{
			size_t i = 0;
			for(auto& row : data) {
				CellData& resultData = row.value;
				
				double area = resultData.totalArea;
				areas.set(i, area);
				centers(i, 0) = resultData.weightedCenter[0] / area;
				centers(i, 1) = resultData.weightedCenter[1] / area;
				centers(i, 2) = resultData.weightedCenter[2] / area;
				
				++i;
			}
		}
		
		writeTensor(centers, result.initCenters());
	});
}



Promise<void> GeometryLibImpl::planarCut(PlanarCutContext context) {
	auto mergeRequest = thisCap().mergeRequest();
	mergeRequest.setNested(context.getParams().getGeometry());
	
	auto geoRef = mergeRequest.send().getRef();
	
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
			
			auto addLine = [&](Vec3d p1, Vec3d p2) {
				// Phi plane intersections are only for half planes
				if(orientation.isPhi()) {
					Vec3d er(std::cos(orientation.getPhi()), std::sin(orientation.getPhi()), 0);
					
					if(er.dot(p1) < 0) return;
					if(er.dot(p2) < 0) return;
				}
				
				lines.add(tuple(p1, p2));
			};
			
			auto processLine = [&](Vec3d p1, Vec3d p2) -> Maybe<Vec3d> {
				double d1 = normal.dot(p1) + d;
				double d2 = normal.dot(p2) + d;
				
				if(d1 * d2 > 0)
					return nullptr; // No intersection
				
				if(d1 == d2) {
					addLine(p1, p2); // Line intersects coplanar with plane
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
						if(nHit == 2)
							break;
						pair[nHit++] = *pHit;
					}
				}
				
				KJ_REQUIRE(nHit == 0 || nHit == 2, "Invalid mid-point count (should be 0 or 2)", nHit);
				
				if(nHit == 2)
					addLine(pair[0], pair[1]);
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