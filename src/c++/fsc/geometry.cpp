#include "geometry.h"
#include "poly.h"
#include "data.h"
#include "tensor.h"

#include "geometry-kernels.h"

#include "kernels/karg.h"
#include "kernels/launch.h"

#include <random>
#include <list>

#include <kj/map.h>

namespace fsc {

namespace {	
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
		case Geometry::FILTER:
		case Geometry::QUAD_MESH:
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
			if(input.getTags().size() == 0)
				return processGeometry(input.getNested(), output, context);
			else
				return processGeometry(input.getNested(), output.initNested(), context);
		}
		case Geometry::MESH: {
			output.setMesh(input.getMesh());
			return READY_NOW;
		}
		case Geometry::MERGED: {
			output.setMerged(input.getMerged());
			return READY_NOW;
		}
		case Geometry::FILTER: {
			auto fIn = input.getFilter();
			auto fOut = output.initFilter();
			
			fOut.setFilter(fIn.getFilter());
			return processGeometry(fIn.getGeometry(), fOut.initGeometry(), context);
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

double angle(Angle::Reader in) {
	switch(in.which()) {
		case Angle::RAD: return in.getRad();
		case Angle::DEG: return pi / 180 * in.getDeg();
	}
	KJ_FAIL_REQUIRE("Unknown angle type");
}
	
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

namespace {

	struct FilterContext : kj::Refcounted {
		Maybe<Own<FilterContext>> parent;
		Temporary<GeometryFilter> filter;
		
		FilterContext() {}
		
		FilterContext(kj::Maybe<FilterContext&> maybeParent, GeometryFilter::Reader f) :
			filter(f)
		{
			KJ_IF_MAYBE(pP, maybeParent) {
				parent = pP -> addRef();
			}
		}
		
		Own<FilterContext> addRef() { return kj::addRef(*this); }
		
		bool matches(kj::HashSet<kj::String>& tagTable, capnp::List<TagValue>::Reader tagValues) const {
			if(!matchImpl(filter, tagTable, tagValues))
				return false;
			
			KJ_IF_MAYBE(pParent, parent) {
				return (**pParent).matches(tagTable, tagValues);
			} else {
				return true;
			}
		}
		
	private:
		bool matchImpl(GeometryFilter::Reader f, kj::HashSet<kj::String>& tagTable, capnp::List<TagValue>::Reader tagValues) const {
			switch(f.which()) {
				case GeometryFilter::TRUE: return true;
				case GeometryFilter::AND: {
					for(auto sub : f.getAnd()) {
						if(!matchImpl(sub, tagTable, tagValues))	return false;
					}
					
					return true;
				}
				
				case GeometryFilter::OR: {
					for(auto sub : f.getOr()) {
						if(matchImpl(sub, tagTable, tagValues)) return true;
					}
					
					return false;
				}
				
				case GeometryFilter::NOT:
					return !matchImpl(f.getNot(), tagTable, tagValues);
				
				case GeometryFilter::IS_ONE_OF: {
					auto oneOf = f.getIsOneOf();
					
					TagValue::Reader value;
		
					KJ_IF_MAYBE(rowPtr, tagTable.find(oneOf.getTagName())) {
						size_t tagIdx = rowPtr - tagTable.begin();
						value = tagValues[tagIdx];
					}
					
					for(auto c : oneOf.getValues()) {
						if(c.which() != value.which()) continue;
						
						switch(c.which()) {
							case TagValue::NOT_SET:
								return true;
							case TagValue::U_INT64:
								if(c.getUInt64() == value.getUInt64())
									return true;
								break;
							case TagValue::TEXT:
								if(c.getText() == value.getText())
									return true;
								break;
						}
					}
					
					return false;
				}
				
				default:
					KJ_FAIL_REQUIRE("Unknown filter type");
			}
		}
	};
	
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
	
	struct MergeScope : kj::Refcounted {
		kj::HashSet<kj::String>& tagTable;
		GeometryAccumulator& output;
		
		Temporary<capnp::List<TagValue>> tagValues;
		Maybe<Mat4d> transform;
		Own<FilterContext> filterContext;
		
		MergeScope(kj::HashSet<kj::String>& tbl, GeometryAccumulator& out) :
			tagTable(tbl), output(out),
			
			tagValues(tbl.size()),
			transform(nullptr), filterContext(kj::refcounted<FilterContext>())
		{}
		
		MergeScope(MergeScope& in) :
			tagTable(in.tagTable), output(in.output),
			
			tagValues(in.tagValues.asReader()),
			transform(in.transform), filterContext(in.filterContext -> addRef())
		{}
		
		Own<MergeScope> clone() { return kj::refcounted<MergeScope>(*this); }
		Own<MergeScope> addRef() { return kj::addRef(*this); }
		
		bool matches() { return filterContext -> matches(tagTable, tagValues); }
		
		void setTag(kj::StringPtr tagName, TagValue::Reader value) {
			KJ_IF_MAYBE(rowPtr, tagTable.find(tagName)) {
				size_t tagIdx = rowPtr - tagTable.begin();
				
				if(tagValues[tagIdx].isNotSet())
					tagValues.setWithCaveats(tagIdx, value);
			} else {
				KJ_FAIL_REQUIRE("Internal error, tag not found in tag table", tagName);
			}
		}
	};
}

// Class GeometryLibImpl

struct GeometryLibImpl : public GeometryLib::Server {
	Own<DeviceBase> device;
	
	GeometryLibImpl(Own<DeviceBase> device) : device(mv(device)) {}
	
	Promise<void> merge(MergeContext) override;
	Promise<void> index(IndexContext) override;
	Promise<void> planarCut(PlanarCutContext) override;
	Promise<void> reduce(ReduceContext) override;
	Promise<void> weightedSample(WeightedSampleContext) override;
	Promise<void> intersect(IntersectContext) override;
	Promise<void> unroll(UnrollContext) override;
	Promise<void> planarClip(PlanarClipContext) override;
	Promise<void> triangulate(TriangulateContext) override;
	
private:
	
	Promise<void> mergeGeometries(Geometry::Reader input, Own<MergeScope> scope);
	Promise<void> mergeGeometries(Transformed<Geometry>::Reader input, Own<MergeScope> scope);
	
	Promise<void> collectTagNames(Geometry::Reader input, kj::HashSet<kj::String>& output);
	Promise<void> collectTagNames(Transformed<Geometry>::Reader input, kj::HashSet<kj::String>& output);
};

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
		case Geometry::FILTER:
			return collectTagNames(input.getFilter().getGeometry(), output);
		case Geometry::MERGED:
			return handleMerged(input.getMerged());
		case Geometry::INDEXED:
			return handleMerged(input.getIndexed().getBase());
		case Geometry::WRAP_TOROIDALLY:
			return READY_NOW;
		case Geometry::QUAD_MESH:
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

Promise<void> GeometryLibImpl::mergeGeometries(Geometry::Reader input, Own<MergeScope> scope) {	
	if(input.getTags().size() > 0 && scope -> isShared())
		scope = scope -> clone();
	
	for(auto tag : input.getTags()) {
		const kj::StringPtr tagName = tag.getName();
		scope -> setTag(tag.getName(), tag.getValue());
	}
	
	auto handleMesh = [](Mesh::Reader inputMesh, MergeScope& scope, bool matchChecked = false) {
		if(!matchChecked && !scope.matches()) return;
		
		auto vertexShape = inputMesh.getVertices().getShape();
		KJ_REQUIRE(vertexShape.size() == 2);
		KJ_REQUIRE(vertexShape[1] == 3);
		
		Temporary<MergedGeometry::Entry> newEntry;
		newEntry.setTags(scope.tagValues);
		
		// Copy mesh and make in-place adjustments
		newEntry.setMesh(inputMesh);
		
		KJ_IF_MAYBE(pTransform, scope.transform) {
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
		
		scope.output.entries.add(mv(newEntry));
	};
	
	auto handleMerged = [handleMesh](DataRef<MergedGeometry>::Client ref, MergeScope& scope) -> Promise<void> {
		return getActiveThread().dataService().download(ref)
		.then([handleMesh = mv(handleMesh), scope = scope.addRef()](LocalDataRef<MergedGeometry> localRef) mutable {
			auto merged = localRef.get();
			
			for(auto entry : merged.getEntries()) {
				Own<MergeScope> meshScope = scope -> clone();
				
				auto tagNames = merged.getTagNames();
				auto eTagVals = entry.getTags();
				
				for(auto iTag : kj::indices(merged.getTagNames())) {
					auto tagName = tagNames[iTag];
					auto tagVal  = eTagVals[iTag];
					
					meshScope -> setTag(tagName, tagVal);
				}
				
				handleMesh(entry.getMesh(), *meshScope);
			}
		});
	};
	
	switch(input.which()) {
		case Geometry::COMBINED: {
			auto promises = kj::heapArrayBuilder<Promise<void>>(input.getCombined().size());
			
			for(auto child : input.getCombined()) {
				promises.add(mergeGeometries(child, scope -> addRef()));
			}
				
			return joinPromises(promises.finish());
		}
		case Geometry::TRANSFORMED:
			return mergeGeometries(input.getTransformed(), mv(scope));
			
		case Geometry::REF:
			return getActiveThread().dataService().download(input.getRef())
			.then([scope = mv(scope), this](LocalDataRef<Geometry> geo) mutable {
				return mergeGeometries(geo.get(), mv(scope));
			});
		case Geometry::NESTED:
			return mergeGeometries(input.getNested(), mv(scope));
			
		case Geometry::MESH:
			if(!scope -> matches()) return READY_NOW;
			
			return getActiveThread().dataService().download(input.getMesh())
			.then([scope = mv(scope), handleMesh](LocalDataRef<Mesh> inputMeshRef) mutable {
				handleMesh(inputMeshRef.get(), *scope, /* matchChecked = */ true);
			});
		
		case Geometry::MERGED:
			return handleMerged(input.getMerged(), *scope);
		
		case Geometry::INDEXED:
			return handleMerged(input.getIndexed().getBase(), *scope);
		
		case Geometry::FILTER: {
			if(scope -> isShared()) scope = scope -> clone();
			scope -> filterContext = kj::refcounted<FilterContext>(
				*(scope -> filterContext),
				input.getFilter().getFilter()
			);
			
			return mergeGeometries(input.getFilter().getGeometry(), mv(scope));
		}
		
		case Geometry::WRAP_TOROIDALLY: {
			if(!scope -> matches()) return READY_NOW;
			
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
			
			using A = Eigen::array<Eigen::Index, 2>;
			
			Tensor<double, 2> flatVerts = vertices.reshape(A({3, nVerts * (nPhi + 1)}));
			KJ_LOG(INFO, "Verts reshaped");
			
			{
				Temporary<Mesh> tmpMesh;
				writeTensor(flatVerts, tmpMesh.getVertices());
				
				auto indices = tmpMesh.initIndices(3 * 2 * nLines * nPhi);
				for(auto i : kj::indices(indices))
					indices.set(i, triangles.data()[i]);
				
				tmpMesh.setTriMesh();
				handleMesh(tmpMesh, *scope, /* matchChecked = */ true);
			}
			
			// Generate end caps
			if(close) {
				Tensor<double, 2> rz(nVerts, 2);
				for(auto i : kj::range(0, nVerts)) {
					rz(i, 0) = r[i];
					rz(i, 1) = z[i];
				}
				
				Tensor<uint32_t, 2> triangulation = fsc::triangulate(rz);
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
					handleMesh(tmpMesh, *scope, /* matchChecked = */ true);
				}
			}
			return READY_NOW;
		}
		
		case Geometry::QUAD_MESH: {
			if(!scope -> matches())
				return READY_NOW;
			
			auto qm = input.getQuadMesh();
			
			auto data = qm.getVertices();
			bool wrapU = qm.getWrapU();
			bool wrapV = qm.getWrapV();
			
			return getActiveThread().dataService().download(data)
			.then([scope = mv(scope), handleMesh, wrapU, wrapV](auto localRef) mutable {
				auto input = localRef.get();
				auto shape = input.getShape();
				KJ_REQUIRE(shape.size() == 3);
				KJ_REQUIRE(shape[2] == 3);
				
				uint32_t nU = shape[0];
				uint32_t nV = shape[1];
				
				auto linearIndex = [&](int64_t u, int64_t v) {
					if(u < 0) u += nU;
					if(v < 0) v += nV;
					
					u %= nU;
					v %= nV;
					
					return v + nV * u;
				};
				
				kj::Vector<uint32_t> indices(4 * nU * nV);
				for(auto iU : kj::range(0, wrapU ? nU : (nU - 1))) {
					for(auto iV : kj::range(0, wrapV ? nV : (nV - 1))) {
						indices.add(linearIndex(iU, iV));
						indices.add(linearIndex(iU + 1, iV));
						indices.add(linearIndex(iU + 1, iV + 1));
						indices.add(linearIndex(iU, iV + 1));
					}
				}
				
				kj::Vector<uint32_t> polys;
				for(uint32_t i = 0; i <= indices.size(); i += 4) // Yes, <= is correct here
					polys.add(i);
				
				Temporary<Mesh> mesh;
				mesh.initVertices().setData(input.getData());
				mesh.getVertices().setShape({nU * nV, 3});
				
				mesh.setIndices(indices);
				mesh.setPolyMesh(polys);
				
				handleMesh(mesh, *scope, /* matchChecked = */ true);
			});
		}
			
		default:
			KJ_FAIL_REQUIRE("Unresolved geometry node encountered during merge operation", input.which());
	}
}

Promise<void> GeometryLibImpl::mergeGeometries(Transformed<Geometry>::Reader input, Own<MergeScope> scope) {
	Mat4d transformBase;
	Transformed<Geometry>::Reader node;
	
	switch(input.which()) {
		case Transformed<Geometry>::LEAF:
			return mergeGeometries(input.getLeaf(), mv(scope));
			
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
	
	// Modify shallow copy of scope
	if(scope -> isShared())
		scope = scope -> clone();
	
	KJ_IF_MAYBE(pTransform, scope -> transform) {
		scope -> transform = (Mat4d)((*pTransform) * transformBase);
	} else {
		scope -> transform = transformBase;
	};
	
	return mergeGeometries(node, mv(scope));
}

Promise<void> GeometryLibImpl::index(IndexContext context) {
	Geometry::Reader geometry = context.getParams().getGeometry();
	while(geometry.isNested() && geometry.getTags().size() == 0) {
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
		auto grid = context.getParams().getGrid();
		
		return getActiveThread().worker().executeAsync([inputRef = inputRef.deepFork(), grid]() mutable {
			MergedGeometry::Reader input = inputRef.get();
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
			
			return indexData;
		})
		.then([context, grid, inputRef](auto indexData) mutable {
			// Create output temporary and read information about input
			auto output = context.getResults().initIndexed();
			
			LocalDataRef<IndexedGeometry::IndexData> indexDataRef = getActiveThread().dataService().publish(indexData.asReader());
			indexData = nullptr;
			
			output.setGrid(grid);
			output.setBase(inputRef);
			output.setData(mv(indexDataRef));
		});
	});
}

Promise<void> GeometryLibImpl::merge(MergeContext context) {
	// Check if already merged
	Geometry::Reader geometry = context.getParams();
	while(geometry.isNested() && geometry.getTags().size() == 0) {
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
		auto mergeScope = kj::refcounted<MergeScope>(*tagNameTable, *geomAccum);
		
		KJ_LOG(INFO, "Beginning merge operation");
		return mergeGeometries(context.getParams(), mv(mergeScope));
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
	while(geometry.isNested() && geometry.getTags().size() == 0) {
		geometry = geometry.getNested();
	}
	
	DataRef<MergedGeometry>::Client ref = nullptr;
	switch(geometry.which()) {
		case Geometry::INDEXED: {
			ref = geometry.getIndexed().getBase();
			break;
		}
		case Geometry::MERGED: {
			ref = geometry.getMerged();
			break;
		}
		default: {
			auto mergeRequest = thisCap().mergeRequest();
			mergeRequest.setNested(geometry);
			ref = mergeRequest.sendForPipeline().getRef();
			break;
		}
	}
	
	return getActiveThread().dataService().download(ref)
	.then([context, params](LocalDataRef<MergedGeometry> localRef) mutable {
		return getActiveThread().worker().executeAsync([localRef = localRef.deepFork(), params]() mutable {
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
					uint32_t indexOffset = 0;
					
					for(auto mesh : meshes) {
						if(mesh.isTriMesh()) {
							for(auto i : kj::range(0, mesh.getIndices().size() / 3)) {
								polys.set(iPoly++, 3 * i + indexOffset);
							}
						} else {
							for(uint32_t poly : mesh.getPolyMesh()) {
								polys.set(iPoly++, poly + indexOffset);
							}
						}
						
						indexOffset += mesh.getIndices().size();
					}
				}
				
				meshesOut.add(mv(newMesh));
			}
							
			Temporary<MergedGeometry> merged;
			auto outEntries = merged.initEntries(meshesOut.size());
			
			for(auto i : kj::indices(outEntries)) {
				outEntries[i].setMesh(meshesOut[i]);
				meshesOut[i] = nullptr;
			}
			
			return merged;
		})
		.then([context](auto merged) mutable {
			context.initResults().setRef(getActiveThread().dataService().publish(mv(merged)));
		});
	});
}

Promise<void> GeometryLibImpl::weightedSample(WeightedSampleContext context) {
	auto mergeRequest = thisCap().mergeRequest();
	mergeRequest.setNested(context.getParams().getGeometry());
	
	auto geoRef = mergeRequest.send().getRef();
	return getActiveThread().dataService().download(geoRef)
	.then([context](LocalDataRef<MergedGeometry> geoRef) mutable {
		double scale = context.getParams().getScale();
		return getActiveThread().worker().executeAsync([geoRef = geoRef.deepFork(), scale]() mutable {
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
			
			return data;
		})
		.then([context](auto data) mutable {
			auto result = context.initResults();
			
			Eigen::Tensor<double, 2> centers(data.size(), 3);
			auto areas = result.initAreas(data.size());
			
			{
				size_t i = 0;
				for(auto& row : data) {
					auto& resultData = row.value;
					
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
	});
}

void interpretPlane(Plane::Reader plane, Vec3d& normal, double& d) {
	auto orientation = plane.getOrientation();
	
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
	
	d = -center.dot(normal);
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
		double d;
		interpretPlane(context.getParams().getPlane(), normal, d);
		
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

Promise<void> GeometryLibImpl::intersect(IntersectContext ctx) {
	return kj::startFiber(1024 * 1024, [this, ctx = mv(ctx)](kj::WaitScope& ws) mutable {
		// Merge geometry
		auto params = ctx.getParams();
		
		auto& ds = getActiveThread().dataService();
		
		// Download all required data
		Temporary<IndexedGeometry> indexedGeo = ctx.getParams().getGeometry();
		auto indexData = ds.download(indexedGeo.getData()).wait(ws);
		auto base = ds.download(indexedGeo.getBase()).wait(ws);
		
		Tensor<double, 2> p1;
		readVardimTensor(params.getPStart(), 1, p1);
		
		Tensor<double, 2> p2;
		readVardimTensor(params.getPEnd(), 1, p2);
		
		KJ_REQUIRE(p2.dimension(0) == p1.dimension(0), "No. of start and end points inconsistent");
		size_t nPoints = p1.dimension(0);
		
		KJ_REQUIRE(p1.dimension(1) == 3);
		KJ_REQUIRE(p2.dimension(1) == 3);
		
		// Prepare kernel space
		auto mappedGeo = FSC_MAP_BUILDER(::fsc, IndexedGeometry, mv(indexedGeo), *device, true);
		auto mappedData = FSC_MAP_READER(::fsc, IndexedGeometry::IndexData, indexData, *device, true);
		auto mappedBase = FSC_MAP_READER(::fsc, MergedGeometry, base, *device, true);
		auto mappedResults = mapToDevice(kj::heapArray<IntersectResult>(nPoints), *device, true);
		
		// Start kernel and wait for result
		FSC_LAUNCH_KERNEL(
			rayCastKernel, *device,
			
			nPoints,
			
			FSC_KARG(p1, ALIAS_IN), FSC_KARG(p2, ALIAS_IN),
			FSC_KARG(mappedBase, IN), FSC_KARG(mappedGeo, IN), FSC_KARG(mappedData, IN),
			
			mappedResults
		)
		.wait(ws);
		
		// Wait until memcpy to host is finished
		device -> barrier().wait(ws);
		
		// Post-process
		auto kernelResult = mappedResults -> getHost();
		auto results = ctx.initResults();
		
		// Set positions
		auto lambdas = results.initLambda().initData(nPoints);
		auto positions = results.initPosition().initData(3 * nPoints);
		for(auto i : kj::indices(lambdas)) {
			double l = kernelResult[i].l;
			
			if(l > 1)
				l = std::numeric_limits<double>::infinity();
			
			lambdas.set(i, l);
			positions.set(3 * i + 0, l * p2(i, 0) + (1 - l) * p1(i, 0));
			positions.set(3 * i + 1, l * p2(i, 1) + (1 - l) * p1(i, 1));
			positions.set(3 * i + 2, l * p2(i, 2) + (1 - l) * p1(i, 2));
		}
		
		auto inShape = params.getPStart().getShape();
		auto vectorShape = kj::heapArray<uint64_t>(inShape.begin(), inShape.end());
		auto scalarShape = vectorShape.slice(1, vectorShape.size());
		
		results.getPosition().setShape(vectorShape);
		results.getLambda().setShape(scalarShape);
		
		// Set tags
		size_t nTags = base.get().getTagNames().size();
		auto tagEntries = results.initTags(nTags);
		
		auto baseVal = base.get();
		
		for(auto iTag : kj::range(0, nTags)) {
			auto entry = tagEntries[iTag];
			entry.setName(base.get().getTagNames()[iTag]);
			
			auto values = entry.initValues();
			values.setShape(scalarShape);
			auto data = values.initData(nPoints);
			
			for(auto iPoint : kj::range(0, nPoints)) {
				auto& res = kernelResult[iPoint];
				
				if(res.l > 1)
					continue;
				
				data.setWithCaveats(iPoint, baseVal.getEntries()[kernelResult[iPoint].iMesh].getTags()[iTag]);
			}
		}
	});
}

// Unrolls a single mesh into the specified phi range. Can produce a lot of excess polygons.
void unrollMesh(double phi1, double phi2, Mesh::Reader meshIn, Mesh::Builder meshOut) {	
	Tensor<double, 2> pointsIn;
	readTensor(meshIn.getVertices(), pointsIn);
	
	kj::Vector<Vec3d> pointsOut;
	kj::HashMap<uint64_t, uint32_t> lookupTable;
	
	kj::Vector<uint32_t> idxBuffer;
	kj::Vector<uint32_t> polyBuffer;
	polyBuffer.add(0);
	
	auto idxBufferIn = meshIn.getIndices();
	
	auto getPoint = [&](size_t idxIn) -> Vec3d {
		return Vec3d(pointsIn(0, idxIn), pointsIn(1, idxIn), pointsIn(2, idxIn));
	};
	
	// Adds a point to the output mesh unwrapped near a specific angle. Checks
	// whether the input, output pair already exists
	auto generatePoint = [&](uint32_t idxIn, double near) -> uint32_t {
		auto pointIn = getPoint(idxIn);
		double phi = atan2(pointIn[1], pointIn[0]);
		
		int32_t offset = round((near - phi) / (2 * pi));
		
		uint64_t key = idxIn;
		key <<= 32;
		key |= offset;
		
		KJ_IF_MAYBE(pPointIdx, lookupTable.find(key)) {
			return *pPointIdx;
		}
		
		size_t idxFull = pointsOut.size();
		uint32_t idx = idxFull;
		KJ_REQUIRE(idx == idxFull, "Index buffer limit exceeded for a mesh");
		lookupTable.insert(key, idx);
		
		double z = pointIn[2];
		double r = sqrt(pointIn[0] * pointIn[0] + pointIn[1] * pointIn[1]);
		
		phi += offset * 2 * pi;
		pointsOut.add(Vec3d(phi, z, r));
		
		return idx;
	};
	
	// Handles an input polygon as an index buffer range
	auto handlePoly = [&](uint32_t iStart, uint32_t iEnd) {
		if(iEnd <= iStart)
			return;
		
		// Extract loop
		kj::Array<uint32_t> loop = kj::heapArray<uint32_t>(iEnd - iStart);
		for(auto i : kj::indices(loop)) {
			loop[i] = idxBufferIn[i + iStart];
		}
		
		// Compute phi of polygon's mean point
		Vec3d mean;
		for(auto idx : loop)
			mean += getPoint(idx);
		mean /= loop.size();
		double phiMean = atan2(mean(1), mean(0));
		
		kj::HashSet<int> offsets;
		
		// Generate a version of this polygon in phi, z, r space near a specific angle
		auto generateNear = [&](double near) {
			// Check if we already generated a polygon at this location
			int offset = round((near - phiMean) / (2 * pi));
			if(offsets.contains(offset))
				return;
			
			offsets.insert(offset);
			// Generate (or reuse) corner points near the new center point
			double newCenterAngle = phiMean + offset * 2 * pi;
			for(auto idx : loop)
				idxBuffer.add(generatePoint(idx, newCenterAngle));
			
			// Register new polygon
			polyBuffer.add(idxBuffer.size());
		};
		
		double angle = phi1;
		while(angle < phi2) {
			generateNear(angle);
			angle += 2 * pi;
		}
		
		generateNear(phi2);
	};
	
	if(meshIn.isTriMesh()) {
		for(uint32_t i = 0; i + 3 <= idxBufferIn.size(); i += 3)
			handlePoly(i, i + 3);
	} else if(meshIn.isPolyMesh()) {
		auto pm = meshIn.getPolyMesh();
		
		for(auto i = 0; i + 1 < pm.size(); ++i) {
			handlePoly(pm[i], pm[i + 1]);
		}
	} else {
		KJ_FAIL_REQUIRE("Unknown mesh type");
	}
		
	{
		meshOut.getVertices().setShape({pointsOut.size(), 3});
		auto resultPoints = meshOut.getVertices().initData(3 * pointsOut.size());
		
		for(auto i : kj::indices(pointsOut)) {
			resultPoints.set(3 * i + 0, pointsOut[i][0]);
			resultPoints.set(3 * i + 1, pointsOut[i][1]);
			resultPoints.set(3 * i + 2, pointsOut[i][2]);
		}
	}
	
	meshOut.setIndices(idxBuffer);
	
	if(meshIn.isTriMesh())
		meshOut.setTriMesh();
	else
		meshOut.setPolyMesh(polyBuffer);
}

// Clip mesh on a normal * x + d >= 0 space
void clipMesh(Vec3d normal, double d, Mesh::Reader meshIn, Mesh::Builder meshOut) {
	Tensor<double, 2> pointsIn;
	readTensor(meshIn.getVertices(), pointsIn);
	
	auto getPoint = [&](size_t idxIn) -> Vec3d {
		return Vec3d(pointsIn(0, idxIn), pointsIn(1, idxIn), pointsIn(2, idxIn));
	};
	
	// Compute for all points where they are on the plane
	kj::Array<const double> planeCoeffs = nullptr;
	{
		auto builder = kj::heapArrayBuilder<const double>(pointsIn.dimension(1));
		
		for(auto i : kj::range(0, pointsIn.dimension(1))) {
			double coeff = 0;
			coeff += pointsIn(0, i) * normal(0);
			coeff += pointsIn(1, i) * normal(1);
			coeff += pointsIn(2, i) * normal(2);
			coeff += d;
			
			builder.add(coeff);
		}
		planeCoeffs = builder.finish();
	}
	
	// Copy all points on the correct size to the output mesh
	auto copied = kj::heapArray<Maybe<uint32_t>>(planeCoeffs.size());
	
	kj::Vector<Vec3d> pointsOut;
	for(auto i : kj::indices(planeCoeffs)) {
		if(planeCoeffs[i] >= 0) {
			copied[i] = (uint32_t) pointsOut.size();
			pointsOut.add(getPoint(i));
		}
	}
	
	// Helper to construct the midpoint of an edge
	kj::HashMap<uint64_t, uint32_t> midpointMap;
	auto edgeMidpoint = [&](uint32_t p1, uint32_t p2) -> uint32_t {
		if(p1 > p2) std::swap(p1, p2);
		
		uint64_t key = p1;
		key <<= 32;
		key |= p2;
		
		KJ_IF_MAYBE(pIdx, midpointMap.find(key)) {
			return *pIdx;
		}
		
		Vec3d x1 = getPoint(p1);
		Vec3d x2 = getPoint(p2);
		double l1 = planeCoeffs[p1];
		double l2 = planeCoeffs[p2];
		
		// We want
		//      a * l1 + (1 - a) * l2 = 0
		// <=>  a * (l1 - l2) = -l2
		// <=>  a = -l2 / (l1 - l2) = l2 / (l2 - l1)
		
		double a = l2 / (l2 - l1);
		Vec3d x = a * x1 + (1 - a) * x2;
		
		uint32_t idx = pointsOut.size();
		pointsOut.add(x);
		midpointMap.insert(key, idx);
		return idx;
	};
	
	auto indicesIn = meshIn.getIndices();
	kj::Vector<uint32_t> indicesOut;
	kj::Vector<uint32_t> polysOut;
	polysOut.add(0);
	auto handlePoly = [&](uint32_t start, uint32_t end) {
		if(start + 1 >= end)
			return;
			
		uint32_t leaveCounter = 0;
		uint32_t enterCounter = 0;
		
		auto handleEdge = [&](uint32_t p1, uint32_t p2) {
			double l1 = planeCoeffs[p1];
			double l2 = planeCoeffs[p2];
			
			KJ_IF_MAYBE(p1New, copied[p1]) {
				KJ_IF_MAYBE(p2New, copied[p2]) {
					// Interior edge in new space. Just add 2nd point
					indicesOut.add(*p2New);
				} else {
					// Edge leaves space. Add midpoint
					indicesOut.add(edgeMidpoint(p1, p2));
				}
			} else {
				KJ_IF_MAYBE(p2New, copied[p2]) {
					// Edge enters space. Add midpoint AND DESTINATION
					indicesOut.add(edgeMidpoint(p2, p1));
					indicesOut.add(*p2New);
				} else {
					// Edge on wrong side. Ignore.
				}
			}
		};
		
		for(auto i = start; i + 1 < end; ++i) {
			handleEdge(indicesIn[i], indicesIn[i+1]);
		}
		handleEdge(indicesIn[end - 1], indicesIn[start]);
		
		// Only add if polygon non-empty
		if(polysOut[polysOut.size() - 1] != indicesOut.size())
			polysOut.add(indicesOut.size());
		
		if(leaveCounter != enterCounter) {
			KJ_FAIL_REQUIRE("Internal error: Edge operations did not tally up correctle", leaveCounter, enterCounter);
		}
		
		if(enterCounter > 1) {
			KJ_LOG(WARNING, "Non-convex polygon encountered - loop crosses cut plane more than twice.", enterCounter, leaveCounter);
		}
	};
	
	if(meshIn.isTriMesh()) {
		for(uint32_t i = 0; i + 3 <= indicesIn.size(); i += 3)
			handlePoly(i, i + 3);
	} else if(meshIn.isPolyMesh()) {
		auto pm = meshIn.getPolyMesh();
		
		for(auto i = 0; i + 1 < pm.size(); ++i) {
			handlePoly(pm[i], pm[i + 1]);
		}
	} else {
		KJ_FAIL_REQUIRE("Unknown mesh type");
	}
		
	{
		meshOut.getVertices().setShape({pointsOut.size(), 3});
		auto resultPoints = meshOut.getVertices().initData(3 * pointsOut.size());
		
		for(auto i : kj::indices(pointsOut)) {
			resultPoints.set(3 * i + 0, pointsOut[i][0]);
			resultPoints.set(3 * i + 1, pointsOut[i][1]);
			resultPoints.set(3 * i + 2, pointsOut[i][2]);
		}
	}
	
	meshOut.setIndices(indicesOut);
	
	bool isTriMesh = true;
	for(auto i : kj::indices(polysOut)) {
		if(polysOut[i] != 3 * i) {
			isTriMesh = false;
			break;
		}
	}
	
	if(isTriMesh)
		meshOut.setTriMesh();
	else
		meshOut.setPolyMesh(polysOut);
}

Promise<void> GeometryLibImpl::unroll(UnrollContext ctx) {
	auto mergeRequest = thisCap().mergeRequest();
	mergeRequest.setNested(ctx.getParams().getGeometry());
	
	auto geoRef = mergeRequest.send().getRef();
	return getActiveThread().dataService().download(geoRef)
	.then([ctx](auto localRef) mutable {
		double phi1 = angle(ctx.getParams().getPhi1());
		double phi2 = angle(ctx.getParams().getPhi2());
		bool clip = ctx.getParams().getClip();
		
		return getActiveThread().worker().executeAsync([localRef = localRef.deepFork(), phi1, phi2, clip]() mutable {
			auto mergedIn = localRef.get();
			
			Temporary<MergedGeometry> result;
			result.setTagNames(mergedIn.getTagNames());
			
			auto entriesIn = mergedIn.getEntries();
			auto entriesOut = result.initEntries(entriesIn.size());
			
			for(auto i : kj::indices(entriesOut)) {
				entriesOut[i].setTags(entriesIn[i].getTags());
				
				if(!clip) {
					auto meshIn = entriesIn[i].getMesh();
					auto meshOut = entriesOut[i].getMesh();
				
					unrollMesh(phi1, phi2, meshIn, meshOut);
				} else {				
					Temporary<Mesh> intermediate1;
					Temporary<Mesh> intermediate2;
					
					unrollMesh(phi1, phi2, entriesIn[i].getMesh(), intermediate1);
					clipMesh(Vec3d(1, 0, 0), -phi1, intermediate1, intermediate2);
					clipMesh(Vec3d(-1, 0, 0), phi2, intermediate2, entriesOut[i].getMesh());
				}
			}
			
			return result;
		});
	})
	.then([ctx](auto result) mutable {
		ctx.initResults().setRef(getActiveThread().dataService().publish(mv(result)));
	});
}

Promise<void> GeometryLibImpl::planarClip(PlanarClipContext ctx) {
	auto mergeRequest = thisCap().mergeRequest();
	mergeRequest.setNested(ctx.getParams().getGeometry());
	
	auto geoRef = mergeRequest.send().getRef();
	return getActiveThread().dataService().download(geoRef)
	.then([ctx](auto localRef) mutable {
		auto mergedIn = localRef.get();
		
		Vec3d normal;
		double d;
		interpretPlane(ctx.getParams().getPlane(), normal, d);
		
		Temporary<MergedGeometry> result;
		result.setTagNames(mergedIn.getTagNames());
		
		auto entriesIn = mergedIn.getEntries();
		auto entriesOut = result.initEntries(entriesIn.size());
		
		for(auto i : kj::indices(entriesOut)) {
			entriesOut[i].setTags(entriesIn[i].getTags());
			
			auto meshIn = entriesIn[i].getMesh();
			auto meshOut = entriesOut[i].getMesh();
			
			clipMesh(normal, d, meshIn, meshOut);
		}
		
		ctx.initResults().setRef(getActiveThread().dataService().publish(mv(result)));
	});
}

void triangulateMesh(Mesh::Reader in, Mesh::Builder out, double maxEdgeLength) {
	kj::Vector<Vec3d> points;
	
	{
		Tensor<double, 2> pointsIn;
		readTensor(in.getVertices(), pointsIn);
		KJ_REQUIRE(pointsIn.dimension(0) == 3);
		
		points.reserve(pointsIn.dimension(1));
		for(auto i : kj::range(0, pointsIn.dimension(1))) {
			points.add(pointsIn(0, i), pointsIn(1, i), pointsIn(2, i));
		}
	}
	
	kj::HashMap<uint64_t, uint32_t> midpointMap;
	auto midpoint = [&](uint32_t p1, uint32_t p2) -> uint32_t {
		if(p1 > p2) std::swap(p1, p2);
		
		uint64_t key = p1;
		key <<= 32;
		key |= p2;
		
		KJ_IF_MAYBE(pIdx, midpointMap.find(key)) {
			return *pIdx;
		}
		
		const Vec3d x1 = points[p1];
		const Vec3d x2 = points[p2];
		
		uint32_t idx = points.size();
		points.add(0.5 * (x1 + x2));
		
		midpointMap.insert(key, idx);
		return idx;
	};
	
	auto edgeLength = [&](uint32_t p1, uint32_t p2) -> double {
		Vec3d& x1 = points[p1];
		Vec3d& x2 = points[p2];
		
		return (x2 - x1).norm();
	};
	
	using Tri = kj::Tuple<uint32_t, uint32_t, uint32_t>;
	std::list<Tri> queue;
	
	// Triangulate input mesh into subdivision queue
	{
		auto indices = in.getIndices();
		auto handlePoly = [&](uint32_t start, uint32_t end) {
			for(uint32_t i = start + 1; i + 1 < end; ++i) {
				queue.push_back(kj::tuple(indices[start], indices[i], indices[i + 1]));
			}
		};
		
		if(in.isPolyMesh()) {
			auto poly = in.getPolyMesh();
			KJ_REQUIRE(poly.size() > 0);
			for(auto i : kj::range(0, poly.size() - 1)) {
				handlePoly(poly[i], poly[i + 1]);
			}
		} else {
			for(auto i = 0; i + 3 <= indices.size(); i += 3) {
				handlePoly(i, i + 3);
			}
		}
	}
	
	kj::Vector<uint32_t> indices;
	auto handleTri = [&](uint32_t p1, uint32_t p2, uint32_t p3) {
		double l12 = edgeLength(p1, p2);
		double l23 = edgeLength(p2, p3);
		double l31 = edgeLength(p3, p1);
		
		if(maxEdgeLength == 0 || kj::max(l12, kj::max(l23, l31)) <= maxEdgeLength) {
			indices.add(p1);
			indices.add(p2);
			indices.add(p3);
			return;
		}
		
		if(l12 > std::max(l23, l31)) {
			uint32_t p12 = midpoint(p1, p2);
			queue.push_back(kj::tuple(p1, p12, p3));
			queue.push_back(kj::tuple(p12, p2, p3));
		} else if(l23 > l31) {
			uint32_t p23 = midpoint(p2, p3);
			queue.push_back(kj::tuple(p1, p2, p23));
			queue.push_back(kj::tuple(p23, p3, p1));
		} else {
			uint32_t p31 = midpoint(p3, p1);
			queue.push_back(kj::tuple(p1, p2, p31));
			queue.push_back(kj::tuple(p31, p2, p3));
		}
	};
	
	while(!queue.empty()) {
		Tri back = queue.back();
		queue.pop_back();
		kj::apply(handleTri, back);
	}
	
	auto pointsOut = out.initVertices();
	pointsOut.setShape({points.size(), 3});
	
	auto pointsData = pointsOut.initData(3 * points.size());
	for(auto i : kj::indices(points)) {
		pointsData.set(3 * i + 0, points[i][0]);
		pointsData.set(3 * i + 1, points[i][1]);
		pointsData.set(3 * i + 2, points[i][2]);
	}
	
	out.setIndices(indices);
	out.setTriMesh();
}

Promise<void> GeometryLibImpl::triangulate(TriangulateContext ctx) {
	auto mr = thisCap().mergeRequest();
	mr.setNested(ctx.getParams().getGeometry());
	
	return getActiveThread().dataService().download(mr.send().getRef())
	.then([ctx](auto localRef) mutable {
		double maxEdgeLength = ctx.getParams().getMaxEdgeLength();
		
		return getActiveThread().worker().executeAsync([maxEdgeLength, localRef = localRef.deepFork()]() mutable {
			MergedGeometry::Reader merged = localRef.get();
			
			Temporary<MergedGeometry> result;
			result.setTagNames(merged.getTagNames());
			
			auto entriesIn = merged.getEntries();
			auto entriesOut = result.initEntries(entriesIn.size());
			for(auto i : kj::indices(entriesIn)) {
				auto eIn = entriesIn[i];
				auto eOut = entriesOut[i];
				eOut.setTags(eIn.getTags());
				triangulateMesh(eIn.getMesh(), eOut.getMesh(), maxEdgeLength);
			}
			
			return result;
		}).then([ctx](auto result) mutable {
			ctx.initResults().setRef(getActiveThread().dataService().publish(mv(result)));
		});
	});
}

Own<GeometryLib::Server> newGeometryLib(Own<DeviceBase> device) {
	return kj::heap<GeometryLibImpl>(mv(device));
};

// Grid location methods

Vec3u locationInGrid(Vec3d point, CartesianGrid::Reader grid) {
	Vec3d min { grid.getXMin(), grid.getYMin(), grid.getZMin() };
	Vec3d max { grid.getXMax(), grid.getYMax(), grid.getZMax() };
	Vec3u size { grid.getNX(), grid.getNY(), grid.getNZ() };
	
	return locationInGrid(point, min, max, size);
}

}
