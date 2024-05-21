#include "geometry.h"

namespace fsc {

void importRaw(kj::ArrayPtr<std::array<const double, 3>> vertices, kj::ArrayPtr<kj::Array<const size_t>> faces, MergedGeometry::Builder out) {
	constexpr uint32_t CAPNP_MAX_LIST_SIZE = (1 << 29) - 1;
	
	constexpr uint32_t nMaxVerts = CAPNP_MAX_LIST_SIZE;
	constexpr uint32_t nMaxEntries = CAPNP_MAX_LIST_SIZE;
	
	auto orphanage = capnp::Orphanage::getForMessageContaining(out);
	
	kj::Vector<capnp::Orphan<Mesh>> meshes;
	
	size_t nActive = 0;
	auto vertexMap = kj::heapArray<Maybe<size_t>>(vertices.size());
	auto checkPoint = [&](size_t i) {
		KJ_IF_MAYBE(pVal, vertexMap[i]) {
			return *pVal;
		} else {
			vertexMap[i] = nActive;
			return nActive++;
		}
	};
			
	size_t iFace = 0;
	while(iFace < faces.size()) {
		// Reset vertex map
		for(auto& e : vertexMap) e = nullptr;
		nActive = 0;
		
		kj::Vector<uint32_t> idxOut;
		kj::Vector<uint32_t> facesOut;
		facesOut.add(0);
		
		for(; iFace < faces.size(); ++iFace) {
			auto& face = faces[iFace];
			
			if(idxOut.size() + face.size() > nMaxEntries)
				break;
			if(nActive + face.size() > nMaxVerts)
				break;
			
			for(auto i : face)
				idxOut.add(checkPoint(i));
			
			facesOut.add(idxOut.size());
		}
		
		// Collect point data into mesh
		auto meshHolder = orphanage.newOrphan<Mesh>();
		Mesh::Builder mesh = meshHolder.get();
		mesh.initVertices().setShape({nActive, 3});
		auto data = mesh.getVertices().initData(3 * nActive);
		
		for(auto iSrc : kj::indices(vertexMap)) {
			KJ_IF_MAYBE(pDstIdx, vertexMap[iSrc]) {
				size_t iDst = *pDstIdx;
				data.set(3 * iDst + 0, vertices[iSrc][0]);
				data.set(3 * iDst + 1, vertices[iSrc][1]);
				data.set(3 * iDst + 2, vertices[iSrc][2]);
			}
		}
		
		mesh.setIndices(idxOut);
		
		// Check if we are a triangle mesh
		bool isTri = true;
		for(auto i : kj::indices(faces)) {
			if(facesOut[i] != 3 * i) {
				isTri = false;
				 break;
			}
		}
		
		if(isTri) {
			mesh.setTriMesh();
		} else {
			mesh.setPolyMesh(facesOut);
		}
		
		meshes.add(mv(meshHolder));
	}
	
	auto entries = out.initEntries(meshes.size());
	for(auto i : kj::indices(meshes)) {
		entries[i].adoptMesh(mv(meshes[i]));
	}
}

kj::Tuple<kj::Array<std::array<double, 3>>, kj::Array<kj::Array<size_t>>> exportRaw(MergedGeometry::Reader merged, bool triangulate) {
	kj::Vector<std::array<double, 3>> meshVertexPositions;
	kj::Vector<kj::Array<size_t>> meshFaceIndices;
	
	kj::Vector<Mesh::Reader> meshes(merged.getEntries().size());
	for(auto e : merged.getEntries())
		meshes.add(e.getMesh());
	
	{
		size_t totalSize = 0;
		for(auto mesh : meshes) {
			totalSize += mesh.getVertices().getData().size() / 3;
		}
		
		meshVertexPositions.reserve(totalSize);
	}
	
	for(auto mesh : meshes) {
		const size_t vertexOffset = meshVertexPositions.size();
		
		auto vertexData = mesh.getVertices().getData();
		const size_t nVertices = vertexData.size() / 3;
		
		for(auto i : kj::range(0, nVertices)) {
			std::array<double, 3> pos = {
				vertexData[3 * i + 0],
				vertexData[3 * i + 1],
				vertexData[3 * i + 2]
			};
			meshVertexPositions.add(pos);
		}
	
		auto indexData = mesh.getIndices();
	
		if(mesh.isTriMesh()) {
			size_t nTri = indexData.size() / 3;
			for(auto i : kj::range(0, nTri)) {
				kj::Vector<size_t> face(3);
				
				face.add(indexData[3 * i + 0] + vertexOffset);
				face.add(indexData[3 * i + 1] + vertexOffset);
				face.add(indexData[3 * i + 2] + vertexOffset);
				
				meshFaceIndices.add(face.releaseAsArray());
			}
		} else {
			auto polys = mesh.getPolyMesh();
			for(auto iFace : kj::range(0, polys.size() - 1)) {
				size_t i1 = polys[iFace];
				size_t i2 = polys[iFace + 1];
				
				// Only add faces with at least 3 points
				if(i2 < i1 + 3) continue;
				
				if(triangulate) {
					size_t t1 = indexData[i1];
					
					for(auto i : kj::range(i1 + 1, i2 - 1)) {
						kj::Vector<size_t> face(3);
						face.add(t1);
						face.add(indexData[i]);
						face.add(indexData[i + 1]);
						
						meshFaceIndices.add(face.releaseAsArray());
					}
				} else {
					kj::Vector<size_t> face(i2 - i1);
					
					for(auto i : kj::range(i1, i2)) {
						face.add(indexData[i] + vertexOffset);
					}
					
					meshFaceIndices.add(face.releaseAsArray());
				}
			}
		}
	}
	
	return kj::tuple(meshVertexPositions.releaseAsArray(), meshFaceIndices.releaseAsArray());
}

}