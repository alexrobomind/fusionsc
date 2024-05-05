#include "geometry.h"

#include <happly.h>

namespace fsc {

Temporary<Geometry> readPly(kj::StringPtr filename, size_t nMaxVerts, size_t nMaxEntries) {
	constexpr uint32_t CAPNP_MAX_LIST_SIZE = (1 << 29) - 1;
	
	KJ_REQUIRE(nMaxEntries <= CAPNP_MAX_LIST_SIZE);
	KJ_REQUIRE(nMaxVerts <= CAPNP_MAX_LIST_SIZE);
	
	if(nMaxVerts == 0) nMaxVerts = CAPNP_MAX_LIST_SIZE;
	if(nMaxEntries == 0) nMaxEntries = CAPNP_MAX_LIST_SIZE;
	
	happly::PLYData plyIn(filename.cStr());
	
	std::vector<std::array<double, 3>> vertices = plyIn.getVertexPositions();
	std::vector<std::vector<size_t>> faces = plyIn.getFaceIndices<size_t>();
	
	kj::Vector<DataRef<Mesh>::Client> meshes;
	
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
		Temporary<Mesh> mesh;
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
		
		meshes.add(getActiveThread().dataService().publish(mesh.asReader()));
	}
	
	Temporary<Geometry> result;
	auto comb = result.initCombined(meshes.size());
	
	for(auto i : kj::indices(comb)) {
		comb[i].setMesh(mv(meshes[i]));
	}
	
	return result;
}

void writePly(kj::ArrayPtr<Mesh::Reader> meshes, kj::StringPtr filename, bool binary) {
	std::vector<std::array<double, 3>> meshVertexPositions;
	std::vector<std::vector<size_t>> meshFaceIndices;
	
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
			meshVertexPositions.push_back(pos);
		}
	
		auto indexData = mesh.getIndices();
	
		if(mesh.isTriMesh()) {
			size_t nTri = indexData.size() / 3;
			for(auto i : kj::range(0, nTri)) {
				std::vector<size_t> face;
				face.reserve(3);
				
				face.push_back(indexData[3 * i + 0] + vertexOffset);
				face.push_back(indexData[3 * i + 1] + vertexOffset);
				face.push_back(indexData[3 * i + 2] + vertexOffset);
				
				meshFaceIndices.push_back(mv(face));
			}
		} else {
			auto polys = mesh.getPolyMesh();
			for(auto iFace : kj::range(0, polys.size() - 1)) {
				size_t i1 = polys[iFace];
				size_t i2 = polys[iFace + 1];
				
				std::vector<size_t> face;
				face.reserve(i2 - i1);
				
				for(auto i : kj::range(i1, i2)) {
					face.push_back(indexData[i] + vertexOffset);
				}
				
				meshFaceIndices.push_back(mv(face));
			}
		}
	}
		
	happly::PLYData plyOut;
	plyOut.addVertexPositions(meshVertexPositions);
	plyOut.addFaceIndices(meshFaceIndices);
	
	plyOut.write(filename.cStr(), binary ? happly::DataFormat::Binary : happly::DataFormat::ASCII);
}

Promise<void> writePly(Geometry::Reader geo, kj::StringPtr filename, bool binary) {
	// Try to push the whole geometry into a single mesh
	// constexpr uint32_t CAPNP_MAX_LIST_SIZE = 1 << 29 - 1;
	// auto req = newGeometryLib().reduceRequest();
	auto req = newGeometryLib().mergeRequest();
	req.setNested(geo);
	// req.setGeometry(geo);
	// req.setMaxVertices(CAPNP_MAX_LIST_SIZE);
	// req.setMaxIndices(CAPNP_MAX_LIST_SIZE);
	
	auto mergedRef = req.send().getRef();
	
	return getActiveThread().dataService().download(mergedRef)
	.then([filename = kj::heapString(filename), binary](auto mergedLocalRef) {
		auto entries = mergedLocalRef.get().getEntries();
		
		kj::Vector<Mesh::Reader> meshes;
		for(auto e : entries) {
			meshes.add(e.getMesh());
		}
		
		writePly(meshes, filename, binary);
	});
}


}