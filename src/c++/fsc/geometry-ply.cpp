#include "geometry.h"

#include <happly.h>

namespace fsc {

Temporary<Geometry> readPly(kj::StringPtr filename) {	
	happly::PLYData plyIn(filename.cStr());
	
	std::vector<std::array<double, 3>> vertices = plyIn.getVertexPositions();
	std::vector<std::vector<size_t>> faces = plyIn.getFaceIndices<size_t>();
	
	kj::Vector<kj::Array<const size_t>> facesKj;
	facesKj.reserve(faces.size());
	
	for(auto& f : faces) {
		facesKj.add(kj::heapArray<const size_t>(f.begin(), f.end()));
	}
	
	faces.clear();
	
	Temporary<MergedGeometry> merged;
	importRaw(
		kj::ArrayPtr<std::array<const double, 3>>((std::array<const double, 3>*) vertices.data(), vertices.size()),
		facesKj.releaseAsArray(),
		merged
	);
	
	Temporary<Geometry> result;
	result.setMerged(getActiveThread().dataService().publish(merged.asReader()));
	return result;
}

void writePly(MergedGeometry::Reader merged, kj::StringPtr filename, bool binary) {
	auto doExport = [&](kj::Array<std::array<double, 3>>&& pos, kj::Array<kj::Array<size_t>>&& ind) {
		std::vector<std::array<double, 3>> meshVertexPositions(pos.begin(), pos.end());
		std::vector<std::vector<size_t>> meshFaceIndices(ind.size());
		
		pos = nullptr;
		
		for(auto& e : ind) {
			meshFaceIndices.push_back(std::vector<size_t>(e.begin(), e.end()));
		}
		
		ind = nullptr;
			
		happly::PLYData plyOut;
		plyOut.addVertexPositions(meshVertexPositions);
		plyOut.addFaceIndices(meshFaceIndices);
		
		plyOut.write(filename.cStr(), binary ? happly::DataFormat::Binary : happly::DataFormat::ASCII);
	};
	
	kj::apply(doExport, exportRaw(merged, /* triangulate = */ false));
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
		writePly(mergedLocalRef.get(), filename, binary);
	});
}


}