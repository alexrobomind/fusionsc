#include "hdf5.h"

#include "hdf5_hl.h" // Required for dimension scale API

namespace fsc {
	H5Dim::H5Dim(const H5DataSet& ds) :
		dataset(ds)
	{
		auto space = ds.getSpace();
		
		// Check if the data space has the right no. of dimensions
		KJ_REQUIRE(space.isSimple());
		KJ_REQUIRE(space.getSimpleExtentNdims() == 1, "Dimensions must be 1d scales");
		
		space.getSimpleExtentDims(&length, &maxLength); // Since we have 1 dimension each, we can copy max lengths directly
	}
		
	H5::DataSet createDataSet(H5::Location& parent, kj::StringPtr name, H5::DataType dType, kj::ArrayPtr<H5Dim> dimensions) {
		size_t rank = dimensions.size();
		
		KJ_STACK_ARRAY(hsize_t, lengths, rank, 8, 32);
		KJ_STACK_ARRAY(hsize_t, maxLengths, rank, 8, 32);
		
		for(auto i : kj::indices(dimensions)) {
			lengths[i] = dimensions[i].length;
			maxLengths[i] = dimensions[i].maxLength;
		}
		
		H5::DataSpace space(rank, lengths.begin(), maxLengths.begin());
		H5::DataSet ds = parent.createDataSet(name.cStr(), dType, space);
		
		// Attach dimension scales
		for(auto i : kj::indices(dimensions)) {
			KJ_IF_MAYBE(pDim, dimensions[i].dataset) {
				H5DSattach_scale(ds.getId(), pDim -> getId(), i);
			}
		}
		
		return ds;
	}
	
	H5::DataSet createDimension(H5::Location& parent, kj::StringPtr name, const H5::DataType& dType, const H5Dim& dim) {
		H5::DataSpace space(1, &(dim.length), &(dim.maxLength));
		H5::DataSet ds = parent.createDataSet(name.cStr(), dType, space);
		
		H5DSset_scale(ds.getId(), name.cStr());
		return ds;
	}
	
	kj::Array<H5Dim> getDimensions(const H5::DataSet& ds) {
		auto space = ds.getSpace();
		KJ_REQUIRE(space.isSimple());
		
		size_t rank = space.getSimpleExtentNdims();
		KJ_STACK_ARRAY(hsize_t, lengths, rank, 8, 32);
		KJ_STACK_ARRAY(hsize_t, maxLengths, rank, 8, 32);
		KJ_STACK_ARRAY(Maybe<hid_t>, dims, rank, 8, 32);
		
		space.getSimpleExtentDims(lengths.begin(), maxLengths.begin());
		
		auto visitor = [](hid_t did, unsigned dim, hid_t dsid, void *visitor_data) noexcept -> herr_t {
			Maybe<hid_t>* begin = visitor_data;
			Maybe<hid_t>& target = *(begin + dim);
			
			if(target == nullptr) {
				target = dsid;
				H5Iinc_ref(dsid);
			}
			return 0;
		};
		
		for(auto i : kj::range(0, rank)) {
			H5DSiterate_scales(ds.getId(), i, nullptr, visitor, dims.begin());
		}
		
		auto out = kj::heapArrayBuilder<H5Dim>(rank);
		for(auto i : kj::range(0, rank)) {
			KJ_IF_MAYBE(pId, dims[i]) {
				out.add(H5::DataSet(*pId), lengths[i], maxLengths[i]);
				H5Idec_ref(*pId);
			} else {
				out.add(lengths[i], maxLengths[i]);
			}
		}
		return out.finish();
	}
	
	size_t totalSize(const H5::DataSpace& space) {
		auto space = ds.getSpace();
		KJ_REQUIRE(space.isSimple());
		
		size_t rank = space.getSimpleExtentNdims();
		KJ_STACK_ARRAY(hsize_t, lengths, rank, 8, 32);
		
		space.getSimpleExtentDims(lengths.begin(), nullptr);
		
		size_t prod = 1;
		for(auto l : lengths)
			prod *= l;
		return prod;
	}
}
