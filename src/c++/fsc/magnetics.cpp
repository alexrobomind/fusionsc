#include <capnp/message.h>

#include "magnetics.h"
#include "magnetics-inl.h"
#include "data.h"

namespace fsc {
	
ToroidalGridStruct readGrid(ToroidalGrid::Reader in, unsigned int maxOrdinal) {
	KJ_REQUIRE(hasMaximumOrdinal(in, maxOrdinal));
	
	ToroidalGridStruct out;
	
	out.rMin = in.getRMin();
	out.rMax = in.getRMax();
	out.zMin = in.getZMin();
	out.zMax = in.getZMax();
	out.nR = (int) in.getNR();
	out.nZ = (int) in.getNZ();
	out.nSym = (int) in.getNSym();
	out.nPhi = (int) in.getNPhi();
	
	KJ_REQUIRE(out.isValid());
	return out;
}

void writeGrid(const ToroidalGridStruct& in, ToroidalGrid::Builder out) {
	KJ_REQUIRE(in.isValid());
	
	out.setRMin(in.rMin);
	out.setRMax(in.rMax);
	out.setZMin(in.zMin);
	out.setZMax(in.zMax);
	out.setNR(in.nR);
	out.setNZ(in.nZ);
	out.setNSym(in.nSym);
	out.setNPhi(in.nPhi);
}

Promise<void> FieldResolverBase::resolve(ResolveContext context) {
	auto input = context.getParams().getField();
	auto output = context.getResults().initField();
	
	return processField(input, output, context);
}

FieldResolverBase::FieldResolverBase(LibraryThread& lt) : lt(lt->addRef()) {}

Promise<void> FieldResolverBase::processField(MagneticField::Reader input, MagneticField::Builder output, ResolveContext context) {
	switch(input.which()) {
		case MagneticField::SUM: {
			auto inSum = input.getSum();
			auto outSum = output.initSum(inSum.size());
			
			auto subTasks = kj::heapArrayBuilder<Promise<void>>(inSum.size());
			for(unsigned int i = 0; i < inSum.size(); ++i) {
				subTasks.add(processField(inSum[i], outSum[i], context));
			}
			
			return kj::joinPromises(subTasks.finish()).attach(thisCap());
		}
		case MagneticField::REF: {
			if(!context.getParams().getFollowRefs()) {
				output.setRef(input.getRef());
				return kj::READY_NOW;
			}
			
			auto tmpMessage = kj::heap<capnp::MallocMessageBuilder>();
			MagneticField::Builder newOutput = tmpMessage -> initRoot<MagneticField>();
			
			return lt->dataService().download(input.getRef())
			.then([this, newOutput, context] (LocalDataRef<MagneticField> ref) mutable {
				return processField(ref.get(), newOutput, context);
			}).then([this, output, newOutput]() mutable {
				return output.setRef(lt->dataService().publish(lt->randomID(), newOutput));
			}).attach(mv(tmpMessage), thisCap());
		}
		case MagneticField::COMPUTED_FIELD: {
			output.setComputedField(input.getComputedField());
			return kj::READY_NOW;
		}
		case MagneticField::FILAMENT_FIELD: {
			auto filIn  = input.getFilamentField();
			auto filOut = output.initFilamentField();
			
			filOut.setCurrent(filIn.getCurrent());
			filOut.setBiotSavartSettings(filIn.getBiotSavartSettings());
			filOut.setWindingNo(filIn.getWindingNo());
			
			return processFilament(filIn.getFilament(), filOut.initFilament(), context);
		}
		case MagneticField::SCALE_BY: {
			output.initScaleBy().setFactor(input.getScaleBy().getFactor());
			return processField(input.getScaleBy().getField(), output.getScaleBy().initField(), context);
		}
		case MagneticField::INVERT: {
			return processField(input.getInvert(), output.initInvert(), context);
		}
		default:
			return kj::READY_NOW;	
	}
}

Promise<void> FieldResolverBase::processFilament(Filament::Reader input, Filament::Builder output, ResolveContext context) {
	switch(input.which()) {
		case Filament::INLINE: {
			output.setInline(input.getInline());
			return kj::READY_NOW;
		}
		case Filament::REF: {
			if(!context.getParams().getFollowRefs()) {
				output.setRef(input.getRef());
				return kj::READY_NOW;
			}
			
			auto tmpMessage = kj::heap<capnp::MallocMessageBuilder>();
			Filament::Builder newOutput = tmpMessage -> initRoot<Filament>();
			
			return lt->dataService().download(input.getRef())
			.then([this, newOutput, context] (LocalDataRef<Filament> ref) mutable {
				return processFilament(ref.get(), newOutput, context);
			}).then([this, output, newOutput]() mutable {
				return output.setRef(lt->dataService().publish(lt->randomID(), newOutput));
			}).attach(mv(tmpMessage), thisCap());
		}
		default:
			return kj::READY_NOW;
	}
}

FieldCalculator::Client newCPUFieldCalculator(LibraryThread& lt) {
	using D = Eigen::ThreadPoolDevice;
	
	auto pool = kj::heap<Eigen::ThreadPool>(numThreads());
	auto dev  = kj::heap<Eigen::ThreadPoolDevice>(pool.get(), numThreads());
	dev = dev.attach(mv(pool));
	
	return FieldCalculator::Client(
		kj::heap<internal::FieldCalculatorImpl<D>>(lt, mv(dev))
	);
	/*return FieldCalculator::Client(
		kj::heap<FieldCalculatorImpl<Eigen::DefaultDevice>>(lt, kj::heap<Eigen::DefaultDevice>())
	);*/
}

#ifdef FSC_WITH_CUDA

#include <cuda_runtime_api.h>

FieldCalculator::Client newGPUFieldCalculator(LibraryThread& lt) {
	using D = Eigen::GpuDevice;
	
	auto stream = kj::heap<Eigen::GpuStreamDevice>();
	
	cudaError_t streamStatus = cudaStreamQuery(stream->stream());
	KJ_REQUIRE(streamStatus == cudaSuccess, "CUDA stream could not be initialized", streamStatus);
	
	auto dev    = kj::heap<Eigen::GpuDevice>(stream);
	
	streamStatus = cudaStreamQuery(stream->stream());
	KJ_REQUIRE(streamStatus == cudaSuccess, "CUDA device could not be initialized", streamStatus);
	
	dev = dev.attach(mv(stream));
	
	return FieldCalculator::Client(
		kj::heap<internal::FieldCalculatorImpl<D>>(lt, mv(dev))
	);
}

#else
	

FieldCalculator::Client newGPUFieldCalculator(LibraryThread& lt) {
	KJ_UNIMPLEMENTED( "FSC was not compiled with GPU support");
}

#endif

}

