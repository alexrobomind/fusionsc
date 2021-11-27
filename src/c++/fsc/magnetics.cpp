#include <capnp/message.h>

#include "magnetics.h"
#include "data.h"

namespace fsc {
	
ToroidalGridVersion toroidalGridVersion(ToroidalGrid::Reader grid) {
	capnp::MallocMessageBuilder tmp;
	auto ref = tmp.initRoot<ToroidalGrid>();
	
	// Version 1
	ref.setRMin(grid.getRMin());
	ref.setRMax(grid.getRMax());
	ref.setZMin(grid.getZMin());
	ref.setZMax(grid.getZMax());
	ref.setNSym(grid.getNSym());
	
	ref.setNR(grid.getNR());
	ref.setNZ(grid.getNZ());
	ref.setNPhi(grid.getNPhi());
	
	if(capnp::canonicalize(ref.asReader()).asBytes() == capnp::canonicalize(grid).asBytes())
		return ToroidalGridVersion::V1;
	
	return ToroidalGridVersion::UNKNOWN;
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


}