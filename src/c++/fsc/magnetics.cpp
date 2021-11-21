namespace fsc {
	
unsigned int toroidalGridVersion(ToroidalGrid::Reader grid) {
	MallocMessageBuilder tmp;
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
	
	if(capnp::canonicalize(ref) == capnp::canonicalize(grid))
		return TGRID_V1;
	
	return TGRID_V_UNKNOWN;
}

virtual Promise<void> FieldResolverBase::resolve(ResolveContext context) override {
	auto input = context.getParams().getField();
	auto output = context.getResults().initField();
	
	return processField(input, output, context.getFollowRefs());
}

Promise<void> FieldResolverBase::processField(MagneticField::Reader input, MagneticField::Builder output, ResolveContext& context) {
	switch(input.which()) {
		case MagneticField::SUM:
			auto inSum = input.getSum();
			auto outSum = output.initSum(inSum.size());
			
			TaskSet subTasks;
			for(size_t i = 0; i < inSum.size(); ++i) {
				subTasks.add(processField(input[i], output[i], context));
			}
			
			return subTasks.onEmpty();
		
		case MagneticField::REF:
			if(!context.getFollowRefs()) {
				output.setRef(input.getRef());
				return kj::READY_NOW;
			}
			
			auto tmpMessage = kj::heap<MallocMessageBuilder>();
			MagneticField::Builder newOutput = tmpMessage -> initRoot<MagneticField>();
			
			return th.dataService().download(input.getRef()).then([newOutput, followRefs, &context] (LocalDataRef<MagneticField> ref) {
				return processField(ref.get(), newOutput, context)
			}).then([newOutput](){
				return output.setRef(lt.dataService().publish(lt.randomID(), newOutput));
			}).attach(mv(tmpMessage), thisCap());
		
		case MagneticField::COMPUTED_FIELD:
			output.setComputedField(input.getComputedField());
			return kj::READY_NOW;
		
		case MagneticField::FILAMENT_FIELD:
			auto filIn  = input.getFilamentField();
			auto filOut = output.initFilamentField();
			
			filOut.setCurrent(filIn.getCurrent());
			filOut.setWidth(filIn.getWidth());
			filOut.setWindingNo(filIn.getWindingNo());
			
			return processFilament(filIn.getFilament(), filOut.initFilament(), context);
		
		case MagneticField::SCALE_BY:
			output.setFactor(input.getFactor());
			return processField(input.getField(), output.initField(), context);
		
		case MagneticField::INVERT:
			return processField(input.getInvert(), output.initInvert(), context);
		
		default:
			return customField(input, output, context);			
	}
}

Promise<void> FieldResolverBase::processFilament(Filament::Reader input, Filament::Builder output, ResolveContext& context) {
	switch(input.which()) {
		case Filament::INLINE:
			output.setInline(input.getInline());
			return kj::READY_NOW;
		
		case Filament::REF:
			if(!context.getFollowRefs()) {
				output.setRef(input.getRef());
				return kj::READY_NOW;
			}
			
			auto tmpMessage = kj::heap<MallocMessageBuilder>();
			MagneticField::Builder newOutput = tmpMessage -> initRoot<MagneticField>();
			
			return th.dataService().download(input.getRef()).then([newOutput, followRefs] (LocalDataRef<MagneticField> ref) {
				return processFilament(ref.get(), newOutput, followRefs)
			}).then([newOutput](){
				return output.setRef(lt.dataService().publish(lt.randomID(), newOutput));
			}).attach(mv(tmpMessage), thisCap());
		
		default:
			return customFilament(Filament::Reader input, Filament::Builder output, bool followRefs);
	}
}


}