namespace fsc {

namespace devices { namespace w7x {

Promise<void> CoilsDBResolver::customField(MagneticField::Reader input, MagneticField::Builder output, ResolveContext& context) override {
	switch(input.which()) {
		case MagneticField::W7X_MAGNETIC_CONFIG:
			auto w7xConfig = input.getW7xMagneticConfig();
			
			output.initSum(N_MAIN_COILS + N_TRIM_COILS + N_CONTROL_COILS);
			
			LocalDataRef<CoilFields> coilFields = getCoilFields(w7xConfig.getCoils());
			size_t offset = 0;
			
			auto addOutput = [&](LocalDataRef<MagneticField> field, double current) {
				auto subField = output.getSum().init(offset++);
				auto scaleBy = subField.initScaleBy();
				scaleBy.setFactor(current);
				scaleBy.initField().setRef(field);				
			}
			
			for(size_t i = 0; i < 5; ++i) {
				addOutput(coilFields.getMainCoils()[i], w7xConfig.getNonplanar()[i]);
			}
			
			for(size_t i = 0; i < 2; ++i) {
				addOutput(coilFields.getMainCoils()[i + 5], w7xConfig.getPlanar()[i]);
			}
			
			KJ_ASSERT(offset == N_MAIN_COILS);
			
			for(size_t i = 0; i < N_TRIM_COILS; ++i) {
				addOutput(coilFields.getTrimCoils()[i], w7xConfig.getTrim()[i]);
			}
			
			//TODO: Extend to the 2 and 5 cases
			KJ_REQUIRE(w7xConfig.getControl().size() == 10);
			
			for(size_t i = 0; i < N_CONTROL_COILS; ++i) {
				addOutput(coilFields.getControlCoils()[i], w7xConfig.getControl()[i]);
			}
			
			KJ_ASSERT(offset == output.getSum().size());
			return kj::READY_NOW;
		
		default:
			return kj::READY_NOW;
	}
}

LocalDataRef<CoilFields> CoilsDBResolver::getCoilFields(W7XCoilSet::Reader reader) {
	ID id = ID::fromReader(reader);
	
	KJ_IF_MAYBE(coilPack, coilPacks.find(id)) {
		return *coilPack;
	}
	
	capnp::MallocMessageBuilder tmp;
	auto coilPack = tmp.initRoot<CoilFields>();
	resolveCoilPack(reader, coilPack);
	
	auto ref = lt.dataService().publish(id, coilPack);
	coilPacks.insert(id, ref);
	return ref;
}

DataRef<Filament>::Client CoilsDBResolver::getCoil(uint64_t cdbID) {
	KJ_IF_MAYBE(coilsRef, coils.findEntry(cdbID)) {
		return *coilsRef;
	}
	 
	auto coilRequest = backend.getCoilRequest();
	coilRequest.setId(cdbID);
	auto newCoil = coilRequest.send().getFilament();
	coils.insert(cdbID, newCoil);
	// TODO: Install an error handler that clears the entry if the request fails
	return newCoil;
}

Promise<void> CoilsDBResolver::customFilament(Filament     ::Reader input, Filament     ::Builder output, ResolveContext& context) override {
	switch(input.which()) {
		case Filament::W7X_COILS_DB:
			output.setRef(getCoil(input.getW7xCoilsDB()));
			return kj::READY_NOW;
			
		default:
			return kj::READY_NOW;
	}
}

void buildCoilFields(W7XCoilSet::Reader reader, CoilFields::Builder coilPack) {	
	auto getMainCoil = [=](size_t i_mod, size_t i_coil) -> DataRef<Filament>::Client {
		if(reader.isCoilsDBSet()) {
			size_t offset = reader.getCoilsDBSet().getMainCoilOffset();
			size_t coilID = i_coil < 5
				? offset + 5 * i_mod + i_coil
				: offset + 2 * i_mod + i_coil + 50;
			return getCoil(coilID);
		} else {
			KJ_REQUIRE(reader.isCustomCoils());
			return reader.getCustomCoils().getNonplanarCoils()[N_MAIN_COILS * i_mod + i_coil];
		}
	};
	
	auto getTrimCoil = [=](size_t i_coil) -> DataRef<Filament>::Client {
		if(reader.isCoilsDBSet())
			return getCoil(reader.getCoilsDBSet().getTrimCoilIDs()[i_coil]);
		else {
			KJ_REQUIRE(reader.isCustomCoils());
			return reader.getCustomCoils().getTrimCoils()[i_coil];
		}
	};
	
	auto getControlCoil = [=](size_t i_coil) -> DataRef<Filament>::Client {
		if(reader.isCoilsDBSet())
			return getCoil(reader.getCoilsDBSet().getControlCoilOffset() + i_coil);
		else {
			KJ_REQUIRE(reader.isCustomCoils());
			return reader.getCustomCoils().getControlCoils()[i_coil];
		}
	};
	
	auto initField = [=](MagneticField::Builder out, DataRef<Filament>::Client coil) {
		auto filField = out.initFilamentField();
		filField.setCurrent(1);
		filField.setWidth(reader.getWidth());
		filField.setWindingNo(MAIN_COIL_WINDINGS);
		filField.initFilament().setRef(coil);
	}
			
	coilPack.initMainCoils(N_MAIN_COILS);
	for(size_t i_coil = 0; i_coil < N_MAIN_COILS; ++icoil) {
		auto coilI = coilPack.getMainCoils().init(i_coil);
		auto coilI.initSum(N_MODULES);
		
		for(size_t i_mod = 0; i_mod < N_MODULES; ++i_mod) {
			initField(coilI.getSum()[i_mod], getMainCoil(i_mod, i_coil));
		}			
	}
	
	coilPack.initTrimCoils(N_TRIM_COILS);
	for(size_t i_coil = 0; i_coil < N_TRIM_COILS; ++i)
		initField(coilPack.getTrimCoils()[i_coil], getTrimCoil(i_coil));
	
	coilPack.initControlCoils(N_CONTROL_COILS);
	for(size_t i_coil = 0; i_coil < N_CONTROL_COILS; ++i)
		initField(coilPack.getControlCoils()[i_coil], getControlCoil(i_coil));
	
	return kj::READY_NOW;		
}
	
}}

}