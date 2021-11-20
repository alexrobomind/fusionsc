#include <fsc/magnetics.capnp.h>

#include <kj/map.h>

namespace fsc {
	
class FieldResolverBase : public FieldResolver::Server {
public:
	LibraryThread lt;
	
	virtual Promise<void> resolve(ResolveContext context) override {
		auto input = context.getParams().getField();
		auto output = context.getResults().initField();
		
		return processField(input, output, context.getFollowRefs());
	}
	
	Promise<void> processField(MagneticField::Reader input, MagneticField::Builder output, ResolveContext& context) {
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
	
	Promise<void> processFilament(Filament::Reader input, Filament::Builder output, ResolveContext& context) {
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
	
	virtual Promise<void> customField   (MagneticField::Reader input, MagneticField::Builder output, ResolveContext& context) = 0;
	virtual Promise<void> customFilament(Filament     ::Reader input, Filament     ::Builder output, ResolveContext& context) = 0;
}
	
namespace w7x {
	
namespace internal {
	void resolveCoilPack(W7XCoilSet::Reader reader, ResolvedCoilPack::Builder coilPack) {		
		constexpr size_t N_MAIN_COILS = 7;
		constexpr size_t N_MODULES = 10;
		constexpr size_t N_TRIM_COILS = 5;
		constexpr size_t N_CONTROL_COILS = 10;
		
		constexpr uint32_t MAIN_COIL_WINDINGS = 108;
		
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
		}
		
		auto getControlCoil = [=](size_t i_coil) -> DataRef<Filament>::Client {
			if(reader.isCoilsDBSet())
				return getCoil(reader.getCoilsDBSet().getControlCoilOffset() + i_coil);
			else {
				KJ_REQUIRE(reader.isCustomCoils());
				return reader.getCustomCoils().getControlCoils()[i_coil];
			}
		}
		
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
		
		
		
		KJ_REQUIRE(false, "Unfinished");		
	}
}

class CoilsDBResolver : public FieldResolverBase {
	kj::TreeMap<uint64_t, DataRef<Filament>::Client> coils;
	kj::TreeMap<ID, LocalDataRef<ResolvedCoilPack>> coilPacks;
	
	fsc::w7x::CoilsDB backend;
	
	Promise<void> customField   (MagneticField::Reader input, MagneticField::Builder output, ResolveContext& context) override {
		switch(input.which()) {
			case MagneticField::W7X_MAGNETIC_CONFIG:
				KJ_ASSERT(false, "This should be implemented, but isn't.");
			
			default:
				return kj::READY_NOW;
		}
	}
	
	LocalDataRef<ResolvedCoilPack> getCoilPack(W7XCoilSet::Reader reader) {
		ID id = ID::fromReader(reader);
		
		KJ_IF_MAYBE(coilPack, coilPacks.find(id)) {
			return *coilPack;
		}
		
		capnp::MallocMessageBuilder tmp;
		auto coilPack = tmp.initRoot<ResolvedCoilPack>();
		resolveCoilPack(reader, coilPack);
		
		auto ref = lt.dataService().publish(id, coilPack);
		coilPacks.insert(id, ref);
		return ref;
	}
	
	DataRef<Filament>::Client getCoil(uint64_t cdbID) {
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
	
	Promise<void> customFilament(Filament     ::Reader input, Filament     ::Builder output, ResolveContext& context) override {
		switch(input.which()) {
			case Filament::W7X_COILS_DB:
				output.setRef(getCoil(input.getW7xCoilsDB()));
				return kj::READY_NOW;
				
			default:
				return kj::READY_NOW;
		}
	}
};



}

}