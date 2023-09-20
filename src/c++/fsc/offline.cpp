#include "offline.h"
#include "data.h"

#include "geometry.h"
#include "magnetics.h"

#include <kj/map.h>

namespace fsc {

namespace {
	
struct OfflineFieldResolver : public FieldResolverBase {
	LocalDataRef<OfflineData> source;
	
	kj::TreeMap<ID, MagneticField::Reader> fields;
	kj::TreeMap<ID, Filament::Reader> filaments;
	
	OfflineFieldResolver(LocalDataRef<OfflineData> _source) :
		source(mv(_source))
	{
		auto data = source.get();
		
		for(auto e : data.getFields()) {
			auto key = e.getKey();
			auto val = e.getVal();
			
			if(isBuiltin(key)) {
				KJ_LOG(WARNING, "Skipping built-in key in offline data");
				continue;
			}
			
			ID id = ID::fromReader(key);
			if(fields.find(id) == nullptr) {
				fields.insert(id, val);
			}
		}
		
		for(auto e : data.getCoils()) {
			auto key = e.getKey();
			auto val = e.getVal();
			
			if(isBuiltin(key)) {
				KJ_LOG(WARNING, "Skipping built-in key in offline data");
				continue;
			}
			
			ID id = ID::fromReader(key);
			if(filaments.find(id) == nullptr) {
				filaments.insert(id, val);
			}
		}
	}
	
	Promise<void> processField(MagneticField::Reader input, MagneticField::Builder output, ResolveFieldContext context) override {
		if(isBuiltin(input)) {
			return FieldResolverBase::processField(input, output, context);
		}
		
		try {
			ID asId = ID::fromReader(input);
			
			KJ_IF_MAYBE(pField, fields.find(asId)) {
				output.setNested(*pField);
				return READY_NOW;
			}
		} catch(kj::Exception e) {
		}
		
		output.setNested(input);
		return READY_NOW;
	}
	
	Promise<void> processFilament(Filament::Reader input, Filament::Builder output, ResolveFieldContext context) override {
		if(isBuiltin(input)) {
			return FieldResolverBase::processFilament(input, output, context);
		}
		
		try {
			ID asId = ID::fromReader(input);
			
			KJ_IF_MAYBE(pField, filaments.find(asId)) {
				output.setNested(*pField);
				return READY_NOW;
			}
		} catch(kj::Exception e) {
		}
		
		output.setNested(input);
		return READY_NOW;
	}
};

struct OfflineGeometryResolver : public GeometryResolverBase {
	LocalDataRef<OfflineData> source;
	
	kj::TreeMap<ID, Geometry::Reader> geometries;
	
	OfflineGeometryResolver(LocalDataRef<OfflineData> _source) :
		source(mv(_source))
	{
		auto data = source.get();
		
		for(auto e : data.getGeometries()) {
			auto key = e.getKey();
			auto val = e.getVal();
			
			if(isBuiltin(key)) {
				KJ_LOG(WARNING, "Skipping built-in key in offline data");
				continue;
			}
			
			ID id = asKey(key);
			if(geometries.find(id) == nullptr) {
				geometries.insert(id, val);
			}
		}
	}
	
	Promise<void> processGeometry(Geometry::Reader input, Geometry::Builder output, ResolveGeometryContext context) override {
		if(isBuiltin(input)) {
			return GeometryResolverBase::processGeometry(input, output, context);
		}
		
		try {
			ID asId = asKey(input);
			
			KJ_IF_MAYBE(pField, geometries.find(asId)) {
				output.setNested(*pField);
				return READY_NOW;
			}
		} catch(kj::Exception e) {
		}
		
		output.setNested(input);
		return READY_NOW;
	}
	
	ID asKey(Geometry::Reader reader) {
		Temporary<Geometry> stripped(reader);
		stripped.initTags(0);
		
		return ID::fromReader(stripped.asReader());
	}
};

}

FieldResolver::Client newOfflineFieldResolver(DataRef<OfflineData>::Client in) {
	return getActiveThread().dataService().download(in)
	.then([](auto in) mutable -> FieldResolver::Client {
		return kj::heap<OfflineFieldResolver>(mv(in));
	});
}

GeometryResolver::Client newOfflineGeometryResolver(DataRef<OfflineData>::Client in) {
	return getActiveThread().dataService().download(in)
	.then([](auto in) mutable -> GeometryResolver::Client {
		return kj::heap<OfflineGeometryResolver>(mv(in));
	});
}

}