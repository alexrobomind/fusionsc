#pragma once 

#include <cupnp/cupnp.h>
#include "offline.capnp.h"
#include "devices/w7x.capnp.cu.h"
#include "geometry.capnp.cu.h"
#include "magnetics.capnp.cu.h"

namespace fsc{
namespace cu{

struct OfflineData{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline OfflineData(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
struct W7XCoil{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline W7XCoil(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline uint64_t getId() const;
	inline void setId(uint64_t newVal);

	inline static const unsigned char FILAMENT_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline fsc::cu::Filament mutateFilament();

	inline const fsc::cu::Filament getFilament() const;
	inline bool nonDefaultFilament() const;

}; // struct fsc::cu::OfflineData::W7XCoil

struct W7XConfig{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline W7XConfig(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline uint64_t getId() const;
	inline void setId(uint64_t newVal);

	inline static const unsigned char CONFIG_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline fsc::devices::w7x::cu::CoilsDBConfig mutateConfig();

	inline const fsc::devices::w7x::cu::CoilsDBConfig getConfig() const;
	inline bool nonDefaultConfig() const;

}; // struct fsc::cu::OfflineData::W7XConfig

struct W7XComponent{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline W7XComponent(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline uint64_t getId() const;
	inline void setId(uint64_t newVal);

	inline static const unsigned char COMPONENT_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline fsc::cu::Mesh mutateComponent();

	inline const fsc::cu::Mesh getComponent() const;
	inline bool nonDefaultComponent() const;

}; // struct fsc::cu::OfflineData::W7XComponent

struct W7XAssembly{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline W7XAssembly(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline uint64_t getId() const;
	inline void setId(uint64_t newVal);

	inline static const unsigned char ASSEMBLY_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<uint64_t> mutateAssembly();

	inline const cupnp::List<uint64_t> getAssembly() const;
	inline bool nonDefaultAssembly() const;

}; // struct fsc::cu::OfflineData::W7XAssembly

	inline static const unsigned char W7X_COILS_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<fsc::cu::OfflineData::W7XCoil> mutateW7xCoils();

	inline const cupnp::List<fsc::cu::OfflineData::W7XCoil> getW7xCoils() const;
	inline bool nonDefaultW7xCoils() const;

	inline static const unsigned char W7X_CONFIGS_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<fsc::cu::OfflineData::W7XConfig> mutateW7xConfigs();

	inline const cupnp::List<fsc::cu::OfflineData::W7XConfig> getW7xConfigs() const;
	inline bool nonDefaultW7xConfigs() const;

	inline static const unsigned char W7X_COMPONENTS_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<fsc::cu::OfflineData::W7XComponent> mutateW7xComponents();

	inline const cupnp::List<fsc::cu::OfflineData::W7XComponent> getW7xComponents() const;
	inline bool nonDefaultW7xComponents() const;

	inline static const unsigned char W7X_ASSEMBLIES_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<fsc::cu::OfflineData::W7XAssembly> mutateW7xAssemblies();

	inline const cupnp::List<fsc::cu::OfflineData::W7XAssembly> getW7xAssemblies() const;
	inline bool nonDefaultW7xAssemblies() const;

}; // struct fsc::cu::OfflineData


}}
// ===== struct fsc::cu::OfflineData::W7XCoil =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::OfflineData::W7XCoil> { using Type = fsc::cu::OfflineData::W7XCoil; }; 
} // namespace ::cupnp

void fsc::cu::OfflineData::W7XCoil::setId(uint64_t newVal) { 
	cupnp::setPrimitiveField<uint64_t, 0>(structure, data, 0, newVal);
} 

uint64_t fsc::cu::OfflineData::W7XCoil::getId() const { 
	return cupnp::getPrimitiveField<uint64_t, 0>(structure, data, 0);
} 

fsc::cu::Filament fsc::cu::OfflineData::W7XCoil::mutateFilament() { 
	CUPNP_REQUIRE(nonDefaultFilament());
	return cupnp::mutatePointerField<fsc::cu::Filament, 0>(structure, data);
} 

bool fsc::cu::OfflineData::W7XCoil::nonDefaultFilament() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const fsc::cu::Filament fsc::cu::OfflineData::W7XCoil::getFilament() const { 
	return cupnp::getPointerField<fsc::cu::Filament, 0>(structure, data, reinterpret_cast<const capnp::word*>(FILAMENT_DEFAULT_VALUE));
} 

// ===== struct fsc::cu::OfflineData::W7XConfig =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::OfflineData::W7XConfig> { using Type = fsc::cu::OfflineData::W7XConfig; }; 
} // namespace ::cupnp

void fsc::cu::OfflineData::W7XConfig::setId(uint64_t newVal) { 
	cupnp::setPrimitiveField<uint64_t, 0>(structure, data, 0, newVal);
} 

uint64_t fsc::cu::OfflineData::W7XConfig::getId() const { 
	return cupnp::getPrimitiveField<uint64_t, 0>(structure, data, 0);
} 

fsc::devices::w7x::cu::CoilsDBConfig fsc::cu::OfflineData::W7XConfig::mutateConfig() { 
	CUPNP_REQUIRE(nonDefaultConfig());
	return cupnp::mutatePointerField<fsc::devices::w7x::cu::CoilsDBConfig, 0>(structure, data);
} 

bool fsc::cu::OfflineData::W7XConfig::nonDefaultConfig() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const fsc::devices::w7x::cu::CoilsDBConfig fsc::cu::OfflineData::W7XConfig::getConfig() const { 
	return cupnp::getPointerField<fsc::devices::w7x::cu::CoilsDBConfig, 0>(structure, data, reinterpret_cast<const capnp::word*>(CONFIG_DEFAULT_VALUE));
} 

// ===== struct fsc::cu::OfflineData::W7XComponent =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::OfflineData::W7XComponent> { using Type = fsc::cu::OfflineData::W7XComponent; }; 
} // namespace ::cupnp

void fsc::cu::OfflineData::W7XComponent::setId(uint64_t newVal) { 
	cupnp::setPrimitiveField<uint64_t, 0>(structure, data, 0, newVal);
} 

uint64_t fsc::cu::OfflineData::W7XComponent::getId() const { 
	return cupnp::getPrimitiveField<uint64_t, 0>(structure, data, 0);
} 

fsc::cu::Mesh fsc::cu::OfflineData::W7XComponent::mutateComponent() { 
	CUPNP_REQUIRE(nonDefaultComponent());
	return cupnp::mutatePointerField<fsc::cu::Mesh, 0>(structure, data);
} 

bool fsc::cu::OfflineData::W7XComponent::nonDefaultComponent() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const fsc::cu::Mesh fsc::cu::OfflineData::W7XComponent::getComponent() const { 
	return cupnp::getPointerField<fsc::cu::Mesh, 0>(structure, data, reinterpret_cast<const capnp::word*>(COMPONENT_DEFAULT_VALUE));
} 

// ===== struct fsc::cu::OfflineData::W7XAssembly =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::OfflineData::W7XAssembly> { using Type = fsc::cu::OfflineData::W7XAssembly; }; 
} // namespace ::cupnp

void fsc::cu::OfflineData::W7XAssembly::setId(uint64_t newVal) { 
	cupnp::setPrimitiveField<uint64_t, 0>(structure, data, 0, newVal);
} 

uint64_t fsc::cu::OfflineData::W7XAssembly::getId() const { 
	return cupnp::getPrimitiveField<uint64_t, 0>(structure, data, 0);
} 

cupnp::List<uint64_t> fsc::cu::OfflineData::W7XAssembly::mutateAssembly() { 
	CUPNP_REQUIRE(nonDefaultAssembly());
	return cupnp::mutatePointerField<cupnp::List<uint64_t>, 0>(structure, data);
} 

bool fsc::cu::OfflineData::W7XAssembly::nonDefaultAssembly() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const cupnp::List<uint64_t> fsc::cu::OfflineData::W7XAssembly::getAssembly() const { 
	return cupnp::getPointerField<cupnp::List<uint64_t>, 0>(structure, data, reinterpret_cast<const capnp::word*>(ASSEMBLY_DEFAULT_VALUE));
} 

// ===== struct fsc::cu::OfflineData =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::OfflineData> { using Type = fsc::cu::OfflineData; }; 
} // namespace ::cupnp

cupnp::List<fsc::cu::OfflineData::W7XCoil> fsc::cu::OfflineData::mutateW7xCoils() { 
	CUPNP_REQUIRE(nonDefaultW7xCoils());
	return cupnp::mutatePointerField<cupnp::List<fsc::cu::OfflineData::W7XCoil>, 0>(structure, data);
} 

bool fsc::cu::OfflineData::nonDefaultW7xCoils() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const cupnp::List<fsc::cu::OfflineData::W7XCoil> fsc::cu::OfflineData::getW7xCoils() const { 
	return cupnp::getPointerField<cupnp::List<fsc::cu::OfflineData::W7XCoil>, 0>(structure, data, reinterpret_cast<const capnp::word*>(W7X_COILS_DEFAULT_VALUE));
} 

cupnp::List<fsc::cu::OfflineData::W7XConfig> fsc::cu::OfflineData::mutateW7xConfigs() { 
	CUPNP_REQUIRE(nonDefaultW7xConfigs());
	return cupnp::mutatePointerField<cupnp::List<fsc::cu::OfflineData::W7XConfig>, 1>(structure, data);
} 

bool fsc::cu::OfflineData::nonDefaultW7xConfigs() const { 
	return cupnp::hasPointerField<1>(structure, data);
} 

const cupnp::List<fsc::cu::OfflineData::W7XConfig> fsc::cu::OfflineData::getW7xConfigs() const { 
	return cupnp::getPointerField<cupnp::List<fsc::cu::OfflineData::W7XConfig>, 1>(structure, data, reinterpret_cast<const capnp::word*>(W7X_CONFIGS_DEFAULT_VALUE));
} 

cupnp::List<fsc::cu::OfflineData::W7XComponent> fsc::cu::OfflineData::mutateW7xComponents() { 
	CUPNP_REQUIRE(nonDefaultW7xComponents());
	return cupnp::mutatePointerField<cupnp::List<fsc::cu::OfflineData::W7XComponent>, 2>(structure, data);
} 

bool fsc::cu::OfflineData::nonDefaultW7xComponents() const { 
	return cupnp::hasPointerField<2>(structure, data);
} 

const cupnp::List<fsc::cu::OfflineData::W7XComponent> fsc::cu::OfflineData::getW7xComponents() const { 
	return cupnp::getPointerField<cupnp::List<fsc::cu::OfflineData::W7XComponent>, 2>(structure, data, reinterpret_cast<const capnp::word*>(W7X_COMPONENTS_DEFAULT_VALUE));
} 

cupnp::List<fsc::cu::OfflineData::W7XAssembly> fsc::cu::OfflineData::mutateW7xAssemblies() { 
	CUPNP_REQUIRE(nonDefaultW7xAssemblies());
	return cupnp::mutatePointerField<cupnp::List<fsc::cu::OfflineData::W7XAssembly>, 3>(structure, data);
} 

bool fsc::cu::OfflineData::nonDefaultW7xAssemblies() const { 
	return cupnp::hasPointerField<3>(structure, data);
} 

const cupnp::List<fsc::cu::OfflineData::W7XAssembly> fsc::cu::OfflineData::getW7xAssemblies() const { 
	return cupnp::getPointerField<cupnp::List<fsc::cu::OfflineData::W7XAssembly>, 3>(structure, data, reinterpret_cast<const capnp::word*>(W7X_ASSEMBLIES_DEFAULT_VALUE));
} 

