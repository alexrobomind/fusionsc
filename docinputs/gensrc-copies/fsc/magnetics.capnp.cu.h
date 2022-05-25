#pragma once 

#include <cupnp/cupnp.h>
#include "magnetics.capnp.h"
#include "data.capnp.cu.h"

namespace fsc{
namespace cu{

struct ToroidalGrid{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline ToroidalGrid(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline double getRMin() const;
	inline void setRMin(double newVal);

	inline double getRMax() const;
	inline void setRMax(double newVal);

	inline double getZMin() const;
	inline void setZMin(double newVal);

	inline double getZMax() const;
	inline void setZMax(double newVal);

	inline uint32_t getNSym() const;
	inline void setNSym(uint32_t newVal);

	inline uint64_t getNR() const;
	inline void setNR(uint64_t newVal);

	inline uint64_t getNZ() const;
	inline void setNZ(uint64_t newVal);

	inline uint64_t getNPhi() const;
	inline void setNPhi(uint64_t newVal);

}; // struct fsc::cu::ToroidalGrid

struct ComputedField{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline ComputedField(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline static const unsigned char GRID_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline fsc::cu::ToroidalGrid mutateGrid();

	inline const fsc::cu::ToroidalGrid getGrid() const;
	inline bool nonDefaultGrid() const;

	inline static const unsigned char DATA_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline fsc::cu::DataRef<fsc::cu::Float64Tensor> mutateData();

	inline const fsc::cu::DataRef<fsc::cu::Float64Tensor> getData() const;
	inline bool nonDefaultData() const;

}; // struct fsc::cu::ComputedField

struct FieldResolver{
	static constexpr cupnp::Kind kind = cupnp::Kind::INTERFACE;
	
	// Interface pointer that holds the capability table offset
	uint64_t ptrData;
	
	FieldResolver(uint64_t structure, cupnp::Location data) : ptrData(structure) {
		cupnp::validateInterfacePointer(structure, data);
	}
	
	bool isDefault() { return ptrData == 0; }
	
};

struct FieldCalculator{
	static constexpr cupnp::Kind kind = cupnp::Kind::INTERFACE;
	
	// Interface pointer that holds the capability table offset
	uint64_t ptrData;
	
	FieldCalculator(uint64_t structure, cupnp::Location data) : ptrData(structure) {
		cupnp::validateInterfacePointer(structure, data);
	}
	
	bool isDefault() { return ptrData == 0; }
	
};

struct FieldCalculationSession{
	static constexpr cupnp::Kind kind = cupnp::Kind::INTERFACE;
	
	// Interface pointer that holds the capability table offset
	uint64_t ptrData;
	
	FieldCalculationSession(uint64_t structure, cupnp::Location data) : ptrData(structure) {
		cupnp::validateInterfacePointer(structure, data);
	}
	
	bool isDefault() { return ptrData == 0; }
	
};

struct BiotSavartSettings{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline BiotSavartSettings(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline double getWidth() const;
	inline void setWidth(double newVal);

	inline double getStepSize() const;
	inline void setStepSize(double newVal);

}; // struct fsc::cu::BiotSavartSettings

struct Filament{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline Filament(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline static const unsigned char INLINE_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline fsc::cu::Float64Tensor mutateInline();

	inline const fsc::cu::Float64Tensor getInline() const;
	inline bool nonDefaultInline() const;
	inline bool hasInline() const;

	inline static const unsigned char REF_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline fsc::cu::DataRef<fsc::cu::Filament> mutateRef();

	inline const fsc::cu::DataRef<fsc::cu::Filament> getRef() const;
	inline bool nonDefaultRef() const;
	inline bool hasRef() const;

	inline uint64_t getW7xCoilsDB() const;
	inline void setW7xCoilsDB(uint64_t newVal);

	inline bool hasW7xCoilsDB() const;
}; // struct fsc::cu::Filament

struct W7XCoilSet{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline W7XCoilSet(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
struct CoilsDBSet{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline CoilsDBSet(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline uint32_t getMainCoilOffset() const;
	inline void setMainCoilOffset(uint32_t newVal);

	inline static const unsigned char TRIM_COIL_I_DS_DEFAULT_VALUE [] = {1, 0, 0, 0, 44, 0, 0, 0, 94, 1, 0, 0, 241, 0, 0, 0, 95, 1, 0, 0, 96, 1, 0, 0, 97, 1, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<uint32_t> mutateTrimCoilIDs();

	inline const cupnp::List<uint32_t> getTrimCoilIDs() const;
	inline bool nonDefaultTrimCoilIDs() const;

	inline uint32_t getControlCoilOffset() const;
	inline void setControlCoilOffset(uint32_t newVal);

}; // struct fsc::cu::W7XCoilSet::CoilsDBSet

struct CustomCoilSet{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline CustomCoilSet(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline static const unsigned char MAIN_COILS_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<fsc::cu::DataRef<fsc::cu::Filament>> mutateMainCoils();

	inline const cupnp::List<fsc::cu::DataRef<fsc::cu::Filament>> getMainCoils() const;
	inline bool nonDefaultMainCoils() const;

	inline static const unsigned char TRIM_COILS_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<fsc::cu::DataRef<fsc::cu::Filament>> mutateTrimCoils();

	inline const cupnp::List<fsc::cu::DataRef<fsc::cu::Filament>> getTrimCoils() const;
	inline bool nonDefaultTrimCoils() const;

	inline static const unsigned char CONTROL_COILS_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<fsc::cu::DataRef<fsc::cu::Filament>> mutateControlCoils();

	inline const cupnp::List<fsc::cu::DataRef<fsc::cu::Filament>> getControlCoils() const;
	inline bool nonDefaultControlCoils() const;

}; // struct fsc::cu::W7XCoilSet::CustomCoilSet

	inline bool getInvertMainCoils() const;
	inline void setInvertMainCoils(bool newVal);

	inline static const unsigned char BIOT_SAVART_SETTINGS_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline fsc::cu::BiotSavartSettings mutateBiotSavartSettings();

	inline const fsc::cu::BiotSavartSettings getBiotSavartSettings() const;
	inline bool nonDefaultBiotSavartSettings() const;

	inline bool isCoilsDBSet() const;
	
	inline fsc::cu::W7XCoilSet::CoilsDBSet getCoilsDBSet() const;

	inline bool isCustomCoilSet() const;
	
	inline fsc::cu::W7XCoilSet::CustomCoilSet getCustomCoilSet() const;

	inline static const unsigned char N_WIND_MAIN_DEFAULT_VALUE [] = {1, 0, 0, 0, 60, 0, 0, 0, 108, 0, 0, 0, 108, 0, 0, 0, 108, 0, 0, 0, 108, 0, 0, 0, 108, 0, 0, 0, 36, 0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<uint32_t> mutateNWindMain();

	inline const cupnp::List<uint32_t> getNWindMain() const;
	inline bool nonDefaultNWindMain() const;

	inline static const unsigned char N_WIND_TRIM_DEFAULT_VALUE [] = {1, 0, 0, 0, 44, 0, 0, 0, 48, 0, 0, 0, 72, 0, 0, 0, 48, 0, 0, 0, 48, 0, 0, 0, 48, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<uint32_t> mutateNWindTrim();

	inline const cupnp::List<uint32_t> getNWindTrim() const;
	inline bool nonDefaultNWindTrim() const;

	inline static const unsigned char N_WIND_CONTROL_DEFAULT_VALUE [] = {1, 0, 0, 0, 84, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0};
	inline cupnp::List<uint32_t> mutateNWindControl();

	inline const cupnp::List<uint32_t> getNWindControl() const;
	inline bool nonDefaultNWindControl() const;

	inline static const unsigned char INVERT_CONTROL_COILS_DEFAULT_VALUE [] = {1, 0, 0, 0, 81, 0, 0, 0, 170, 2, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<bool> mutateInvertControlCoils();

	inline const cupnp::List<bool> getInvertControlCoils() const;
	inline bool nonDefaultInvertControlCoils() const;

}; // struct fsc::cu::W7XCoilSet

struct MagneticField{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline MagneticField(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
struct FilamentField{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline FilamentField(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline double getCurrent() const;
	inline void setCurrent(double newVal);

	inline static const unsigned char BIOT_SAVART_SETTINGS_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline fsc::cu::BiotSavartSettings mutateBiotSavartSettings();

	inline const fsc::cu::BiotSavartSettings getBiotSavartSettings() const;
	inline bool nonDefaultBiotSavartSettings() const;

	inline static const unsigned char FILAMENT_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline fsc::cu::Filament mutateFilament();

	inline const fsc::cu::Filament getFilament() const;
	inline bool nonDefaultFilament() const;

	inline uint32_t getWindingNo() const;
	inline void setWindingNo(uint32_t newVal);

}; // struct fsc::cu::MagneticField::FilamentField

struct ScaleBy{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline ScaleBy(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline static const unsigned char FIELD_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline fsc::cu::MagneticField mutateField();

	inline const fsc::cu::MagneticField getField() const;
	inline bool nonDefaultField() const;

	inline double getFactor() const;
	inline void setFactor(double newVal);

}; // struct fsc::cu::MagneticField::ScaleBy

struct W7xMagneticConfig{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline W7xMagneticConfig(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
struct ConfigurationDB{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline ConfigurationDB(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline static const unsigned char BIOT_SAVART_SETTINGS_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline fsc::cu::BiotSavartSettings mutateBiotSavartSettings();

	inline const fsc::cu::BiotSavartSettings getBiotSavartSettings() const;
	inline bool nonDefaultBiotSavartSettings() const;

	inline uint64_t getConfigID() const;
	inline void setConfigID(uint64_t newVal);

}; // struct fsc::cu::MagneticField::W7xMagneticConfig::ConfigurationDB

struct CoilsAndCurrents{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline CoilsAndCurrents(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline static const unsigned char NONPLANAR_DEFAULT_VALUE [] = {1, 0, 0, 0, 45, 0, 0, 0, 0, 0, 0, 0, 0, 136, 195, 64, 0, 0, 0, 0, 0, 136, 195, 64, 0, 0, 0, 0, 0, 136, 195, 64, 0, 0, 0, 0, 0, 136, 195, 64, 0, 0, 0, 0, 0, 136, 195, 64};
	inline cupnp::List<double> mutateNonplanar();

	inline const cupnp::List<double> getNonplanar() const;
	inline bool nonDefaultNonplanar() const;

	inline static const unsigned char PLANAR_DEFAULT_VALUE [] = {1, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<double> mutatePlanar();

	inline const cupnp::List<double> getPlanar() const;
	inline bool nonDefaultPlanar() const;

	inline static const unsigned char TRIM_DEFAULT_VALUE [] = {1, 0, 0, 0, 45, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<double> mutateTrim();

	inline const cupnp::List<double> getTrim() const;
	inline bool nonDefaultTrim() const;

	inline static const unsigned char CONTROL_DEFAULT_VALUE [] = {1, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<double> mutateControl();

	inline const cupnp::List<double> getControl() const;
	inline bool nonDefaultControl() const;

	inline static const unsigned char COILS_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline fsc::cu::W7XCoilSet mutateCoils();

	inline const fsc::cu::W7XCoilSet getCoils() const;
	inline bool nonDefaultCoils() const;

}; // struct fsc::cu::MagneticField::W7xMagneticConfig::CoilsAndCurrents

	inline bool isConfigurationDB() const;
	
	inline fsc::cu::MagneticField::W7xMagneticConfig::ConfigurationDB getConfigurationDB() const;

	inline bool isCoilsAndCurrents() const;
	
	inline fsc::cu::MagneticField::W7xMagneticConfig::CoilsAndCurrents getCoilsAndCurrents() const;

}; // struct fsc::cu::MagneticField::W7xMagneticConfig

	inline static const unsigned char SUM_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<fsc::cu::MagneticField> mutateSum();

	inline const cupnp::List<fsc::cu::MagneticField> getSum() const;
	inline bool nonDefaultSum() const;
	inline bool hasSum() const;

	inline static const unsigned char REF_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline fsc::cu::DataRef<fsc::cu::MagneticField> mutateRef();

	inline const fsc::cu::DataRef<fsc::cu::MagneticField> getRef() const;
	inline bool nonDefaultRef() const;
	inline bool hasRef() const;

	inline static const unsigned char COMPUTED_FIELD_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline fsc::cu::ComputedField mutateComputedField();

	inline const fsc::cu::ComputedField getComputedField() const;
	inline bool nonDefaultComputedField() const;
	inline bool hasComputedField() const;

	inline bool isFilamentField() const;
	
	inline fsc::cu::MagneticField::FilamentField getFilamentField() const;

	inline bool isScaleBy() const;
	
	inline fsc::cu::MagneticField::ScaleBy getScaleBy() const;

	inline static const unsigned char INVERT_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline fsc::cu::MagneticField mutateInvert();

	inline const fsc::cu::MagneticField getInvert() const;
	inline bool nonDefaultInvert() const;
	inline bool hasInvert() const;

	inline bool isW7xMagneticConfig() const;
	
	inline fsc::cu::MagneticField::W7xMagneticConfig getW7xMagneticConfig() const;

}; // struct fsc::cu::MagneticField


}}
// ===== struct fsc::cu::ToroidalGrid =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::ToroidalGrid> { using Type = fsc::cu::ToroidalGrid; }; 
} // namespace ::cupnp

void fsc::cu::ToroidalGrid::setRMin(double newVal) { 
	cupnp::setPrimitiveField<double, 0>(structure, data, 0, newVal);
} 

double fsc::cu::ToroidalGrid::getRMin() const { 
	return cupnp::getPrimitiveField<double, 0>(structure, data, 0);
} 

void fsc::cu::ToroidalGrid::setRMax(double newVal) { 
	cupnp::setPrimitiveField<double, 1>(structure, data, 0, newVal);
} 

double fsc::cu::ToroidalGrid::getRMax() const { 
	return cupnp::getPrimitiveField<double, 1>(structure, data, 0);
} 

void fsc::cu::ToroidalGrid::setZMin(double newVal) { 
	cupnp::setPrimitiveField<double, 2>(structure, data, 0, newVal);
} 

double fsc::cu::ToroidalGrid::getZMin() const { 
	return cupnp::getPrimitiveField<double, 2>(structure, data, 0);
} 

void fsc::cu::ToroidalGrid::setZMax(double newVal) { 
	cupnp::setPrimitiveField<double, 3>(structure, data, 0, newVal);
} 

double fsc::cu::ToroidalGrid::getZMax() const { 
	return cupnp::getPrimitiveField<double, 3>(structure, data, 0);
} 

void fsc::cu::ToroidalGrid::setNSym(uint32_t newVal) { 
	cupnp::setPrimitiveField<uint32_t, 8>(structure, data, 1, newVal);
} 

uint32_t fsc::cu::ToroidalGrid::getNSym() const { 
	return cupnp::getPrimitiveField<uint32_t, 8>(structure, data, 1);
} 

void fsc::cu::ToroidalGrid::setNR(uint64_t newVal) { 
	cupnp::setPrimitiveField<uint64_t, 5>(structure, data, 0, newVal);
} 

uint64_t fsc::cu::ToroidalGrid::getNR() const { 
	return cupnp::getPrimitiveField<uint64_t, 5>(structure, data, 0);
} 

void fsc::cu::ToroidalGrid::setNZ(uint64_t newVal) { 
	cupnp::setPrimitiveField<uint64_t, 6>(structure, data, 0, newVal);
} 

uint64_t fsc::cu::ToroidalGrid::getNZ() const { 
	return cupnp::getPrimitiveField<uint64_t, 6>(structure, data, 0);
} 

void fsc::cu::ToroidalGrid::setNPhi(uint64_t newVal) { 
	cupnp::setPrimitiveField<uint64_t, 7>(structure, data, 0, newVal);
} 

uint64_t fsc::cu::ToroidalGrid::getNPhi() const { 
	return cupnp::getPrimitiveField<uint64_t, 7>(structure, data, 0);
} 

// ===== struct fsc::cu::ComputedField =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::ComputedField> { using Type = fsc::cu::ComputedField; }; 
} // namespace ::cupnp

fsc::cu::ToroidalGrid fsc::cu::ComputedField::mutateGrid() { 
	CUPNP_REQUIRE(nonDefaultGrid());
	return cupnp::mutatePointerField<fsc::cu::ToroidalGrid, 0>(structure, data);
} 

bool fsc::cu::ComputedField::nonDefaultGrid() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const fsc::cu::ToroidalGrid fsc::cu::ComputedField::getGrid() const { 
	return cupnp::getPointerField<fsc::cu::ToroidalGrid, 0>(structure, data, reinterpret_cast<const capnp::word*>(GRID_DEFAULT_VALUE));
} 

fsc::cu::DataRef<fsc::cu::Float64Tensor> fsc::cu::ComputedField::mutateData() { 
	CUPNP_REQUIRE(nonDefaultData());
	return cupnp::mutatePointerField<fsc::cu::DataRef<fsc::cu::Float64Tensor>, 1>(structure, data);
} 

bool fsc::cu::ComputedField::nonDefaultData() const { 
	return cupnp::hasPointerField<1>(structure, data);
} 

const fsc::cu::DataRef<fsc::cu::Float64Tensor> fsc::cu::ComputedField::getData() const { 
	return cupnp::getPointerField<fsc::cu::DataRef<fsc::cu::Float64Tensor>, 1>(structure, data, reinterpret_cast<const capnp::word*>(DATA_DEFAULT_VALUE));
} 

// ===== struct fsc::cu::BiotSavartSettings =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::BiotSavartSettings> { using Type = fsc::cu::BiotSavartSettings; }; 
} // namespace ::cupnp

void fsc::cu::BiotSavartSettings::setWidth(double newVal) { 
	cupnp::setPrimitiveField<double, 0>(structure, data, 0, newVal);
} 

double fsc::cu::BiotSavartSettings::getWidth() const { 
	return cupnp::getPrimitiveField<double, 0>(structure, data, 0);
} 

void fsc::cu::BiotSavartSettings::setStepSize(double newVal) { 
	cupnp::setPrimitiveField<double, 1>(structure, data, 0, newVal);
} 

double fsc::cu::BiotSavartSettings::getStepSize() const { 
	return cupnp::getPrimitiveField<double, 1>(structure, data, 0);
} 

// ===== struct fsc::cu::Filament =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::Filament> { using Type = fsc::cu::Filament; }; 
} // namespace ::cupnp

fsc::cu::Float64Tensor fsc::cu::Filament::mutateInline() { 
	CUPNP_REQUIRE(nonDefaultInline());
	return cupnp::mutatePointerField<fsc::cu::Float64Tensor, 0>(structure, data);
} 

bool fsc::cu::Filament::hasInline() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 0;
} 

bool fsc::cu::Filament::nonDefaultInline() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 0 && cupnp::hasPointerField<0>(structure, data);
} 

const fsc::cu::Float64Tensor fsc::cu::Filament::getInline() const { 
	if(cupnp::getDiscriminant<0>(structure, data) != 0)
		return cupnp::getPointer<fsc::cu::Float64Tensor>(reinterpret_cast<const capnp::word*>(INLINE_DEFAULT_VALUE));
	
	return cupnp::getPointerField<fsc::cu::Float64Tensor, 0>(structure, data, reinterpret_cast<const capnp::word*>(INLINE_DEFAULT_VALUE));
} 

fsc::cu::DataRef<fsc::cu::Filament> fsc::cu::Filament::mutateRef() { 
	CUPNP_REQUIRE(nonDefaultRef());
	return cupnp::mutatePointerField<fsc::cu::DataRef<fsc::cu::Filament>, 0>(structure, data);
} 

bool fsc::cu::Filament::hasRef() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 1;
} 

bool fsc::cu::Filament::nonDefaultRef() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 1 && cupnp::hasPointerField<0>(structure, data);
} 

const fsc::cu::DataRef<fsc::cu::Filament> fsc::cu::Filament::getRef() const { 
	if(cupnp::getDiscriminant<0>(structure, data) != 1)
		return cupnp::getPointer<fsc::cu::DataRef<fsc::cu::Filament>>(reinterpret_cast<const capnp::word*>(REF_DEFAULT_VALUE));
	
	return cupnp::getPointerField<fsc::cu::DataRef<fsc::cu::Filament>, 0>(structure, data, reinterpret_cast<const capnp::word*>(REF_DEFAULT_VALUE));
} 

void fsc::cu::Filament::setW7xCoilsDB(uint64_t newVal) { 
	cupnp::setPrimitiveField<uint64_t, 1>(structure, data, 0, newVal);
	cupnp::setDiscriminant<0>(structure, data, 2);
} 

uint64_t fsc::cu::Filament::getW7xCoilsDB() const { 
	if(cupnp::getDiscriminant<0>(structure, data) != 2)
		return 0;
	
	return cupnp::getPrimitiveField<uint64_t, 1>(structure, data, 0);
} 

bool fsc::cu::Filament::hasW7xCoilsDB() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 2;
} 

// ===== struct fsc::cu::W7XCoilSet::CoilsDBSet =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::W7XCoilSet::CoilsDBSet> { using Type = fsc::cu::W7XCoilSet::CoilsDBSet; }; 
} // namespace ::cupnp

void fsc::cu::W7XCoilSet::CoilsDBSet::setMainCoilOffset(uint32_t newVal) { 
	cupnp::setPrimitiveField<uint32_t, 1>(structure, data, 160, newVal);
} 

uint32_t fsc::cu::W7XCoilSet::CoilsDBSet::getMainCoilOffset() const { 
	return cupnp::getPrimitiveField<uint32_t, 1>(structure, data, 160);
} 

cupnp::List<uint32_t> fsc::cu::W7XCoilSet::CoilsDBSet::mutateTrimCoilIDs() { 
	CUPNP_REQUIRE(nonDefaultTrimCoilIDs());
	return cupnp::mutatePointerField<cupnp::List<uint32_t>, 1>(structure, data);
} 

bool fsc::cu::W7XCoilSet::CoilsDBSet::nonDefaultTrimCoilIDs() const { 
	return cupnp::hasPointerField<1>(structure, data);
} 

const cupnp::List<uint32_t> fsc::cu::W7XCoilSet::CoilsDBSet::getTrimCoilIDs() const { 
	return cupnp::getPointerField<cupnp::List<uint32_t>, 1>(structure, data, reinterpret_cast<const capnp::word*>(TRIM_COIL_I_DS_DEFAULT_VALUE));
} 

void fsc::cu::W7XCoilSet::CoilsDBSet::setControlCoilOffset(uint32_t newVal) { 
	cupnp::setPrimitiveField<uint32_t, 2>(structure, data, 230, newVal);
} 

uint32_t fsc::cu::W7XCoilSet::CoilsDBSet::getControlCoilOffset() const { 
	return cupnp::getPrimitiveField<uint32_t, 2>(structure, data, 230);
} 

// ===== struct fsc::cu::W7XCoilSet::CustomCoilSet =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::W7XCoilSet::CustomCoilSet> { using Type = fsc::cu::W7XCoilSet::CustomCoilSet; }; 
} // namespace ::cupnp

cupnp::List<fsc::cu::DataRef<fsc::cu::Filament>> fsc::cu::W7XCoilSet::CustomCoilSet::mutateMainCoils() { 
	CUPNP_REQUIRE(nonDefaultMainCoils());
	return cupnp::mutatePointerField<cupnp::List<fsc::cu::DataRef<fsc::cu::Filament>>, 1>(structure, data);
} 

bool fsc::cu::W7XCoilSet::CustomCoilSet::nonDefaultMainCoils() const { 
	return cupnp::hasPointerField<1>(structure, data);
} 

const cupnp::List<fsc::cu::DataRef<fsc::cu::Filament>> fsc::cu::W7XCoilSet::CustomCoilSet::getMainCoils() const { 
	return cupnp::getPointerField<cupnp::List<fsc::cu::DataRef<fsc::cu::Filament>>, 1>(structure, data, reinterpret_cast<const capnp::word*>(MAIN_COILS_DEFAULT_VALUE));
} 

cupnp::List<fsc::cu::DataRef<fsc::cu::Filament>> fsc::cu::W7XCoilSet::CustomCoilSet::mutateTrimCoils() { 
	CUPNP_REQUIRE(nonDefaultTrimCoils());
	return cupnp::mutatePointerField<cupnp::List<fsc::cu::DataRef<fsc::cu::Filament>>, 2>(structure, data);
} 

bool fsc::cu::W7XCoilSet::CustomCoilSet::nonDefaultTrimCoils() const { 
	return cupnp::hasPointerField<2>(structure, data);
} 

const cupnp::List<fsc::cu::DataRef<fsc::cu::Filament>> fsc::cu::W7XCoilSet::CustomCoilSet::getTrimCoils() const { 
	return cupnp::getPointerField<cupnp::List<fsc::cu::DataRef<fsc::cu::Filament>>, 2>(structure, data, reinterpret_cast<const capnp::word*>(TRIM_COILS_DEFAULT_VALUE));
} 

cupnp::List<fsc::cu::DataRef<fsc::cu::Filament>> fsc::cu::W7XCoilSet::CustomCoilSet::mutateControlCoils() { 
	CUPNP_REQUIRE(nonDefaultControlCoils());
	return cupnp::mutatePointerField<cupnp::List<fsc::cu::DataRef<fsc::cu::Filament>>, 3>(structure, data);
} 

bool fsc::cu::W7XCoilSet::CustomCoilSet::nonDefaultControlCoils() const { 
	return cupnp::hasPointerField<3>(structure, data);
} 

const cupnp::List<fsc::cu::DataRef<fsc::cu::Filament>> fsc::cu::W7XCoilSet::CustomCoilSet::getControlCoils() const { 
	return cupnp::getPointerField<cupnp::List<fsc::cu::DataRef<fsc::cu::Filament>>, 3>(structure, data, reinterpret_cast<const capnp::word*>(CONTROL_COILS_DEFAULT_VALUE));
} 

// ===== struct fsc::cu::W7XCoilSet =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::W7XCoilSet> { using Type = fsc::cu::W7XCoilSet; }; 
} // namespace ::cupnp

void fsc::cu::W7XCoilSet::setInvertMainCoils(bool newVal) { 
	cupnp::setPrimitiveField<bool, 0>(structure, data, true, newVal);
} 

bool fsc::cu::W7XCoilSet::getInvertMainCoils() const { 
	return cupnp::getPrimitiveField<bool, 0>(structure, data, true);
} 

fsc::cu::BiotSavartSettings fsc::cu::W7XCoilSet::mutateBiotSavartSettings() { 
	CUPNP_REQUIRE(nonDefaultBiotSavartSettings());
	return cupnp::mutatePointerField<fsc::cu::BiotSavartSettings, 0>(structure, data);
} 

bool fsc::cu::W7XCoilSet::nonDefaultBiotSavartSettings() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const fsc::cu::BiotSavartSettings fsc::cu::W7XCoilSet::getBiotSavartSettings() const { 
	return cupnp::getPointerField<fsc::cu::BiotSavartSettings, 0>(structure, data, reinterpret_cast<const capnp::word*>(BIOT_SAVART_SETTINGS_DEFAULT_VALUE));
} 

bool fsc::cu::W7XCoilSet::isCoilsDBSet() const { 
	return cupnp::getDiscriminant<1>(structure, data) == 0;
} 

fsc::cu::W7XCoilSet::CoilsDBSet fsc::cu::W7XCoilSet::getCoilsDBSet() const { 
	return fsc::cu::W7XCoilSet::CoilsDBSet(structure, data);
} 

bool fsc::cu::W7XCoilSet::isCustomCoilSet() const { 
	return cupnp::getDiscriminant<1>(structure, data) == 1;
} 

fsc::cu::W7XCoilSet::CustomCoilSet fsc::cu::W7XCoilSet::getCustomCoilSet() const { 
	return fsc::cu::W7XCoilSet::CustomCoilSet(structure, data);
} 

cupnp::List<uint32_t> fsc::cu::W7XCoilSet::mutateNWindMain() { 
	CUPNP_REQUIRE(nonDefaultNWindMain());
	return cupnp::mutatePointerField<cupnp::List<uint32_t>, 4>(structure, data);
} 

bool fsc::cu::W7XCoilSet::nonDefaultNWindMain() const { 
	return cupnp::hasPointerField<4>(structure, data);
} 

const cupnp::List<uint32_t> fsc::cu::W7XCoilSet::getNWindMain() const { 
	return cupnp::getPointerField<cupnp::List<uint32_t>, 4>(structure, data, reinterpret_cast<const capnp::word*>(N_WIND_MAIN_DEFAULT_VALUE));
} 

cupnp::List<uint32_t> fsc::cu::W7XCoilSet::mutateNWindTrim() { 
	CUPNP_REQUIRE(nonDefaultNWindTrim());
	return cupnp::mutatePointerField<cupnp::List<uint32_t>, 5>(structure, data);
} 

bool fsc::cu::W7XCoilSet::nonDefaultNWindTrim() const { 
	return cupnp::hasPointerField<5>(structure, data);
} 

const cupnp::List<uint32_t> fsc::cu::W7XCoilSet::getNWindTrim() const { 
	return cupnp::getPointerField<cupnp::List<uint32_t>, 5>(structure, data, reinterpret_cast<const capnp::word*>(N_WIND_TRIM_DEFAULT_VALUE));
} 

cupnp::List<uint32_t> fsc::cu::W7XCoilSet::mutateNWindControl() { 
	CUPNP_REQUIRE(nonDefaultNWindControl());
	return cupnp::mutatePointerField<cupnp::List<uint32_t>, 6>(structure, data);
} 

bool fsc::cu::W7XCoilSet::nonDefaultNWindControl() const { 
	return cupnp::hasPointerField<6>(structure, data);
} 

const cupnp::List<uint32_t> fsc::cu::W7XCoilSet::getNWindControl() const { 
	return cupnp::getPointerField<cupnp::List<uint32_t>, 6>(structure, data, reinterpret_cast<const capnp::word*>(N_WIND_CONTROL_DEFAULT_VALUE));
} 

cupnp::List<bool> fsc::cu::W7XCoilSet::mutateInvertControlCoils() { 
	CUPNP_REQUIRE(nonDefaultInvertControlCoils());
	return cupnp::mutatePointerField<cupnp::List<bool>, 7>(structure, data);
} 

bool fsc::cu::W7XCoilSet::nonDefaultInvertControlCoils() const { 
	return cupnp::hasPointerField<7>(structure, data);
} 

const cupnp::List<bool> fsc::cu::W7XCoilSet::getInvertControlCoils() const { 
	return cupnp::getPointerField<cupnp::List<bool>, 7>(structure, data, reinterpret_cast<const capnp::word*>(INVERT_CONTROL_COILS_DEFAULT_VALUE));
} 

// ===== struct fsc::cu::MagneticField::FilamentField =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::MagneticField::FilamentField> { using Type = fsc::cu::MagneticField::FilamentField; }; 
} // namespace ::cupnp

void fsc::cu::MagneticField::FilamentField::setCurrent(double newVal) { 
	cupnp::setPrimitiveField<double, 1>(structure, data, 0, newVal);
} 

double fsc::cu::MagneticField::FilamentField::getCurrent() const { 
	return cupnp::getPrimitiveField<double, 1>(structure, data, 0);
} 

fsc::cu::BiotSavartSettings fsc::cu::MagneticField::FilamentField::mutateBiotSavartSettings() { 
	CUPNP_REQUIRE(nonDefaultBiotSavartSettings());
	return cupnp::mutatePointerField<fsc::cu::BiotSavartSettings, 0>(structure, data);
} 

bool fsc::cu::MagneticField::FilamentField::nonDefaultBiotSavartSettings() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const fsc::cu::BiotSavartSettings fsc::cu::MagneticField::FilamentField::getBiotSavartSettings() const { 
	return cupnp::getPointerField<fsc::cu::BiotSavartSettings, 0>(structure, data, reinterpret_cast<const capnp::word*>(BIOT_SAVART_SETTINGS_DEFAULT_VALUE));
} 

fsc::cu::Filament fsc::cu::MagneticField::FilamentField::mutateFilament() { 
	CUPNP_REQUIRE(nonDefaultFilament());
	return cupnp::mutatePointerField<fsc::cu::Filament, 1>(structure, data);
} 

bool fsc::cu::MagneticField::FilamentField::nonDefaultFilament() const { 
	return cupnp::hasPointerField<1>(structure, data);
} 

const fsc::cu::Filament fsc::cu::MagneticField::FilamentField::getFilament() const { 
	return cupnp::getPointerField<fsc::cu::Filament, 1>(structure, data, reinterpret_cast<const capnp::word*>(FILAMENT_DEFAULT_VALUE));
} 

void fsc::cu::MagneticField::FilamentField::setWindingNo(uint32_t newVal) { 
	cupnp::setPrimitiveField<uint32_t, 1>(structure, data, 1, newVal);
} 

uint32_t fsc::cu::MagneticField::FilamentField::getWindingNo() const { 
	return cupnp::getPrimitiveField<uint32_t, 1>(structure, data, 1);
} 

// ===== struct fsc::cu::MagneticField::ScaleBy =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::MagneticField::ScaleBy> { using Type = fsc::cu::MagneticField::ScaleBy; }; 
} // namespace ::cupnp

fsc::cu::MagneticField fsc::cu::MagneticField::ScaleBy::mutateField() { 
	CUPNP_REQUIRE(nonDefaultField());
	return cupnp::mutatePointerField<fsc::cu::MagneticField, 0>(structure, data);
} 

bool fsc::cu::MagneticField::ScaleBy::nonDefaultField() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const fsc::cu::MagneticField fsc::cu::MagneticField::ScaleBy::getField() const { 
	return cupnp::getPointerField<fsc::cu::MagneticField, 0>(structure, data, reinterpret_cast<const capnp::word*>(FIELD_DEFAULT_VALUE));
} 

void fsc::cu::MagneticField::ScaleBy::setFactor(double newVal) { 
	cupnp::setPrimitiveField<double, 1>(structure, data, 0, newVal);
} 

double fsc::cu::MagneticField::ScaleBy::getFactor() const { 
	return cupnp::getPrimitiveField<double, 1>(structure, data, 0);
} 

// ===== struct fsc::cu::MagneticField::W7xMagneticConfig::ConfigurationDB =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::MagneticField::W7xMagneticConfig::ConfigurationDB> { using Type = fsc::cu::MagneticField::W7xMagneticConfig::ConfigurationDB; }; 
} // namespace ::cupnp

fsc::cu::BiotSavartSettings fsc::cu::MagneticField::W7xMagneticConfig::ConfigurationDB::mutateBiotSavartSettings() { 
	CUPNP_REQUIRE(nonDefaultBiotSavartSettings());
	return cupnp::mutatePointerField<fsc::cu::BiotSavartSettings, 0>(structure, data);
} 

bool fsc::cu::MagneticField::W7xMagneticConfig::ConfigurationDB::nonDefaultBiotSavartSettings() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const fsc::cu::BiotSavartSettings fsc::cu::MagneticField::W7xMagneticConfig::ConfigurationDB::getBiotSavartSettings() const { 
	return cupnp::getPointerField<fsc::cu::BiotSavartSettings, 0>(structure, data, reinterpret_cast<const capnp::word*>(BIOT_SAVART_SETTINGS_DEFAULT_VALUE));
} 

void fsc::cu::MagneticField::W7xMagneticConfig::ConfigurationDB::setConfigID(uint64_t newVal) { 
	cupnp::setPrimitiveField<uint64_t, 1>(structure, data, 0, newVal);
} 

uint64_t fsc::cu::MagneticField::W7xMagneticConfig::ConfigurationDB::getConfigID() const { 
	return cupnp::getPrimitiveField<uint64_t, 1>(structure, data, 0);
} 

// ===== struct fsc::cu::MagneticField::W7xMagneticConfig::CoilsAndCurrents =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::MagneticField::W7xMagneticConfig::CoilsAndCurrents> { using Type = fsc::cu::MagneticField::W7xMagneticConfig::CoilsAndCurrents; }; 
} // namespace ::cupnp

cupnp::List<double> fsc::cu::MagneticField::W7xMagneticConfig::CoilsAndCurrents::mutateNonplanar() { 
	CUPNP_REQUIRE(nonDefaultNonplanar());
	return cupnp::mutatePointerField<cupnp::List<double>, 0>(structure, data);
} 

bool fsc::cu::MagneticField::W7xMagneticConfig::CoilsAndCurrents::nonDefaultNonplanar() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const cupnp::List<double> fsc::cu::MagneticField::W7xMagneticConfig::CoilsAndCurrents::getNonplanar() const { 
	return cupnp::getPointerField<cupnp::List<double>, 0>(structure, data, reinterpret_cast<const capnp::word*>(NONPLANAR_DEFAULT_VALUE));
} 

cupnp::List<double> fsc::cu::MagneticField::W7xMagneticConfig::CoilsAndCurrents::mutatePlanar() { 
	CUPNP_REQUIRE(nonDefaultPlanar());
	return cupnp::mutatePointerField<cupnp::List<double>, 1>(structure, data);
} 

bool fsc::cu::MagneticField::W7xMagneticConfig::CoilsAndCurrents::nonDefaultPlanar() const { 
	return cupnp::hasPointerField<1>(structure, data);
} 

const cupnp::List<double> fsc::cu::MagneticField::W7xMagneticConfig::CoilsAndCurrents::getPlanar() const { 
	return cupnp::getPointerField<cupnp::List<double>, 1>(structure, data, reinterpret_cast<const capnp::word*>(PLANAR_DEFAULT_VALUE));
} 

cupnp::List<double> fsc::cu::MagneticField::W7xMagneticConfig::CoilsAndCurrents::mutateTrim() { 
	CUPNP_REQUIRE(nonDefaultTrim());
	return cupnp::mutatePointerField<cupnp::List<double>, 2>(structure, data);
} 

bool fsc::cu::MagneticField::W7xMagneticConfig::CoilsAndCurrents::nonDefaultTrim() const { 
	return cupnp::hasPointerField<2>(structure, data);
} 

const cupnp::List<double> fsc::cu::MagneticField::W7xMagneticConfig::CoilsAndCurrents::getTrim() const { 
	return cupnp::getPointerField<cupnp::List<double>, 2>(structure, data, reinterpret_cast<const capnp::word*>(TRIM_DEFAULT_VALUE));
} 

cupnp::List<double> fsc::cu::MagneticField::W7xMagneticConfig::CoilsAndCurrents::mutateControl() { 
	CUPNP_REQUIRE(nonDefaultControl());
	return cupnp::mutatePointerField<cupnp::List<double>, 3>(structure, data);
} 

bool fsc::cu::MagneticField::W7xMagneticConfig::CoilsAndCurrents::nonDefaultControl() const { 
	return cupnp::hasPointerField<3>(structure, data);
} 

const cupnp::List<double> fsc::cu::MagneticField::W7xMagneticConfig::CoilsAndCurrents::getControl() const { 
	return cupnp::getPointerField<cupnp::List<double>, 3>(structure, data, reinterpret_cast<const capnp::word*>(CONTROL_DEFAULT_VALUE));
} 

fsc::cu::W7XCoilSet fsc::cu::MagneticField::W7xMagneticConfig::CoilsAndCurrents::mutateCoils() { 
	CUPNP_REQUIRE(nonDefaultCoils());
	return cupnp::mutatePointerField<fsc::cu::W7XCoilSet, 4>(structure, data);
} 

bool fsc::cu::MagneticField::W7xMagneticConfig::CoilsAndCurrents::nonDefaultCoils() const { 
	return cupnp::hasPointerField<4>(structure, data);
} 

const fsc::cu::W7XCoilSet fsc::cu::MagneticField::W7xMagneticConfig::CoilsAndCurrents::getCoils() const { 
	return cupnp::getPointerField<fsc::cu::W7XCoilSet, 4>(structure, data, reinterpret_cast<const capnp::word*>(COILS_DEFAULT_VALUE));
} 

// ===== struct fsc::cu::MagneticField::W7xMagneticConfig =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::MagneticField::W7xMagneticConfig> { using Type = fsc::cu::MagneticField::W7xMagneticConfig; }; 
} // namespace ::cupnp

bool fsc::cu::MagneticField::W7xMagneticConfig::isConfigurationDB() const { 
	return cupnp::getDiscriminant<2>(structure, data) == 0;
} 

fsc::cu::MagneticField::W7xMagneticConfig::ConfigurationDB fsc::cu::MagneticField::W7xMagneticConfig::getConfigurationDB() const { 
	return fsc::cu::MagneticField::W7xMagneticConfig::ConfigurationDB(structure, data);
} 

bool fsc::cu::MagneticField::W7xMagneticConfig::isCoilsAndCurrents() const { 
	return cupnp::getDiscriminant<2>(structure, data) == 1;
} 

fsc::cu::MagneticField::W7xMagneticConfig::CoilsAndCurrents fsc::cu::MagneticField::W7xMagneticConfig::getCoilsAndCurrents() const { 
	return fsc::cu::MagneticField::W7xMagneticConfig::CoilsAndCurrents(structure, data);
} 

// ===== struct fsc::cu::MagneticField =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::MagneticField> { using Type = fsc::cu::MagneticField; }; 
} // namespace ::cupnp

cupnp::List<fsc::cu::MagneticField> fsc::cu::MagneticField::mutateSum() { 
	CUPNP_REQUIRE(nonDefaultSum());
	return cupnp::mutatePointerField<cupnp::List<fsc::cu::MagneticField>, 0>(structure, data);
} 

bool fsc::cu::MagneticField::hasSum() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 0;
} 

bool fsc::cu::MagneticField::nonDefaultSum() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 0 && cupnp::hasPointerField<0>(structure, data);
} 

const cupnp::List<fsc::cu::MagneticField> fsc::cu::MagneticField::getSum() const { 
	if(cupnp::getDiscriminant<0>(structure, data) != 0)
		return cupnp::getPointer<cupnp::List<fsc::cu::MagneticField>>(reinterpret_cast<const capnp::word*>(SUM_DEFAULT_VALUE));
	
	return cupnp::getPointerField<cupnp::List<fsc::cu::MagneticField>, 0>(structure, data, reinterpret_cast<const capnp::word*>(SUM_DEFAULT_VALUE));
} 

fsc::cu::DataRef<fsc::cu::MagneticField> fsc::cu::MagneticField::mutateRef() { 
	CUPNP_REQUIRE(nonDefaultRef());
	return cupnp::mutatePointerField<fsc::cu::DataRef<fsc::cu::MagneticField>, 0>(structure, data);
} 

bool fsc::cu::MagneticField::hasRef() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 1;
} 

bool fsc::cu::MagneticField::nonDefaultRef() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 1 && cupnp::hasPointerField<0>(structure, data);
} 

const fsc::cu::DataRef<fsc::cu::MagneticField> fsc::cu::MagneticField::getRef() const { 
	if(cupnp::getDiscriminant<0>(structure, data) != 1)
		return cupnp::getPointer<fsc::cu::DataRef<fsc::cu::MagneticField>>(reinterpret_cast<const capnp::word*>(REF_DEFAULT_VALUE));
	
	return cupnp::getPointerField<fsc::cu::DataRef<fsc::cu::MagneticField>, 0>(structure, data, reinterpret_cast<const capnp::word*>(REF_DEFAULT_VALUE));
} 

fsc::cu::ComputedField fsc::cu::MagneticField::mutateComputedField() { 
	CUPNP_REQUIRE(nonDefaultComputedField());
	return cupnp::mutatePointerField<fsc::cu::ComputedField, 0>(structure, data);
} 

bool fsc::cu::MagneticField::hasComputedField() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 2;
} 

bool fsc::cu::MagneticField::nonDefaultComputedField() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 2 && cupnp::hasPointerField<0>(structure, data);
} 

const fsc::cu::ComputedField fsc::cu::MagneticField::getComputedField() const { 
	if(cupnp::getDiscriminant<0>(structure, data) != 2)
		return cupnp::getPointer<fsc::cu::ComputedField>(reinterpret_cast<const capnp::word*>(COMPUTED_FIELD_DEFAULT_VALUE));
	
	return cupnp::getPointerField<fsc::cu::ComputedField, 0>(structure, data, reinterpret_cast<const capnp::word*>(COMPUTED_FIELD_DEFAULT_VALUE));
} 

bool fsc::cu::MagneticField::isFilamentField() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 3;
} 

fsc::cu::MagneticField::FilamentField fsc::cu::MagneticField::getFilamentField() const { 
	return fsc::cu::MagneticField::FilamentField(structure, data);
} 

bool fsc::cu::MagneticField::isScaleBy() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 4;
} 

fsc::cu::MagneticField::ScaleBy fsc::cu::MagneticField::getScaleBy() const { 
	return fsc::cu::MagneticField::ScaleBy(structure, data);
} 

fsc::cu::MagneticField fsc::cu::MagneticField::mutateInvert() { 
	CUPNP_REQUIRE(nonDefaultInvert());
	return cupnp::mutatePointerField<fsc::cu::MagneticField, 0>(structure, data);
} 

bool fsc::cu::MagneticField::hasInvert() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 5;
} 

bool fsc::cu::MagneticField::nonDefaultInvert() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 5 && cupnp::hasPointerField<0>(structure, data);
} 

const fsc::cu::MagneticField fsc::cu::MagneticField::getInvert() const { 
	if(cupnp::getDiscriminant<0>(structure, data) != 5)
		return cupnp::getPointer<fsc::cu::MagneticField>(reinterpret_cast<const capnp::word*>(INVERT_DEFAULT_VALUE));
	
	return cupnp::getPointerField<fsc::cu::MagneticField, 0>(structure, data, reinterpret_cast<const capnp::word*>(INVERT_DEFAULT_VALUE));
} 

bool fsc::cu::MagneticField::isW7xMagneticConfig() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 6;
} 

fsc::cu::MagneticField::W7xMagneticConfig fsc::cu::MagneticField::getW7xMagneticConfig() const { 
	return fsc::cu::MagneticField::W7xMagneticConfig(structure, data);
} 

