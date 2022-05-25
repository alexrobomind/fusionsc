#pragma once 

#include <cupnp/cupnp.h>
#include "flt.capnp.h"
#include "data.capnp.cu.h"
#include "magnetics.capnp.cu.h"

namespace fsc{
namespace cu{

enum class FLTStopReason {
    UNKNOWN = 0,
    STEP_LIMIT = 1,
    DISTANCE_LIMIT = 2,
    TURN_LIMIT = 3,
    EVENT_BUFFER_FULL = 4,
    OUT_OF_GRID = 5,
};

struct FLTKernelState{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline FLTKernelState(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline static const unsigned char POSITION_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<float> mutatePosition();

	inline const cupnp::List<float> getPosition() const;
	inline bool nonDefaultPosition() const;

	inline uint32_t getNumSteps() const;
	inline void setNumSteps(uint32_t newVal);

	inline float getDistance() const;
	inline void setDistance(float newVal);

	inline uint32_t getTurnCount() const;
	inline void setTurnCount(uint32_t newVal);

	inline float getPhi0() const;
	inline void setPhi0(float newVal);

	inline uint32_t getEventCount() const;
	inline void setEventCount(uint32_t newVal);

}; // struct fsc::cu::FLTKernelState

struct FLTKernelEvent{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline FLTKernelEvent(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
struct PhiPlaneIntersection{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline PhiPlaneIntersection(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline uint32_t getPlaneNo() const;
	inline void setPlaneNo(uint32_t newVal);

}; // struct fsc::cu::FLTKernelEvent::PhiPlaneIntersection

	inline static const unsigned char LOCATION_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<float> mutateLocation();

	inline const cupnp::List<float> getLocation() const;
	inline bool nonDefaultLocation() const;

	inline uint32_t getStep() const;
	inline void setStep(uint32_t newVal);

	inline float getDistance() const;
	inline void setDistance(float newVal);

	inline bool hasOutOfGrid() const;
	inline bool isPhiPlaneIntersection() const;
	
	inline fsc::cu::FLTKernelEvent::PhiPlaneIntersection getPhiPlaneIntersection() const;

	inline uint32_t getNewTurn() const;
	inline void setNewTurn(uint32_t newVal);

	inline bool hasNewTurn() const;
}; // struct fsc::cu::FLTKernelEvent

struct FLTKernelData{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline FLTKernelData(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
struct Entry{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline Entry(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline fsc::cu::FLTStopReason getStopReason() const;
	inline void setStopReason(fsc::cu::FLTStopReason newVal);

	inline static const unsigned char STATE_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline fsc::cu::FLTKernelState mutateState();

	inline const fsc::cu::FLTKernelState getState() const;
	inline bool nonDefaultState() const;

	inline static const unsigned char EVENTS_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<fsc::cu::FLTKernelEvent> mutateEvents();

	inline const cupnp::List<fsc::cu::FLTKernelEvent> getEvents() const;
	inline bool nonDefaultEvents() const;

}; // struct fsc::cu::FLTKernelData::Entry

	inline static const unsigned char DATA_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<fsc::cu::FLTKernelData::Entry> mutateData();

	inline const cupnp::List<fsc::cu::FLTKernelData::Entry> getData() const;
	inline bool nonDefaultData() const;

}; // struct fsc::cu::FLTKernelData

struct FLTKernelRequest{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline FLTKernelRequest(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline static const unsigned char PHI_PLANES_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<float> mutatePhiPlanes();

	inline const cupnp::List<float> getPhiPlanes() const;
	inline bool nonDefaultPhiPlanes() const;

	inline uint32_t getTurnLimit() const;
	inline void setTurnLimit(uint32_t newVal);

	inline float getDistanceLimit() const;
	inline void setDistanceLimit(float newVal);

	inline uint32_t getStepLimit() const;
	inline void setStepLimit(uint32_t newVal);

	inline float getStepSize() const;
	inline void setStepSize(float newVal);

	inline static const unsigned char GRID_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline fsc::cu::ToroidalGrid mutateGrid();

	inline const fsc::cu::ToroidalGrid getGrid() const;
	inline bool nonDefaultGrid() const;

	inline static const unsigned char FIELD_DATA_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline fsc::cu::Float64Tensor mutateFieldData();

	inline const fsc::cu::Float64Tensor getFieldData() const;
	inline bool nonDefaultFieldData() const;

}; // struct fsc::cu::FLTKernelRequest


}}
// ===== struct fsc::cu::FLTKernelState =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::FLTKernelState> { using Type = fsc::cu::FLTKernelState; }; 
} // namespace ::cupnp

cupnp::List<float> fsc::cu::FLTKernelState::mutatePosition() { 
	CUPNP_REQUIRE(nonDefaultPosition());
	return cupnp::mutatePointerField<cupnp::List<float>, 0>(structure, data);
} 

bool fsc::cu::FLTKernelState::nonDefaultPosition() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const cupnp::List<float> fsc::cu::FLTKernelState::getPosition() const { 
	return cupnp::getPointerField<cupnp::List<float>, 0>(structure, data, reinterpret_cast<const capnp::word*>(POSITION_DEFAULT_VALUE));
} 

void fsc::cu::FLTKernelState::setNumSteps(uint32_t newVal) { 
	cupnp::setPrimitiveField<uint32_t, 0>(structure, data, 0, newVal);
} 

uint32_t fsc::cu::FLTKernelState::getNumSteps() const { 
	return cupnp::getPrimitiveField<uint32_t, 0>(structure, data, 0);
} 

void fsc::cu::FLTKernelState::setDistance(float newVal) { 
	cupnp::setPrimitiveField<float, 1>(structure, data, 0, newVal);
} 

float fsc::cu::FLTKernelState::getDistance() const { 
	return cupnp::getPrimitiveField<float, 1>(structure, data, 0);
} 

void fsc::cu::FLTKernelState::setTurnCount(uint32_t newVal) { 
	cupnp::setPrimitiveField<uint32_t, 2>(structure, data, 0, newVal);
} 

uint32_t fsc::cu::FLTKernelState::getTurnCount() const { 
	return cupnp::getPrimitiveField<uint32_t, 2>(structure, data, 0);
} 

void fsc::cu::FLTKernelState::setPhi0(float newVal) { 
	cupnp::setPrimitiveField<float, 3>(structure, data, 0, newVal);
} 

float fsc::cu::FLTKernelState::getPhi0() const { 
	return cupnp::getPrimitiveField<float, 3>(structure, data, 0);
} 

void fsc::cu::FLTKernelState::setEventCount(uint32_t newVal) { 
	cupnp::setPrimitiveField<uint32_t, 4>(structure, data, 0, newVal);
} 

uint32_t fsc::cu::FLTKernelState::getEventCount() const { 
	return cupnp::getPrimitiveField<uint32_t, 4>(structure, data, 0);
} 

// ===== struct fsc::cu::FLTKernelEvent::PhiPlaneIntersection =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::FLTKernelEvent::PhiPlaneIntersection> { using Type = fsc::cu::FLTKernelEvent::PhiPlaneIntersection; }; 
} // namespace ::cupnp

void fsc::cu::FLTKernelEvent::PhiPlaneIntersection::setPlaneNo(uint32_t newVal) { 
	cupnp::setPrimitiveField<uint32_t, 3>(structure, data, 0, newVal);
} 

uint32_t fsc::cu::FLTKernelEvent::PhiPlaneIntersection::getPlaneNo() const { 
	return cupnp::getPrimitiveField<uint32_t, 3>(structure, data, 0);
} 

// ===== struct fsc::cu::FLTKernelEvent =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::FLTKernelEvent> { using Type = fsc::cu::FLTKernelEvent; }; 
} // namespace ::cupnp

cupnp::List<float> fsc::cu::FLTKernelEvent::mutateLocation() { 
	CUPNP_REQUIRE(nonDefaultLocation());
	return cupnp::mutatePointerField<cupnp::List<float>, 0>(structure, data);
} 

bool fsc::cu::FLTKernelEvent::nonDefaultLocation() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const cupnp::List<float> fsc::cu::FLTKernelEvent::getLocation() const { 
	return cupnp::getPointerField<cupnp::List<float>, 0>(structure, data, reinterpret_cast<const capnp::word*>(LOCATION_DEFAULT_VALUE));
} 

void fsc::cu::FLTKernelEvent::setStep(uint32_t newVal) { 
	cupnp::setPrimitiveField<uint32_t, 0>(structure, data, 0, newVal);
} 

uint32_t fsc::cu::FLTKernelEvent::getStep() const { 
	return cupnp::getPrimitiveField<uint32_t, 0>(structure, data, 0);
} 

void fsc::cu::FLTKernelEvent::setDistance(float newVal) { 
	cupnp::setPrimitiveField<float, 1>(structure, data, 0, newVal);
} 

float fsc::cu::FLTKernelEvent::getDistance() const { 
	return cupnp::getPrimitiveField<float, 1>(structure, data, 0);
} 

bool fsc::cu::FLTKernelEvent::hasOutOfGrid() const { 
	return cupnp::getDiscriminant<4>(structure, data) == 0;
} 

bool fsc::cu::FLTKernelEvent::isPhiPlaneIntersection() const { 
	return cupnp::getDiscriminant<4>(structure, data) == 1;
} 

fsc::cu::FLTKernelEvent::PhiPlaneIntersection fsc::cu::FLTKernelEvent::getPhiPlaneIntersection() const { 
	return fsc::cu::FLTKernelEvent::PhiPlaneIntersection(structure, data);
} 

void fsc::cu::FLTKernelEvent::setNewTurn(uint32_t newVal) { 
	cupnp::setPrimitiveField<uint32_t, 3>(structure, data, 0, newVal);
	cupnp::setDiscriminant<4>(structure, data, 2);
} 

uint32_t fsc::cu::FLTKernelEvent::getNewTurn() const { 
	if(cupnp::getDiscriminant<4>(structure, data) != 2)
		return 0;
	
	return cupnp::getPrimitiveField<uint32_t, 3>(structure, data, 0);
} 

bool fsc::cu::FLTKernelEvent::hasNewTurn() const { 
	return cupnp::getDiscriminant<4>(structure, data) == 2;
} 

// ===== struct fsc::cu::FLTKernelData::Entry =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::FLTKernelData::Entry> { using Type = fsc::cu::FLTKernelData::Entry; }; 
} // namespace ::cupnp

void fsc::cu::FLTKernelData::Entry::setStopReason(fsc::cu::FLTStopReason newVal) { 
	cupnp::setPrimitiveField<uint16_t, 0>(structure, data, 0, static_cast<uint16_t>(newVal));
} 

fsc::cu::FLTStopReason fsc::cu::FLTKernelData::Entry::getStopReason() const { 
	return static_cast<fsc::cu::FLTStopReason>(cupnp::getPrimitiveField<uint16_t, 0>(structure, data, 0));
} 

fsc::cu::FLTKernelState fsc::cu::FLTKernelData::Entry::mutateState() { 
	CUPNP_REQUIRE(nonDefaultState());
	return cupnp::mutatePointerField<fsc::cu::FLTKernelState, 0>(structure, data);
} 

bool fsc::cu::FLTKernelData::Entry::nonDefaultState() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const fsc::cu::FLTKernelState fsc::cu::FLTKernelData::Entry::getState() const { 
	return cupnp::getPointerField<fsc::cu::FLTKernelState, 0>(structure, data, reinterpret_cast<const capnp::word*>(STATE_DEFAULT_VALUE));
} 

cupnp::List<fsc::cu::FLTKernelEvent> fsc::cu::FLTKernelData::Entry::mutateEvents() { 
	CUPNP_REQUIRE(nonDefaultEvents());
	return cupnp::mutatePointerField<cupnp::List<fsc::cu::FLTKernelEvent>, 1>(structure, data);
} 

bool fsc::cu::FLTKernelData::Entry::nonDefaultEvents() const { 
	return cupnp::hasPointerField<1>(structure, data);
} 

const cupnp::List<fsc::cu::FLTKernelEvent> fsc::cu::FLTKernelData::Entry::getEvents() const { 
	return cupnp::getPointerField<cupnp::List<fsc::cu::FLTKernelEvent>, 1>(structure, data, reinterpret_cast<const capnp::word*>(EVENTS_DEFAULT_VALUE));
} 

// ===== struct fsc::cu::FLTKernelData =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::FLTKernelData> { using Type = fsc::cu::FLTKernelData; }; 
} // namespace ::cupnp

cupnp::List<fsc::cu::FLTKernelData::Entry> fsc::cu::FLTKernelData::mutateData() { 
	CUPNP_REQUIRE(nonDefaultData());
	return cupnp::mutatePointerField<cupnp::List<fsc::cu::FLTKernelData::Entry>, 0>(structure, data);
} 

bool fsc::cu::FLTKernelData::nonDefaultData() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const cupnp::List<fsc::cu::FLTKernelData::Entry> fsc::cu::FLTKernelData::getData() const { 
	return cupnp::getPointerField<cupnp::List<fsc::cu::FLTKernelData::Entry>, 0>(structure, data, reinterpret_cast<const capnp::word*>(DATA_DEFAULT_VALUE));
} 

// ===== struct fsc::cu::FLTKernelRequest =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::FLTKernelRequest> { using Type = fsc::cu::FLTKernelRequest; }; 
} // namespace ::cupnp

cupnp::List<float> fsc::cu::FLTKernelRequest::mutatePhiPlanes() { 
	CUPNP_REQUIRE(nonDefaultPhiPlanes());
	return cupnp::mutatePointerField<cupnp::List<float>, 0>(structure, data);
} 

bool fsc::cu::FLTKernelRequest::nonDefaultPhiPlanes() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const cupnp::List<float> fsc::cu::FLTKernelRequest::getPhiPlanes() const { 
	return cupnp::getPointerField<cupnp::List<float>, 0>(structure, data, reinterpret_cast<const capnp::word*>(PHI_PLANES_DEFAULT_VALUE));
} 

void fsc::cu::FLTKernelRequest::setTurnLimit(uint32_t newVal) { 
	cupnp::setPrimitiveField<uint32_t, 0>(structure, data, 0, newVal);
} 

uint32_t fsc::cu::FLTKernelRequest::getTurnLimit() const { 
	return cupnp::getPrimitiveField<uint32_t, 0>(structure, data, 0);
} 

void fsc::cu::FLTKernelRequest::setDistanceLimit(float newVal) { 
	cupnp::setPrimitiveField<float, 1>(structure, data, 0, newVal);
} 

float fsc::cu::FLTKernelRequest::getDistanceLimit() const { 
	return cupnp::getPrimitiveField<float, 1>(structure, data, 0);
} 

void fsc::cu::FLTKernelRequest::setStepLimit(uint32_t newVal) { 
	cupnp::setPrimitiveField<uint32_t, 2>(structure, data, 0, newVal);
} 

uint32_t fsc::cu::FLTKernelRequest::getStepLimit() const { 
	return cupnp::getPrimitiveField<uint32_t, 2>(structure, data, 0);
} 

void fsc::cu::FLTKernelRequest::setStepSize(float newVal) { 
	cupnp::setPrimitiveField<float, 3>(structure, data, 0, newVal);
} 

float fsc::cu::FLTKernelRequest::getStepSize() const { 
	return cupnp::getPrimitiveField<float, 3>(structure, data, 0);
} 

fsc::cu::ToroidalGrid fsc::cu::FLTKernelRequest::mutateGrid() { 
	CUPNP_REQUIRE(nonDefaultGrid());
	return cupnp::mutatePointerField<fsc::cu::ToroidalGrid, 1>(structure, data);
} 

bool fsc::cu::FLTKernelRequest::nonDefaultGrid() const { 
	return cupnp::hasPointerField<1>(structure, data);
} 

const fsc::cu::ToroidalGrid fsc::cu::FLTKernelRequest::getGrid() const { 
	return cupnp::getPointerField<fsc::cu::ToroidalGrid, 1>(structure, data, reinterpret_cast<const capnp::word*>(GRID_DEFAULT_VALUE));
} 

fsc::cu::Float64Tensor fsc::cu::FLTKernelRequest::mutateFieldData() { 
	CUPNP_REQUIRE(nonDefaultFieldData());
	return cupnp::mutatePointerField<fsc::cu::Float64Tensor, 2>(structure, data);
} 

bool fsc::cu::FLTKernelRequest::nonDefaultFieldData() const { 
	return cupnp::hasPointerField<2>(structure, data);
} 

const fsc::cu::Float64Tensor fsc::cu::FLTKernelRequest::getFieldData() const { 
	return cupnp::getPointerField<fsc::cu::Float64Tensor, 2>(structure, data, reinterpret_cast<const capnp::word*>(FIELD_DATA_DEFAULT_VALUE));
} 

