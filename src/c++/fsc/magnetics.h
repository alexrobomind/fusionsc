#pragma once

#include <fsc/magnetics.capnp.h>
#include <kj/map.h>

#include "common.h"
#include "local.h"
#include "tensor.h"

namespace fsc {

class FieldResolverBase : public FieldResolver::Server {
public:
	LibraryThread lt;
	FieldResolverBase(LibraryThread& lt);
	
	virtual Promise<void> resolve(ResolveContext context) override;
	
	virtual Promise<void> processField(MagneticField::Reader input, MagneticField::Builder output, ResolveContext context);
	virtual Promise<void> processFilament(Filament::Reader input, Filament::Builder output, ResolveContext context);
};

struct ToroidalGridStruct {
	double rMin; double rMax; unsigned int nR;
	double zMin; double zMax; unsigned int nZ;
	unsigned int nSym; unsigned int nPhi;
	
	ToroidalGridStruct() = default;
	inline ToroidalGridStruct(ToroidalGrid::Reader in, unsigned int maxOrdinal) { read(in, maxOrdinal); }
	
	inline bool isValid() {
		return (nR >=2) && (nZ >= 2) && (nPhi >= 1) && (nSym >= 1) && (rMax > rMin) && (zMax > zMin);
	}
	
	void read(ToroidalGrid::Reader in, unsigned int maxOrdinal);
	void write(ToroidalGrid::Builder out);
	
	inline EIGEN_DEVICE_FUNC double phi(int i_phi);
	inline EIGEN_DEVICE_FUNC double r(int i_r);
	inline EIGEN_DEVICE_FUNC double z(int i_z);
	
	inline EIGEN_DEVICE_FUNC Vec3<double> xyz(int i_phi, int i_z, int i_r);
	inline EIGEN_DEVICE_FUNC Vec3<double> phizr(int i_phi, int i_z, int i_r);
};

/**
 * Creates a new field calculator.
 */
FieldCalculator::Client newCPUFieldCalculator(LibraryThread& lt);

// Inline Implementation

inline EIGEN_DEVICE_FUNC double ToroidalGridStruct::phi(int i) {
	return 2 * pi / nSym / nPhi * i;
}

inline EIGEN_DEVICE_FUNC double ToroidalGridStruct::r(int i) {
	return rMin + (rMax - rMin) / (nR - 1) * i;
}

inline EIGEN_DEVICE_FUNC double ToroidalGridStruct::z(int i) {
	return zMin + (zMax - zMin) / (nZ - 1) * i;
}
	
inline EIGEN_DEVICE_FUNC Vec3<double> ToroidalGridStruct::xyz(int i_phi, int i_z, int i_r) {
	double r = rMin + (rMax - rMin) / (nR - 1) * i_r;
	double z = zMin + (zMax - zMin) / (nZ - 1) * i_z;
	double phi = 2 * pi / nSym / nPhi;
	
	double x = r * cos(phi);
	double y = r * sin(phi);
	
	Vec3<double> result;
	result(0) = x;
	result(1) = y;
	result(2) = z;
	
	return result;
}

inline EIGEN_DEVICE_FUNC Vec3<double> ToroidalGridStruct::phizr(int i_phi, int i_z, int i_r) {
	double r = rMin + (rMax - rMin) / (nR - 1) * i_r;
	double z = zMin + (zMax - zMin) / (nZ - 1) * i_z;
	double vphi = phi(i_phi);
	
	Vec3<double> result;
	result(0) = vphi;
	result(1) = z;
	result(2) = r;
	
	return result;
}

}