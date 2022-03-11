#pragma once

#include "tensor.h"

namespace fsc {

struct ToroidalGridStruct {
	double rMin; double rMax; unsigned int nR;
	double zMin; double zMax; unsigned int nZ;
	unsigned int nSym; unsigned int nPhi;
		
	inline bool isValid() const {
		return (nR >=2) && (nZ >= 2) && (nPhi >= 1) && (nSym >= 1) && (rMax > rMin) && (zMax > zMin);
	}
	
	inline EIGEN_DEVICE_FUNC double phi(int i_phi) const;
	inline EIGEN_DEVICE_FUNC double r(int i_r) const;
	inline EIGEN_DEVICE_FUNC double z(int i_z) const;
	
	inline EIGEN_DEVICE_FUNC Vec3d xyz(int i_phi, int i_z, int i_r) const;
	inline EIGEN_DEVICE_FUNC Vec3d phizr(int i_phi, int i_z, int i_r) const;
};

inline EIGEN_DEVICE_FUNC double ToroidalGridStruct::phi(int i) const {
	return 2 * pi / nSym / nPhi * i;
}

inline EIGEN_DEVICE_FUNC double ToroidalGridStruct::r(int i) const {
	return rMin + (rMax - rMin) / (nR - 1) * i;
}

inline EIGEN_DEVICE_FUNC double ToroidalGridStruct::z(int i) const {
	return zMin + (zMax - zMin) / (nZ - 1) * i;
}
	
inline EIGEN_DEVICE_FUNC Vec3d ToroidalGridStruct::xyz(int i_phi, int i_z, int i_r) const {
	double rv = r(i_r);
	double phiv = phi(i_phi);
	
	double x = rv * cos(phiv);
	double y = rv * sin(phiv);
	
	/*Vec3<double> result;
	result(0) = x;
	result(1) = y;
	result(2) = z(i_z);*/
	
	return {x, y, z(i_z) };
}

inline EIGEN_DEVICE_FUNC Vec3d ToroidalGridStruct::phizr(int i_phi, int i_z, int i_r) const {
	/*Vec3<double> result;
	result(0) = phi(i_phi);
	result(1) = z(i_z);
	result(2) = r(i_r);*/
	
	return { phi(i_phi), z(i_z), r(i_r) };
}

}