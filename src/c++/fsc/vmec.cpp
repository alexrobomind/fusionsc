#include "vmec.h"
#include "common.h"
#include "tensor.h"

#include <H5cpp.h>

namespace fsc {

namespace {

// Helper functions to create VMEC input files

kj::StringPtr fBool(bool val) {
	return val ? "T" : "F";
}

template<typename L>
kj::StringTree fArray(L&& l) {
	auto builder = kj::heapArrayBuilder<kj::StringTree>(l.size());
	
	for(auto el : l) {
		builder.add(kj::strTree(el));
	}
	
	return kj::StringTree(builder.finish(), " ");
}

kj::StringPtr decodeSplineType(VmecProfile::SplineType type) {
	switch(type) {
		case VmecProfile::SplineType::AKIMA: return "Akima_spline";
		case VmecProfile::SplineType::CUBIC: return "cubic_spline";
	}
	
	KJ_FAIL_REQUIRE("Unknown spline type", type);
}

kj::StringTree makeCurrentProfile(VmecProfile::Reader cp) {	
	if(cp.isPowerSeries()) {
		return kj::strTree(
			"PCURR_TYPE = 'power_series_I'\n"
			"AC = ", fArray(cp.getPowerSeries()), "\n"
		);
	} else if(cp.isSpline()) {
		auto s = cp.getSpline();
		
		return kj::strTree(
			"PCURR_TYPE = ", decodeSplineType(s.getType()), "_I\n"
			"ac_aux_s = ", fArray(s.getLocations()), "\n"
			"ac_aux_f = ", fArray(s.getValues()), "\n"
		);
	}
	
	KJ_FAIL_REQUIRE("Unknown profile type", cp);
}

kj::StringTree makeCurrentDensityProfile(VmecProfile::Reader cp) {	
	if(cp.isPowerSeries()) {
		return kj::strTree(
			"PCURR_TYPE = 'power_series'\n"
			"AC = ", fArray(cp.getPowerSeries()), "\n"
		);
	} else if(cp.isSpline()) {
		auto s = cp.getSpline();
		
		return kj::strTree(
			"PCURR_TYPE = ", decodeSplineType(s.getType()), "_Ip\n"
			"ac_aux_s = ", fArray(s.getLocations()), "\n"
			"ac_aux_f = ", fArray(s.getValues()), "\n"
		);
	}
	
	KJ_FAIL_REQUIRE("Unknown profile type", cp);
}

kj::StringTree makeIotaProfile(VmecProfile::Reader cp) {	
	if(cp.isPowerSeries()) {
		return kj::strTree(
			"PIOTA_TYPE = 'power_series'\n"
			"AI = ", fArray(cp.getPowerSeries()), "\n"
		);
	} else if(cp.isSpline()) {
		auto s = cp.getSpline();
		
		return kj::strTree(
			"PIOTA_TYPE = ", decodeSplineType(s.getType()), "\n"
			"ai_aux_s = ", fArray(s.getLocations()), "\n"
			"ai_aux_f = ", fArray(s.getValues()), "\n"
		);
	}
	
	KJ_FAIL_REQUIRE("Unknown profile type", cp);
}

kj::StringTree makeMassProfile(VmecProfile::Reader mp) {	
	if(mp.isPowerSeries()) {
		return kj::strTree(
			"PMASS_TYPE = 'power_series'\n"
			"AM = ", fArray(mp.getPowerSeries()), "\n"
		);
	} else if(mp.isSpline()) {
		auto s = mp.getSpline();
		
		return kj::strTree(
			"PMASS_TYPE = ", decodeSplineType(s.getType()), "\n"
			"am_aux_s = ", fArray(s.getLocations()), "\n"
			"am_aux_f = ", fArray(s.getValues()), "\n"
		);
	}
	
	KJ_FAIL_REQUIRE("Unknown profile type", mp);
}

kj::StringTree makeCurrentOrDensityProfile(VmecRequest::Iota::FromCurrent::Reader in) {
	if(in.isCurrentProfile()) {
		return makeCurrentProfile(in.getCurrentProfile());
	}
	
	if(in.isCurrentDensityProfile()) {
		return makeCurrentDensityProfile(in.getCurrentDensityProfile());
	}
	
	KJ_FAIL_REQUIRE("Unknown current interpretation, only current or current density supported", in);
}

void validateSurfaces(VmecSurfaces::Reader in) {
	auto nTor = in.getNTor();
	auto mPol = in.getMPol();
	
	auto nSurf = in.getRCos().getShape()[0];
	KJ_REQUIRE(nSurf >= 2);
	
	validateTensor(in.getRCos(), {nSurf, 2 * mPol + 1, nTor});
	validateTensor(in.getZSin(), {nSurf, 2 * mPol + 1, nTor});
	
	if(in.isNonSymmetric()) {
		validateTensor(in.getNonSymmetric().getRSin(), {nSurf, 2 * mPol + 1, nTor});
		validateTensor(in.getNonSymmetric().getZCos(), {nSurf, 2 * mPol + 1, nTor});
	}
}

kj::StringTree makeAxisData(VmecSurfaces::Reader in) {
	validateSurfaces();
	
	auto nTor = in.getNTor();
	auto mPol = in.getMPol();
	
	// Symmetric axis coefficients
	Tensor<double, 3> rCos;
	Tensor<double, 3> zSin;
	readTensor(in.getRCos(), rCos);
	readTensor(in.getZSin(), zSin);
	
	kj::StringTree result;
	auto rAxisCC = kj::heapArrayBuilder<double>(nTor);
	auto zAxisCS = kj::heapArrayBuilder<double>(nTor);
	
	for(auto iTor : kj::range(0, nTor)) {
		rAxisCC.add(rCos(iTor, mPol, 0));
		zAxisCs.add(zSin(iTor, mPol, 0));
	}
	
	auto result = kj::strTree(
		"RAXIS_CC = ", fArray(rAxisCC.finish()), "\n"
		"ZAXIS_CS = ", fArray(zAxisCs.finish()), "\n"
	);
	
	if(in.isNonSymmetric()) {
		Tensor<double, 3> rSin;
		Tensor<double, 3> zCos;
		readTensor(in.getNonSymmetric().getRSin(), rSin);
		readTensor(in.getNonSymmetric().getZCos(), zCos);
	
		auto rAxisCS = kj::heapArrayBuilder<double>(nTor);
		auto zAxisCC = kj::heapArrayBuilder<double>(nTor);
		for(auto iTor : kj::range(0, nTor)) {
			rAxisCS.add(rCos(iTor, mPol, 0));
			zAxisCC.add(zSin(iTor, mPol, 0));
		}
		
		result = kj::strTree(
			mv(result),
			"RAXIS_CS = ", fArray(rAxisCS.finish()), "\n"
			"ZAXIS_CC = ", fArray(zAxisCC.finish()), "\n"
		);
	}
	
	return result;
}

kj::StringTree makeBoundaryData(VmecSurfaces::Reader in) {
	validateSurfaces();
	
	auto nTor = in.getNTor();
	auto mPol = in.getMPol();
	
	// Symmetric axis coefficients
	Tensor<double, 3> rCos;
	Tensor<double, 3> zSin;
	readTensor(in.getRCos(), rCos);
	readTensor(in.getZSin(), zSin);
	
	auto iMax = rCos.dimension(2) - 1;
	
	kj::StringTree result;
	
	for(auto iTor : kj::range(0, nTor)) {
		for(auto jPol : kj::range(0, 2 * mPol + 1)) {
			int64_t jReal = static_cast<int64_t>(jPol) - mPol;
			
			result = kj::strTree(
				mv(result),
				"RBC(", jReal, ", ", iTor, ") = ", rCos(iTor, mPol, iMax), "  "
				"ZBS(", jReal, ", ", iTor, ") = ", zSin(iTor, mPol, iMax), "\n"
			);
		}
	}
	
	if(in.isNonSymmetric()) {
		Tensor<double, 3> rSin;
		Tensor<double, 3> zCos;
		readTensor(in.getNonSymmetric().getRSin(), rSin);
		readTensor(in.getNonSymmetric().getZCos(), zCos);
	
		for(auto iTor : kj::range(0, nTor)) {
			for(auto jPol : kj::range(0, 2 * mPol + 1)) {
				int64_t jReal = static_cast<int64_t>(jPol) - mPol;
				
				result = kj::strTree(
					mv(result),
					"RBZ(", jReal, ", ", iTor, ") = ", rSin(iTor, mPol, iMax), "  "
					"ZBC(", jReal, ", ", iTor, ") = ", zCos(iTor, mPol, iMax), "\n"
				);
			}
		}
	}
	
	return result;
}

Promise<void> writeMGridFile(kj::Path path, ComputedField::Reader cField) {
	return getActiveThread().dataService().download(cField.getData())
	.then([path, grid = cField.getGrid()](LocalDataRef<Float64Tensor> data) {
		// Data types
		H5::DataType intType(H5::PredType::NATIVE_INT);
		H5::DataType doubleType(H5::PredType::NATIVE_DOUBLE);
		H5::DataType charType(H5::PredType::NATIVE_CHAR);
		
		// Data shapes
		hsize_t shapeContainer[4];
		
		H5::DataSpace scalarShape();
		
		shapeContainer[0] = grid.getNPhi();
		shapeContainer[1] = grid.getNZ();
		shapeContainer[2] = grid.getNR();
		H5::DataSpace fieldShape(3, shapeContainer);
		
		shapeContainer[0] = 1;
		shapeContainer[1] = 30;
		H5::DataSpace coilGroupShape(2, shapeContainer);
		
		shapeContainer[0] = 1;
		H5::DataSpace mgridModeShape(1, shapeContainer);
		
		// File
		H5::H5File file(path.toNativeString(true).cStr(), H5F_ACC_TRUNC);
		
		// Variables
		auto writeScalar = [&](auto value
	};
}

//! Generates the input file for a VMEC request
kj::StringTree generateVmecInput(VmecRequest::Reader request, kj::Path mgridFile) {
	auto sp = request.getStartingPoint();
	auto surfShape = sp.getRCos().getShape();
	auto nSurf = surfShape[0];
	auto n = surfShape[1];
	auto m = surfShape[2];
	auto period = sp.getPeriod();
	
	auto iota = request.getIota();
	auto rp = request.getRunParams();
	
	KJ_REQUIRE(request.getPhiEdge() != 0, "phiEdge must be provided");
	
	kj::StringTree result = kj::strTree(
		"LFREEB = ", fBool(request.isFreeBoundary()), "\n"
		"LOLDOUT = F\n"
		"LWOUTTXT = T\n"
		"LDIAGNO = F\n"
		"LFULL3D1OUT = F\n"
		"LGIVEUP = F\n"
		"FGIVEUP = 30.\n"
		"MAX_MAIN_ITERATIONS = 2\n"
		"DELT = ", rp.getTimeStep(), "\n"
		"TCON0 = 2.\n"
		"NTOR = ", rp.getStartingPoint().getNTor(), "\n"
		"MPOL = ", rp.getStartingPoint().getMPol(), "\n"
		"NFP = ", period, "\n"
		"NCURR = ", iota.isIotaProfile() ? 0 : 1, "\n"
		"MPOL = ", m, " NTOR = ", n, "\n"
		"NZETA = ", rp.getNGridToroidal(), " NTHETA = ", rp.getNGridPoloidal(), "\n"
		"NS_ARRAY = ", fArray(rp.getNGridRadial()), "\n"
		"NITER = ", rp.getMaxIterationsPerSequence(), "\n"
		"NSTEP = ", rp.getConvergenceSteps(), "\n"
		"NVACSKIP = ", rp.getVacuumCalcSteps(), "\n"
		"GAMMA = ", request.getGamma(), "\n"
		"FTOL_ARRAY = ", fArray(rp.getForceToleranceLevels()), "\n"
		"PHIEDGE = ", request.getPhiEdge(), "\n"
		"BLOAT = 1.\n",
		
		makeMassProfile(request.getMassProfile()),
		
		makeAxisData(request.getStartingPoint()),
		makeBoundaryData(request.getStartingPoint())
	);
	
	if(iota.isIotaProfile()) {
		result = kj::strTree(
			mv(result),
			makeIotaProfile(iota.getIotaProfile())
		);
	} else if(iota.isFromCurrent()) {
		auto fromC = iota.getFromCurrent();
		result = kj::strTree(
			mv(result),
			"CURTOR = ", fromC.getTotalCurrent(), "\n",
			makeCurrentOrDensityProfile(fromC)
		);
	}
	
	if(request.isFreeBoundary()) {
		result = kj::strTree(
			mv(result),
			"MGRID_FILE = '", mgridFile.toNativeString(true), "'\n",
			"EXTCUR = 1\n"
		);
		KJ_UNIMPLEMENTED("Free boundary runs not supported");
	}
	
	result = kj::strTree(
		"&INDATA\n", mv(result), "/\n"
	);
	
	return result;
}

struct VmecDriverImpl : public VmecDriver::Server {
	JobScheduler::Client scheduler;
	
	kj::Path rootPath;
	Own<kj::Directory> rootDirectory;
	
	uint64_t jobDirCounter = 0;
	
	VmecDriverImpl(JobScheduler::Client scheduler, kj::Filesystem& fs, kj::Path rootPath) :
		scheduler(mv(scheduler)), rootPath(rootPath),
		rootDirectory(fs.getCurrent() -> openSubdir(rootPath, WriteMode::CREATE | WriteMode::MODIFY))
	{}
	
	kj::Path createJobDirectory() {
		auto names = rootDirectory -> listNames();
		while(true) {
			auto nameCandidate = kj::str("vmecJob", jobDirCounter++);
			
			for(auto& name : names) {
				if(name == nameCandidate) {
					goto nextCandidate;
				}
			}
			
			rootDirectory -> openSubdir(nameCandidate, WriteMode::CREATE);
			return rootPath.append(nameCandidate);
			
			nextCandidate:
				continue;
		}
	}
};

}

VmecDriver::Client createVmecDriver(JobScheduler::Client scheduler) {
	return kj::heap<VmecDriverImpl>(mv(scheduler));
}

}