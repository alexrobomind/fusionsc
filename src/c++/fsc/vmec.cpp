#include "vmec.h"

#include "common.h"
#include "tensor.h"
#include "data.h"

#include "hdf5.h"

#include <kj/filesystem.h>

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
	validateSurfaces(in);
	
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
		zAxisCS.add(zSin(iTor, mPol, 0));
	}
	
	result = kj::strTree(
		"\n",
		"! --- Magnetic axis ---\n",
		"RAXIS_CC = ", fArray(rAxisCC.finish()), "\n"
		"ZAXIS_CS = ", fArray(zAxisCS.finish()), "\n"
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
	validateSurfaces(in);
	
	auto nTor = in.getNTor();
	auto mPol = in.getMPol();
	
	// Symmetric axis coefficients
	Tensor<double, 3> rCos;
	Tensor<double, 3> zSin;
	readTensor(in.getRCos(), rCos);
	readTensor(in.getZSin(), zSin);
	
	auto iMax = rCos.dimension(2) - 1;
	
	kj::StringTree result;
	
	result = kj::strTree(
		mv(result), "\n",
		"! --- Symmetric flux surface coefficients --- \n"
	);
	
	for(auto iTor : kj::range(0, nTor)) {
		result = kj::strTree(
			mv(result),
			"! n = ", iTor, "\n"
		);
		for(auto jPol : kj::range(0, 2 * mPol + 1)) {
			int64_t jReal = static_cast<int64_t>(jPol) - mPol;
			
			result = kj::strTree(
				mv(result),
				"RBC(", jReal, ", ", iTor, ") = ", rCos(iTor, jPol, iMax), "  "
				"ZBS(", jReal, ", ", iTor, ") = ", zSin(iTor, jPol, iMax), "\n"
			);
		}
	}
	
	if(in.isNonSymmetric()) {
		Tensor<double, 3> rSin;
		Tensor<double, 3> zCos;
		readTensor(in.getNonSymmetric().getRSin(), rSin);
		readTensor(in.getNonSymmetric().getZCos(), zCos);
	
		result = kj::strTree(
			mv(result),
			"! --- Antisymmetric flux surface coefficients --- \n"
		);
	
		for(auto iTor : kj::range(0, nTor)) {
			result = kj::strTree(
				mv(result),
				"! n = ", iTor, "\n"
			);
			for(auto jPol : kj::range(0, 2 * mPol + 1)) {
				int64_t jReal = static_cast<int64_t>(jPol) - mPol;
				
				result = kj::strTree(
					mv(result),
					"RBZ(", jReal, ", ", iTor, ") = ", rSin(iTor, jPol, iMax), "  "
					"ZBC(", jReal, ", ", iTor, ") = ", zCos(iTor, jPol, iMax), "\n"
				);
			}
		}
	}
	
	return result;
}

struct VmecDriverImpl : public VmecDriver::Server {
	JobScheduler::Client scheduler;
	
	kj::Path rootPath;
	Own<const kj::Directory> rootDirectory;
	
	uint64_t jobDirCounter = 0;
	
	VmecDriverImpl(JobScheduler::Client scheduler, kj::Filesystem& fs, kj::PathPtr parPath) :
		scheduler(mv(scheduler)), rootPath(parPath.clone()),
		rootDirectory(fs.getCurrent().openSubdir(rootPath, kj::WriteMode::CREATE | kj::WriteMode::MODIFY))
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
			
			rootDirectory -> openSubdir(kj::Path(nameCandidate), kj::WriteMode::CREATE);
			return rootPath.append(nameCandidate);
			
			nextCandidate:
				continue;
		}
	}
};

}

//! Generates the input file for a VMEC request
kj::String generateVmecInput(VmecRequest::Reader request, kj::PathPtr mgridFile) {
	auto sp = request.getStartingPoint();
	auto iota = request.getIota();
	auto runParams = request.getRunParams();
	
	KJ_REQUIRE(request.getPhiEdge() != 0, "phiEdge must be provided");
	KJ_REQUIRE(runParams.getNGridPoloidal() >= 2 * runParams.getMPolMax() + 6, "Poloidal grid resolution too low");
	
	kj::StringTree result = kj::strTree(
		"! --- Fixed settings ---\n"
		"LOLDOUT = F\n"
		"LWOUTTXT = T\n"
		"LDIAGNO = F\n"
		"LFULL3D1OUT = F\n"
		"LGIVEUP = F\n"
		"FGIVEUP = 30.\n"
		"MAX_MAIN_ITERATIONS = 2\n"
		"TCON0 = 2.\n"
		"BLOAT = 1.\n"
		"\n"
		
		"! --- General settings ---\n"
		"LFREEB = ", fBool(request.isFreeBoundary()), " ! Whether to use a free boundary run\n"
		"DELT = ", runParams.getTimeStep(), " ! Blend factor between runs\n"
		"NTOR = ", runParams.getNTorMax(), "\n"
		"MPOL = ", runParams.getMPolMax(), "\n"
		"NFP = ", sp.getPeriod(), "\n"
		"NTHETA = ", runParams.getNGridPoloidal(), "\n"
		"NS_ARRAY = ", fArray(runParams.getNGridRadial()), "\n"
		"NITER = ", runParams.getMaxIterationsPerSequence(), "\n"
		"NSTEP = ", runParams.getConvergenceSteps(), "\n"
		"NVACSKIP = ", runParams.getVacuumCalcSteps(), "\n"
		"FTOL_ARRAY = ", fArray(runParams.getForceToleranceLevels()), "\n"
		"PHIEDGE = ", request.getPhiEdge(), "\n"
		"\n"
		
		"! --- Mass / pressure profile ---\n",
		"GAMMA = ", request.getGamma(), "\n",
		makeMassProfile(request.getMassProfile()),
		
		makeAxisData(request.getStartingPoint()),
		makeBoundaryData(request.getStartingPoint())
	);
	
	if(iota.isIotaProfile()) {
		result = kj::strTree(
			mv(result), "\n",
			"! --- Iota profile ---\n",
			"NCURR = 0\n",
			makeIotaProfile(iota.getIotaProfile())
		);
	} else if(iota.isFromCurrent()) {
		auto fromC = iota.getFromCurrent();
		result = kj::strTree(
			mv(result), "\n",
			"! --- Current profile ---\n",
			"NCURR = 1\n"
			"CURTOR = ", fromC.getTotalCurrent(), "\n",
			makeCurrentOrDensityProfile(fromC)
		);
	}
	
	if(request.isFreeBoundary()) {
		result = kj::strTree(
			mv(result), "\n",
			"! --- Free boundary inputs ---\n",
			"MGRID_FILE = '", mgridFile.toNativeString(true), "'\n",
			"NZETA = ", request.getFreeBoundary().getVacuumField().getGrid().getNPhi(), "\n",
			"EXTCUR = 1\n"
		);
	}
	
	result = kj::strTree(
		"&INDATA\n", mv(result), "/\n"
	);
	
	return result.flatten();
}


Promise<void> writeMGridFile(kj::Path path, ComputedField::Reader cField) {
	return getActiveThread().dataService().download(cField.getData())
	.then([path = mv(path), grid = cField.getGrid()](LocalDataRef<Float64Tensor> data) {
		// File
		H5::H5File file(path.toWin32String(true).cStr(), H5F_ACC_TRUNC);
		
		// Scalars
		auto nR = grid.getNR();
		auto nZ = grid.getNZ();
		auto nPhi = grid.getNPhi();
		
		writeScalar(createDataSet<uint32_t>(file, "ir"), nR);
		writeScalar(createDataSet<uint32_t>(file, "jz"), nZ);
		writeScalar(createDataSet<uint32_t>(file, "kp"), nPhi);
		
		writeScalar(createDataSet<uint32_t>(file, "nfp"), grid.getNSym());
		writeScalar(createDataSet<double>(file, "rmin"), grid.getRMin());
		writeScalar(createDataSet<double>(file, "rmax"), grid.getRMax());
		writeScalar(createDataSet<double>(file, "zmin"), grid.getZMin());
		writeScalar(createDataSet<double>(file, "zmax"), grid.getZMax());
		writeScalar(createDataSet<uint32_t>(file, "nextcur"), 1);
		
		// Mgrid mode
		writeScalar<char>(createDataSet<char>(file, "mgrid_mode", {1}), 'H');
		
		// Coil groups
		writeArray<char>(createDataSet<char>(file, "coil_group", {1, 5}), {'C', 'O', 'I', 'L', '\0'});
		
		// Field data
		
		// Fortunately, we use exactly the same tensor shapes (phi, z, r) for the data
		// The only difference is that the fusionsc storage tensor uses {phi z r} as the last dimension
		// while makegrid stores the orientations separately. We therefore only need to demultiplex the
		// arrays point by point.
		{
			auto dataIn = data.get().getData();
			size_t nPoints = dataIn.size() / 3;
			
			auto bp = kj::heapArray<double>(nPoints);
			auto br = kj::heapArray<double>(nPoints);
			auto bz = kj::heapArray<double>(nPoints);
			for(auto i : kj::range(0, nPoints)) {
				bp[i] = dataIn[3 * i + 0];
				bz[i] = dataIn[3 * i + 1];
				br[i] = dataIn[3 * i + 2];
			}
			
			writeArray<double>(createDataSet<double>(file, "bp_001", {nPhi, nZ, nR}), bp);
			writeArray<double>(createDataSet<double>(file, "bz_001", {nPhi, nZ, nR}), bz);
			writeArray<double>(createDataSet<double>(file, "br_001", {nPhi, nZ, nR}), br);
		}
	});
}

VmecDriver::Client createVmecDriver(JobScheduler::Client scheduler, kj::PathPtr workDir) {
	return kj::heap<VmecDriverImpl>(mv(scheduler), *(kj::newDiskFilesystem()), workDir);
}

}