#include "vmec.h"

#include "common.h"
#include "tensor.h"
#include "data.h"

#include "hdf5.h"

#include "vmec-kernels.h"

#include "kernels/message.h"
#include "kernels/launch.h"

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

void validateProfile(VmecProfile::Reader p) {
	if(p.isSpline()) {
		auto s = p.getSpline();
		
		KJ_REQUIRE(s.getLocations().size() == s.getValues().size(), "Spline profile has mismatch between support points and values");
		
		if(s.getType() == VmecProfile::SplineType::AKIMA) {
			KJ_REQUIRE(s.getValues().size() >= 3, "Akima splines need at least 3 support points");
		} else if(s.getType() == VmecProfile::SplineType::CUBIC) {
			KJ_REQUIRE(s.getValues().size() >= 4, "Cubic splines need at least 4 support points");
		}
	}
}

kj::StringTree makeCurrentProfile(VmecProfile::Reader cp) {	
	validateProfile(cp);
	
	if(cp.isPowerSeries()) {
		return kj::strTree(
			"PCURR_TYPE = 'power_series_I'\n"
			"AC = ", fArray(cp.getPowerSeries()), "\n"
		);
	} else if(cp.isSpline()) {
		auto s = cp.getSpline();
		
		return kj::strTree(
			"PCURR_TYPE = '", decodeSplineType(s.getType()), "_I'\n"
			"ac_aux_s = ", fArray(s.getLocations()), "\n"
			"ac_aux_f = ", fArray(s.getValues()), "\n"
		);
	} else if(cp.isTwoPower()) {
		auto tp = cp.getTwoPower();
		
		return kj::strTree(
			"PCURR_TYPE = 'two_power'\n"
			"AC = ", tp.getBase(), " ", tp.getInner(), " ", tp.getOuter(), "\n"
		);
	}
	
	KJ_FAIL_REQUIRE("Unknown profile type", cp);
}

kj::StringTree makeCurrentDensityProfile(VmecProfile::Reader cp) {
	validateProfile(cp);
	
	if(cp.isPowerSeries()) {
		return kj::strTree(
			"PCURR_TYPE = 'power_series_I'\n"
			"AC = ", fArray(cp.getPowerSeries()), "\n"
		);
	} else if(cp.isSpline()) {
		auto s = cp.getSpline();
		
		return kj::strTree(
			"PCURR_TYPE = '", decodeSplineType(s.getType()), "_I'\n"
			"ac_aux_s = ", fArray(s.getLocations()), "\n"
			"ac_aux_f = ", fArray(s.getValues()), "\n"
		);
	} else if(cp.isTwoPower()) {
		auto tp = cp.getTwoPower();
		
		return kj::strTree(
			"PCURR_TYPE = 'two_power'\n"
			"AC = ", tp.getBase(), " ", tp.getInner(), " ", tp.getOuter(), "\n"
		);
	}
	
	KJ_FAIL_REQUIRE("Unknown profile type", cp);
}

kj::StringTree makeIotaProfile(VmecProfile::Reader cp) {	
	validateProfile(cp);
	
	if(cp.isPowerSeries()) {
		return kj::strTree(
			"PIOTA_TYPE = 'power_series'\n"
			"AI = ", fArray(cp.getPowerSeries()), "\n"
		);
	} else if(cp.isSpline()) {
		auto s = cp.getSpline();
		
		return kj::strTree(
			"PIOTA_TYPE = '", decodeSplineType(s.getType()), "'\n"
			"ai_aux_s = ", fArray(s.getLocations()), "\n"
			"ai_aux_f = ", fArray(s.getValues()), "\n"
		);
	}
	
	KJ_FAIL_REQUIRE("Unknown profile type", cp);
}

kj::StringTree makeMassProfile(VmecProfile::Reader mp) {
	validateProfile(mp);
	
	if(mp.isPowerSeries()) {
		return kj::strTree(
			"PMASS_TYPE = 'power_series'\n"
			"AM = ", fArray(mp.getPowerSeries()), "\n"
		);
	} else if(mp.isSpline()) {
		auto s = mp.getSpline();
		
		return kj::strTree(
			"PMASS_TYPE = '", decodeSplineType(s.getType()), "'\n"
			"am_aux_s = ", fArray(s.getLocations()), "\n"
			"am_aux_f = ", fArray(s.getValues()), "\n"
		);
	} else if(mp.isTwoPower()) {
		auto tp = mp.getTwoPower();
		
		return kj::strTree(
			"PMASS_TYPE = 'two_power'\n"
			"AM = ", tp.getBase(), " ", tp.getInner(), " ", tp.getOuter(), "\n"
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
	KJ_REQUIRE(nSurf >= 1);
	
	validateTensor(in.getRCos(), {nSurf, 2 * nTor + 1, mPol + 1});
	validateTensor(in.getZSin(), {nSurf, 2 * nTor + 1, mPol + 1});
	
	if(in.isNonSymmetric()) {
		validateTensor(in.getNonSymmetric().getRSin(), {nSurf, 2 * nTor + 1, mPol + 1});
		validateTensor(in.getNonSymmetric().getZCos(), {nSurf, 2 * nTor + 1, mPol + 1});
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
		rAxisCC.add(rCos(0, iTor, 0));
		zAxisCS.add(zSin(0, iTor, 0));
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
			rAxisCS.add(rCos(0, iTor, 0));
			zAxisCC.add(zSin(0, iTor, 0));
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
	
	for(auto iTorUnsigned : kj::range(0, 2 * nTor + 1)) {
		int64_t iTorSigned = iTorUnsigned;
		if(iTorSigned > nTor)
			iTorSigned -= 2 * nTor + 1;
	
		result = kj::strTree(
			mv(result),
			"! n = ", iTorSigned, "\n"
		);
		for(auto jPol : kj::range(0, mPol + 1)) {
			if(jPol == 0 && iTorSigned < 0) {
				KJ_REQUIRE(rCos(jPol, iTorUnsigned, iMax) == 0, "n<1, m=0 components must be 0", jPol, iTorSigned);
				KJ_REQUIRE(zSin(jPol, iTorUnsigned, iMax) == 0, "n<1, m=0 components must be 0", jPol, iTorSigned);
				continue;
			}
			
			result = kj::strTree(
				mv(result),
				"RBC(", iTorSigned, ", ", jPol, ") = ", rCos(jPol, iTorUnsigned, iMax), "  "
				"ZBS(", iTorSigned, ", ", jPol, ") = ", zSin(jPol, iTorUnsigned, iMax), "\n"
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
	
		for(auto iTorUnsigned : kj::range(0, 2 * nTor + 1)) {
			int64_t iTorSigned = iTorUnsigned;
			if(iTorSigned > nTor)
				iTorUnsigned -= 2 * nTor + 1;
			
			result = kj::strTree(
				mv(result),
				"! n = ", iTorSigned, "\n"
			);
			for(auto jPol : kj::range(0, mPol + 1)) {
				if(jPol == 0 && iTorSigned < 0) {
					KJ_REQUIRE(rSin(jPol, iTorUnsigned, iMax) == 0, "n<1, m=0 components must be 0", jPol, iTorSigned);
					KJ_REQUIRE(zCos(jPol, iTorUnsigned, iMax) == 0, "n<1, m=0 components must be 0", jPol, iTorSigned);
					continue;
				}
				
				result = kj::strTree(
					mv(result),
					"RBZ(", jPol, ", ", iTorSigned, ") = ", rSin(jPol, iTorUnsigned, iMax), "  "
					"ZBC(", jPol, ", ", iTorSigned, ") = ", zCos(jPol, iTorUnsigned, iMax), "\n"
				);
			}
		}
	}
	
	return result;
}

struct VmecRun {
	Own<JobDir> workDir;
	VmecRequest::Reader in;
	VmecResponse::Builder out;
	
	VmecRun(VmecRequest::Reader in, VmecResponse::Builder out, Own<JobDir> d) :
		workDir(mv(d)),
		in(in), out(out)
	{}
	
	Promise<void> run(JobLauncher& launcher) {
		auto mgridPath = workDir -> absPath.append("vacField.nc");
		KJ_DBG(mgridPath);
		
		Promise<void> prepareInput = READY_NOW;
		
		if(in.isFreeBoundary()) {	
			// Write the mgrid file
			prepareInput = writeMGridFile(mgridPath, in.getFreeBoundary().getVacuumField());
		}
		
		// Prepare the VMEC input file
		auto inputFile = workDir -> dir -> openFile(kj::Path("input.inputFile"), kj::WriteMode::CREATE);
		
		auto inputString = generateVmecInput(in, mgridPath);
		inputFile -> writeAll(inputString);
		out.setInputFile(inputString);
		
		return prepareInput.then([this, &launcher]() {
			// Launch the VMEC code
			JobRequest req;
			req.command = kj::str("xvmec2000");
			req.setArguments({"inputFile"});
			req.workDir = workDir -> absPath.clone();
			
			auto job = launcher.launch(mv(req));
			
			// Extract output streams		
			auto streams = job.attachRequest().sendForPipeline();
			
			auto readStdout = streams.getStdout().readAllStringRequest().send()
			.then([this](auto resp) mutable {
				out.setStdout(resp.getText());
			});
			
			auto readStderr = streams.getStderr().readAllStringRequest().send()
			.then([this](auto resp) mutable {
				out.setStderr(resp.getText());
			});
			
			// Wait for job to finish
			auto actualJob = job.whenCompletedRequest().send()
			.then(
				// Success
				[this](auto wcResponse) {
					auto& ds = getActiveThread().dataService();
					
					Temporary<VmecResult> tmp;
					
					// Read in wout.nc
					tmp.setWoutNc(ds.publishFile(
						*workDir -> dir -> openFile(kj::Path("wout_inputFile.nc")),
						true
					));
					interpretOutputFile(workDir -> absPath.append("wout_inputFile.nc"), tmp);
					
					out.getResult().setOk(ds.publish(tmp.asReader()));
					
					// TODO: Parse file
					KJ_LOG(WARNING, "Incomplete code: No parsing of VMEC result");
				}
			)
			.catch_(
				// Failure
				[this](kj::Exception&& e) {
					out.getResult().setFailed(kj::str("VMEC run failed - ", e));
				} 
			);
			
			auto pBuilder = kj::heapArrayBuilder<Promise<void>>(3);
			pBuilder.add(mv(actualJob));
			pBuilder.add(mv(readStdout));
			pBuilder.add(mv(readStderr));
			return kj::joinPromises(pBuilder.finish());
		});
	}
	
	
};

struct VmecDriverImpl : public VmecDriver::Server {
	Own<JobLauncher> launcher;
	Own<DeviceBase> device;
	
	VmecDriverImpl(Own<DeviceBase> dev, Own<JobLauncher> l) :
		launcher(mv(l)),
		device(mv(dev))
	{}
	
	Promise<void> run(RunContext ctx) override {
		auto run = heapHeld<VmecRun>(ctx.getParams(), ctx.initResults(), launcher -> createDir());
		
		// Wrap inside evalLater so that we don't get the bogus "unwind across heapHeld" warning
		// when the inner part throws.
		return kj::evalLater([this, run]() mutable {
			return run -> run(*launcher);
		})
		.attach(run.x());
	}
	
	Promise<void> computePositions(ComputePositionsContext ctx) override {
		auto params = ctx.getParams();
		auto spt = params.getSPhiTheta();
		
		validateTensor(spt);
		KJ_REQUIRE(spt.getShape().size() >= 1);
		KJ_REQUIRE(spt.getShape()[0] == 3);
		
		auto output = ctx.initResults().getPhiZR();
		output.setShape(spt.getShape());
		
		auto surf = params.getSurfaces();
		validateSurfaces(surf);
		
		auto mapping = FSC_MAP_BUILDER(
			fsc, VmecKernelComm, MapNewMessage(), *device, true
		);
		
		auto host = mapping -> getHost();
		host.setSurfaces(surf);
		host.setSpt(spt.getData());
		host.initPzr(spt.getData().size());
		
		// Update segment structure
		mapping -> updateStructureOnDevice();
		
		// Fill in results
		Promise<void> result = FSC_LAUNCH_KERNEL(
			computeSurfaceKernel,
			
			*device,
			spt.getData().size() / 3,
			mapping
		);
		
		return result.then([mapping = mv(mapping), host, ctx]() mutable {
			ctx.getResults().getPhiZR().setData(host.getPzr());
		});
	}
	
	Promise<void> invertPositions(InvertPositionsContext ctx) override {
		auto params = ctx.getParams();
		auto pzr = params.getPhiZR();
		
		validateTensor(pzr);
		KJ_REQUIRE(pzr.getShape().size() >= 1);
		KJ_REQUIRE(pzr.getShape()[0] == 3);
		
		auto output = ctx.initResults().getSPhiTheta();
		output.setShape(pzr.getShape());
		
		auto surf = params.getSurfaces();
		validateSurfaces(surf);
		
		auto mapping = FSC_MAP_BUILDER(
			fsc, VmecKernelComm, MapNewMessage(), *device, true
		);
		
		auto host = mapping -> getHost();
		host.setSurfaces(surf);
		host.setPzr(pzr.getData());
		host.initSpt(pzr.getData().size());
		
		// Update segment structure
		mapping -> updateStructureOnDevice();
		
		// Fill in results
		Promise<void> result = FSC_LAUNCH_KERNEL(
			invertSurfaceKernel,
			
			*device,
			pzr.getData().size() / 3,
			mapping
		);
		
		return result.then([mapping = mv(mapping), host, ctx]() mutable {
			ctx.getResults().getSPhiTheta().setData(host.getSpt());
		});
	}
};

}

//! Generates the input file for a VMEC request
kj::String generateVmecInput(VmecRequest::Reader request, kj::PathPtr mgridFile) {
	auto sp = request.getStartingPoint();
	auto iota = request.getIota();
	auto runParams = request.getRunParams();
	
	KJ_REQUIRE(request.getPhiEdge() != 0, "phiEdge must be provided");
	KJ_REQUIRE(request.getNTor() != 0, "Maximum toroidal Fourier number must be provided");
	KJ_REQUIRE(request.getMPol() != 0, "Maximum poloidal Fourier number must be provided ");
	
	KJ_REQUIRE(request.getMPol() >= request.getStartingPoint().getMPol(), "Run data must be able to house all input poloidal modes");
	KJ_REQUIRE(request.getNTor() >= request.getStartingPoint().getNTor(), "Run data must be able to house all input toroidal modes");
	
	// Compute reasonable defaults for grid
	uint32_t nGridTor = runParams.getNGridToroidal();
	if(nGridTor == 0) {
		if(request.isFreeBoundary()) {
			nGridTor = request.getFreeBoundary().getVacuumField().getGrid().getNPhi();
		} else {
			nGridTor = 2 * request.getNTor() + 4;
		}
	}
	
	uint32_t nGridPol = runParams.getNGridToroidal();
	uint32_t nGridPolMin = 2 * (request.getMPol() + 1) + 6;
	
	if(nGridPol == 0)
		nGridPol = nGridPolMin;
	
	KJ_REQUIRE(nGridPol >= nGridPolMin, "Poloidal grid resolution to small");
	
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
		"NTOR = ", request.getNTor(), "\n"
		"MPOL = ", request.getMPol() + 1, "\n"
		"NFP = ", sp.getPeriod(), "\n"
		"NTHETA = ", nGridPol, "\n"
		"NZETA = ", nGridTor, "\n"
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
		KJ_REQUIRE(request.getFreeBoundary().getVacuumField().getGrid().getNSym() == sp.getPeriod(), "Vacuum field symmetry must match field period of start surfaces");
		KJ_REQUIRE(nGridTor == request.getFreeBoundary().getVacuumField().getGrid().getNPhi(), "Toroidal grid dimension of VMEC and vacuum file must match");
		
		result = kj::strTree(
			mv(result), "\n",
			"! --- Free boundary inputs ---\n",
			"MGRID_FILE = '", mgridFile.toNativeString(true), "'\n",
			"EXTCUR = 1\n"
		);
	}
	
	result = kj::strTree(
		"&INDATA\n", mv(result), "/\n"
	);
	
	return result.flatten();
}

void interpretOutputFile(kj::PathPtr path, VmecResult::Builder out) {
	H5::H5File file(path.toNativeString(true).cStr(), 0);
	
	uint32_t nTor = readScalar<uint32_t>(file.openDataSet("ntor"));
	uint32_t mPol = readScalar<uint32_t>(file.openDataSet("mpol"));
	uint32_t nFP  = readScalar<uint32_t>(file.openDataSet("nfp"));
	
	auto surf = out.initSurfaces();
	
	surf.setNTor(nTor);
	surf.setMPol(mPol - 1);
	surf.setPeriod(nFP);
	
	auto arrayDims = getDimensions(file.openDataSet("rmnc"));
	int64_t nSurf = arrayDims[0].length;
	size_t nPerSurf = arrayDims[1].length;
	
	KJ_REQUIRE(nPerSurf == nTor + 1 + (mPol - 1) * (2 * nTor + 1), "Unexpected output format");
	
	// Note:
	//
	// The data in these arrays is (per surface) ordered as
	// m/n = 0/0, ..., 0/nTor, 1/-nTor, ..., 1/ntor, 2/-nTor, ..., ..., mTor/nTor
	// This means when reading we can simply take our indexing scheme (which goes
	// from 0 to nTor, then starts from -nTor to -1) and write to that. We just need
	// to increment the m accordingly.
	
	int64_t offset = 0;
	
	Tensor<double, 3> rmncT(mPol, 2 * nTor + 1, nSurf);
	Tensor<double, 3> zmnsT(mPol, 2 * nTor + 1, nSurf);
	
	auto rmnc = readArray<double>(file.openDataSet("rmnc"));
	auto zmns = readArray<double>(file.openDataSet("zmns"));
	
	for(auto iPol : kj::range(0, mPol)) {
		for(auto iLinear : kj::range(0, 2 * nTor + 1)) {
			// No negative n for m == 0
			if(iPol == 0 && iLinear >= nTor + 1)
				break;
			
			// Here we use the linear index shift above
			size_t iTor = offset % (2 * nTor + 1);
			for(auto iSurf : kj::range(0, nSurf)) {
				rmncT(iPol, iTor, iSurf) = rmnc[iSurf * nPerSurf + offset];
				zmnsT(iPol, iTor, iSurf) = zmns[iSurf * nPerSurf + offset];
			}
			
			++offset;
		}
	}
	
	writeTensor(rmncT, surf.initRCos());
	writeTensor(zmnsT, surf.initZSin());
	
	if(file.nameExists("zmnc")) {
		// We also have non-symmetric components
		// Extract these as well
		offset = 0;
		
		Tensor<double, 3> rmnsT(mPol, 2 * nTor + 1, nSurf);
		Tensor<double, 3> zmncT(mPol, 2 * nTor + 1, nSurf);
		
		auto rmns = readArray<double>(file.openDataSet("rmns"));
		auto zmnc = readArray<double>(file.openDataSet("zmnc"));
		
		for(auto iPol : kj::range(0, mPol)) {
			for(auto iLinear : kj::range(0, 2 * nTor + 1)) {
				// No negative n for m == 0
				if(iPol == 0 && iLinear >= nTor + 1)
					break;
				
				// Here we use the linear index shift above
				size_t iTor = offset % (2 * nTor + 1);
				for(auto iSurf : kj::range(0, nSurf)) {
					rmnsT(iPol, iTor, iSurf) = rmns[iSurf * nPerSurf + offset];
					zmncT(iPol, iTor, iSurf) = zmnc[iSurf * nPerSurf + offset];
				}
				
				++offset;
			}
		}
		
		auto nonsym = surf.initNonSymmetric();
		writeTensor(rmnsT, nonsym.initRSin());
		writeTensor(zmncT, nonsym.initZCos());
	}
}

Promise<void> writeMGridFile(kj::PathPtr path, ComputedField::Reader cField) {
	return getActiveThread().dataService().download(cField.getData())
	.then([path = path.clone(), grid = cField.getGrid()](LocalDataRef<Float64Tensor> data) {
		// File
		KJ_DBG(path.toNativeString(true).cStr());
		H5::H5File file(path.toNativeString(true).cStr(), H5F_ACC_TRUNC);
		KJ_DBG(path.toNativeString(true).cStr());
		
		// Scalars
		auto nR = grid.getNR();
		auto nZ = grid.getNZ();
		auto nPhi = grid.getNPhi();
		
		writeScalar(createDataSet<int32_t>(file, "ir"), nR);
		writeScalar(createDataSet<int32_t>(file, "jz"), nZ);
		writeScalar(createDataSet<int32_t>(file, "kp"), nPhi);
		
		writeScalar(createDataSet<int32_t>(file, "nfp"), grid.getNSym());
		writeScalar(createDataSet<double>(file, "rmin"), grid.getRMin());
		writeScalar(createDataSet<double>(file, "rmax"), grid.getRMax());
		writeScalar(createDataSet<double>(file, "zmin"), grid.getZMin());
		writeScalar(createDataSet<double>(file, "zmax"), grid.getZMax());
		writeScalar(createDataSet<int32_t>(file, "nextcur"), 1);
		
		// Named dimensions
		auto dimR = createDimension<double>(file, "r", nR, true);
		auto dimZ = createDimension<double>(file, "z", nZ, true);
		auto dimP = createDimension<double>(file, "phi", nPhi, true);
		
		auto dim1 = createDimension<double>(file, "dim001", 1, true);
		auto dim4 = createDimension<double>(file, "dim004", 4, true);
		
		// Mgrid mode
		writeScalar<char>(createDataSet<char>(file, "mgrid_mode", {dim1}), 'R');
		
		// Coil groups
		writeArray<char>(createDataSet<char>(file, "coil_group", {dim4}), {'C', 'O', 'I', 'L'});
		
		// External currents
		writeArray<double>(createDataSet<double>(file, "raw_coil_cur", {dim1}), {1});
		
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
			
			writeArray<double>(createDataSet<double>(file, "bp_001", {dimP, dimZ, dimR}), bp);
			writeArray<double>(createDataSet<double>(file, "bz_001", {dimP, dimZ, dimR}), bz);
			writeArray<double>(createDataSet<double>(file, "br_001", {dimP, dimZ, dimR}), br);
		}
		
		file.close();
	});
}

VmecDriver::Client createVmecDriver(Own<DeviceBase>&& dev, Own<JobLauncher>&& scheduler) {
	return kj::heap<VmecDriverImpl>(mv(dev), mv(scheduler));
}

}