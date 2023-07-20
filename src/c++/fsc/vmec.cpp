#include "vmec.h"
#include "common.h"

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

//! Generates the input file for a VMEC request
kj::StringTree generateVmecInput(VmecRequest::Reader request) {
	auto sp = request.getStartingPoint();
	auto surfShape = sp.getRCos().getShape();
	auto nSurf = surfShape[0];
	auto n = surfShape[1];
	auto m = surfShape[2];
	auto period = sp.getPeriod();
	
	auto iota = request.getIota();
	auto rp = request.getRunParams();
	
	KJ_REQUIRE(request.getPhiEdge() != 0, "phiEdge must be provided");
	
	if(request.isFreeBoundary()) {
		KJ_UNIMPLEMENTED("Free boundary runs not supported");
	}
	
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
		makeMassProfile(request.getMassProfile())
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
	
	result = kj::strTree(
		"&INDATA\n", mv(result), "/\n"
	);
	
	return result;
}

}

struct VmecDriverImpl : public VmecDriver::Server {
	JobScheduler::Client scheduler;
	
	VmecDriverImpl(JobScheduler::Client scheduler) :
		scheduler(mv(scheduler))
	{}
};

VmecDriver::Client createVmecDriver(JobScheduler::Client scheduler) {
	return kj::heap<VmecDriverImpl>(mv(scheduler));
}

}