namespace fsc { namespace nudft {

template<unsigned int xdim, unsigned int ydim>
struct FourierPoint {
	double angles[xdim];
	double y[ydim];
};

template<unsigned int xdim, unsigned int ydim>
struct FourierMode {
	int coeffs[xdim];
	double cosCoeffs[ydim];
	double sinCoeffs[ydim];
};

template<unsigned int xdim, unsigned int ydim>
void calculateModes(kj::ArrayPtr<const FourierPoint<xdim, ydim>>, kj::ArrayPtr<FourierMode<xdim, ydim>>);

}}

// Implementation

namespace fsc { namespace nudft {

template<unsigned int xdim, unsigned int ydim>
void calculateModes(kj::ArrayPtr<const FourierPoint<xdim, ydim>> points, kj::ArrayPtr<FourierMode<xdim, ydim>> modes) {
	auto D = Eigen::Dynamic;
	
	using Mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
	using Vec = Eigen::Vector<double, Eigen::Dynamic>;
	
	//REMOVE THIS AFTER DBG
	//KJ_DBG
	// kj::FixedArray<Vec, 2> holder;
	KJ_DBG("NUDFT prepare", points.size(), modes.size());
		
	Mat A(points.size(), 2 * modes.size());
	
	#pragma omp parallel for
	for(int64_t iPoint = 0; iPoint < points.size(); ++iPoint) {
		for(auto iMode : kj::indices(modes)) {
			double angle = 0;
			
			for(unsigned int i = 0; i < xdim; ++i) {
				angle += modes[iMode].coeffs[i] * points[iPoint].angles[i];
			}
			
			A(iPoint, 2 * iMode) = std::sin(angle);
			A(iPoint, 2 * iMode + 1) = std::cos(angle);
		}
	}
		
	KJ_DBG("NUDFT reduce");
	
	Mat AtA = A.transpose() * A; 
		
	KJ_DBG("NUDFT cholesky");

	auto cholesky = AtA.ldlt();
	
	#pragma omp parallel for
	for(int iDim = 0; iDim < ydim; ++iDim) {
		// KJ_DBG(iDim);
		// We minimize (y - modeBasis * coeffs)**2
		
		// With A = modeBasis and x = coeffs
		// we minimize y^T y - y^T A x - x^T A^T y + x^T A^T A x
		//
		// 0 = -2 A^T y + 2 A^T A x
		// => A^T A x = A^T y
		
		KJ_DBG("NUDFT build RHS", iDim);
		
		Vec y(points.size());
		for(auto i : kj::indices(points))
			y[i] = points[i].y[iDim];
		
		Vec Aty = A.transpose() * y;
		
		// Perform Cholesky decomposition
		KJ_DBG("NUDFT solve");
		Vec x = cholesky.solve(Aty);
		
		// Check against reference
		/*Vec yOpt = A * x;
		for(auto iPoint : kj::indices(points)) {
			KJ_DBG(iPoint, y[iPoint], yOpt[iPoint]);
		Vec AtyOpt = A.transpose() * yOpt;
		
		for(auto i : kj::range(0, modes.size())) {
			auto& mode = modes[i];
			auto n = mode.coeffs[0];
			auto m = mode.coeffs[1];
			KJ_DBG("Sin", i, n, m, AtyOpt[2 * i + 0], Aty[2 * i + 0], x[2 * i + 0]);
			KJ_DBG("Cos", i, n, m, AtyOpt[2 * i + 1], Aty[2 * i + 1], x[2 * i + 1]);
		}*/
		KJ_DBG("NUDFT extract");
		
		for(auto iMode : kj::indices(modes)) {
			modes[iMode].sinCoeffs[iDim] = x[2 * iMode];
			modes[iMode].cosCoeffs[iDim] = x[2 * iMode + 1];
		}
		
		// holder[iDim] = yOpt;
	}
	
	/*for(auto i : kj::indices(points)) {
		const auto& p = points[i];
		double r = p.y[0];
		double z = p.y[1];
		double phi = p.angles[0];
		double theta = p.angles[1];
		double rOpt = holder[0][i];
		double zOpt = holder[1][i];
		KJ_DBG(phi, theta, r, rOpt, z, zOpt);
	}*/
}

}}