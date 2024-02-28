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
	
	for(unsigned int iDim : kj::range(0, ydim)) {
		// We minimize (y - modeBasis * coeffs)**2
		
		// With A = modeBasis and x = coeffs
		// we minimize y^T y - y^T A x - x^T A^T y + x^T A^T A x
		//
		// 0 = -2 A^T y + 2 A^T A x
		// => A^T A x = A^T y
		
		Vec y(points.size());
		for(auto i : kj::indices(points))
			y[i] = points[i].y[iDim];
		
		Mat A(points.size(), 2 * modes.size());
		for(auto iPoint : kj::indices(points)) {
			for(auto iMode : kj::indices(modes)) {
				double angle = 0;
				
				for(unsigned int i = 0; i < xdim; ++i) {
					angle += modes[iMode].coeffs[i] * points[iPoint].angles[i];
				}
				
				A(iPoint, 2 * iMode) = std::sin(angle);
				A(iPoint, 2 * iMode + 1) = std::cos(angle);
			}
		}
		
		Mat AtA = A.transpose() * A;
		Vec Aty = A.transpose() * y;
		
		// Perform Cholesky decomposition
		auto cholesky = AtA.ldlt();
		Vec x = cholesky.solve(Aty);
		
		for(auto iMode : kj::indices(modes)) {
			modes[iMode].sinCoeffs[iDim] = x[2 * iMode];
			modes[iMode].cosCoeffs[iDim] = x[2 * iMode + 1];
		}
	}
}

}}