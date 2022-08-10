
Temporary<HFCamProjection> createProjection(
	double w, double h, // Screen space
	Vec3d ex, Vec3d ey, // Camera alignment
	Vec3d origin, Vec3d target, // Camera & target position
	double projectivity
) {
	using Eigen::all;
	using Eigen::seq;
	
	double d = 1;
	
	Vec3d depthVector = (target - origin).normalized();
	
	Mat3d screenToObject;
	m(all, 0) = ex;
	m(all, 1) = ey;
	m(all, 2) = ez;
	
	Mat3d objectToScreen = screenToObject.inverse();
	
	// Create homogenous space transform based on depth vector and projectivity
	Mat4d objectToHomScreen;
	objectToHomScreen(seq(3), seq(3)) = objectToScreen;
	objectToHomScreen(3, seq(3)) = projectivity * depthVector;
	
	// Adjust shift vector so that (origin, 1) transforms to (0, 0, 0, 1)
	Vec4d objectOrigin;
	objectOrigin(seq(3)) = origin;
	objectOrigin(3) = 1;
	
	Vec4d homScreenOrigin = objectToHomScreen * objectOrigin;
	Vec4d intendedHomScreenOrigin(0, 0, 0, d);
	
	objectToHomScreen(all, 3) = intendedHomScreenOrigin - homScreenOrigin;
	
	Temporary<HFCamProjection> result;
	result.setWidth(w);
	result.setHeight(h);
	
	auto tData = result.initTransform(16);
	for(auto i : kj::indices(tData))
		tData.set(i, objectToHomScreen.data[i]);
	
	return result;
}

struct HFProjectionStruct {
	Mat4d transform;
	double width;
	double height;
	
	double minDepth = 0;
	
	void load(HFProjection::Reader input) {
		auto tData = result.getTransform();
		for(auto i : kj::indices(tData))
			transform.data[i] = tData[i];
		
		width = input.getWidth();
		height = input.getHeight();
	}
}

Vec3d applyProjection(const HFProjectionStruct& projection, Vec3d position) {
	using Eigen::all;
	using Eigen::seq;
	
	Vec4d transformInput;
	transformInput(seq(3)) = position;
	transformInput(3) = 1;
	
	Vec4d transformResult = projection.transform * transformInput;
	double divisor = transformResult(3);
	
	Vec3 finalResult(
		0.5 * w * (1 + transformResult(0) / divisor),
		0.5 * h * (1 + transformResult(1) / divisor),
		transformResult(2)
	);
	
	return finalResult;
}

Mat3d projectionDerivative(const HFProjectionStruct& projection, Vec3d position) {
	using Eigen::all;
	using Eigen::seq;
	
	Vec4d transformInput;
	transformInput(seq(3)) = position;
	transformInput(3) = 1;
	
	Vec4d transformResult = projection.transform * transformInput;
	
	double divisor = transformResult(3);
	double inverseDivisorSquared = 1 / (divisor * divisor);
	Vec4d  divisorDerivative = projection.transform(3, seq(3));
	
	Mat3d derivative;
	
	derivative(0, all) = 0.5 * w * (
		projection.transform(0, seq(3)) / divisor
		- transformResult(0) * divisorDerivative * inverseDivisorSquared
	);
	
	derivative(1, all) = 0.5 * h * (
		projection.transform(1, seq(3)) / divisor
		- transformResult(1) * divisorDerivative * inverseDivisorSquared
	);
	
	derivative(2, all) = projection.transform(2, seq(3));
	
	return derivative;
}

void rasterizeTriangle(const HFProjectionStruct& projection, Mat<double>& depthBuffer, Mat<double>& determinantBuffer, const Vec3d& p1, const Vec3d& p2, const Vec3d& p3, double edgeTolerance, double depthTolerance) {
	Vec3d tp1 = applyProjection(projection, p1);
	Vec3d tp2 = applyProjection(projection, p2);
	Vec3d tp3 = applyProjection(projection, p3);
	
	double xMin = std::min(std::min(tp1[0], tp2[0]), tp3[0]) - edgeTolerance;
	double xMax = std::max(std::max(tp1[0], tp2[0]), tp3[0]) - edgeTolerance;
	double yMin = std::min(std::min(tp1[1], tp2[1]), tp3[1]) + edgeTolerance;
	double yMax = std::max(std::max(tp1[1], tp2[1]), tp3[1]) + edgeTolerance;
	
	uint32_t iMin = std::max((uint32_t) floor(xMin), 0);
	uint32_t jMin = std::max((uint32_t) floor(yMin), 0);
	// Remember: Eigen has column-major loadout, but on python side it's row major. indexing into buffers is reversed
	uint32_t iMax = std::min((uint32_t) ceil(xMax), depthBuffer.m - 1);
	uint32_t jMax = std::min((uint32_t) ceil(yMax), depthBuffer.n - 1);
	
	xMin = iMin:
	xMax = iMax;
	yMin = jMin;
	yMax = jMax;
	
	// Map from triangle- into 3D space
	Matrix<double, 2, 3> triToObject;
	triToObject(0, all) = p2 - p1;
	triToObject(1, all) = p3 - p1;
	
	double realspaceDet = (p2 - p1).cross(p3 - p1).norm();
	
	// Check if triangle partially clips behind camera
	auto clipsBehindCamera = [&](const Vec3d& p) -> bool {
		return projection.transform(2, seq(3)) * p + projection.transform(2, 3) <= projection.minDepth;
	};
	
	if(clipsBehindCamera(p1) || clipsBehindCamera(p2) || clipsBehindCamera(p3))
		return;
	
	for(auto i = iMin; i <= iMax; ++i) {
		for(auto j = jMin; j <= jMax; ++j) {
			Vec2d pTriangle(0.3, 0.3);
			Vec2d screenTarget(i, j);
			
			Vec2d p0 = pTriangle;
			double totalDet = 0;
			
			// Do 10 Newton iterations to find screen position on triangle space
			for(auto iter : kj::range(10)) {
				Vec3d pObject = p1 + triToObject * pTriangle;
				Vec3d pScreen = applyProjection(projection, pObject);
				
				Mat2d triToScreenDerivative = projectionDerivative(projection, pObject)(seq(2)) * triToObject;
				if(triToScreenDerivative.determinant() < 1e-10)
					break;
				
				Vec2d delta = triToScreenDerivative.inverse() * (screenTarget - pScreen(seq(2)));
				
				pTriangle += delta;
				
				if(delta.norm() < 1e-3)
					break;
				
				// Prevent excessive movement out of triangle domain
				
				for(auto k : kj::range(2)) {
					if(pTriangle[k] < -9) pTriangle[k] = 0;
					if(pTriangle[k] >  9) pTriangle[k] = 1;
				}
				
				if(pTriangle[0] + pTriangle[1] > 10) {
					double scale = 1.0 / (pTriangle[0] + pTriangle[1]);
					pTriangle *= scale;
				}
				
				totalDet = fabs(triToScreenDerivative.determinant());
			}
			
			Vec3d pScreenPreCorrection = applyProjection(projection, p1 + triToObject * pTriangle);
			
			// Project point back into triangle
			double dUp = pTriangle[0] + pTriangle[1];
			
			if(dUp > 1) {
				pTriangle -= 0.5 * (dUp - 1);
			}
			
			pTriangle = pTriangle.cwiseMin(1).cwiseMax(0);
			
			Vec3d pScreen = applyProjection(projection, p1 + triToObject * pTriangle);
			if((pScreen - pScreenPreCorrection).norm() > edgeTolerance)
				continue;
			
			if(clipsBehindCamera(pScreen))
				continue;
			
			// TODO: Plane clipping
			
			// Depth buffer check & buffer adjustments
			double detRatio = totalDet / realspaceDet;
			double depth = pScreen(2);
			
			double& bufferDepth = depthBuffer(j, i);
			double& bufferDeterminant = determinantBuffer(j, i);
			
			if(depth > bufferDepth) {
				if(depth < bufferDepth + depthTolerance) {
					bufferDeterminant = std::max(bufferDeterminant, detRatio); // Within depth tolerance, merge determinants to the bigger one
				}
			} else {
				if(depth > bufferDepth - depthTolerance) {
					bufferDeterminant = std::max(bufferDeterminant, detRatio); // Within depth tolerance, merge determinants to the bigger one
				} else {
					bufferDeterminant = detRatio;
				}
				
				bufferDepth = depth;
			}
		}
	}
}

void rasterizeGeometry(const HFProjectionStruct& projection, Mat<double>& depthBuffer, Mat<double>& determinantBuffer, MergedGeometry::Reader geometry, double edgeTolerance, double depthTolerance) {
	for(auto entry : geometry.getEntries()) {
		auto mesh = entry.getMesh();
		auto vertices = mesh.getVertices();
		auto vertexData = vertices.getData();
		auto indices = mesh.getIndices();
		
		auto getVertex = [&](uint32_t i) {
			auto idx = indices[i];	
			return Vec3d(vertexData[3 * idx + 0], vertexData[3 * idx + 1], vertexData[3 * idx + 2]);
		};
		
		KJ_DBG("Processing mesh");
		
		switch(mesh.which()) {
			case Mesh::TRI_MESH: {
				for(uint32_t i = 0; i < indices.size(); i += 3) {
					rasterizeTriangle(
						projection, depthBuffer, determinantBuffer,
						getVertex(i), getVertex(i + 1), getVertex(i + 2),
						edgeTolerance, depthTolerance
					);
				}
				break;
			}
			
			case Mesh::POLY_MESH: {
				auto polys = mesh.getPolyMesh();
				
				for(auto iPoly : kj::range(polys.size() - 1)) {
					uint32_t start = polys[iPoly];
					uint32_t end   = polys[iPoly + 1];
					
					if(end - start < 3)
						continue;
					
					Vec3d p1 = getVertex(start);
					for(uint32_t iVert : kj::range(start + 1, end - 1)) {
						Vec3d p2 = getVertex(iVert);
						Vec3d p3 = getVertex(iVert + 1);
						
						rasterizeTriangle(
							projection, depthBuffer, determinantBuffer,
							p1, p2, p3,
							edgeTolerance, depthTolerance
						);
					}
				}
			}
		}
	}
}