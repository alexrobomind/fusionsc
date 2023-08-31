#include "tensor.h"
#include "data.h"
#include "geometry.h"

#include "hfcam.h"

#include <fsc/hfcam.capnp.h>
#include <fsc/geometry.capnp.h>

#include <iostream>
#include <limits>


namespace fsc { namespace {
	
Temporary<HFCamProjection> createProjection(
	uint32_t w, uint32_t h, // Screen space
	Vec3d ex, Vec3d ey, // Camera alignment
	Vec3d origin, Vec3d target, // Camera & target position
	double invScaling, double projectivity, // Projection scaling formula. w = scaling + projectivity * <target - origin, x> / |target - origin|
	HFCamProjection::Builder result
) {
	using Eigen::all;
	using Eigen::seq;
	auto xyz = seq(0, 2);
	
	Vec3d depthVector = (target - origin).normalized();
	
	Mat3d screenToObject;
	screenToObject(all, 0) = ex;
	screenToObject(all, 1) = ey;
	screenToObject(all, 2) = depthVector;
	
	Mat3d objectToScreen = screenToObject.inverse();
	
	// Create homogenous space transform based on depth vector and projectivity
	Mat4d objectToHomScreen;
	objectToHomScreen(xyz, xyz) = objectToScreen;
	objectToHomScreen(3, xyz) = projectivity * depthVector;
	objectToHomScreen(all, 3).setZero();
	
	// Adjust shift vector so that (origin, 1) transforms to (0, 0, 0, invScaling)
	Vec4d objectOrigin;
	objectOrigin(xyz) = origin;
	objectOrigin(3) = 1;
	
	Vec4d homScreenOrigin = objectToHomScreen * objectOrigin;
	Vec4d intendedHomScreenOrigin(0, 0, 0, invScaling);
	
	objectToHomScreen(all, 3) = intendedHomScreenOrigin - homScreenOrigin;
	
	// std::cout << objectToHomScreen << std::endl;
	
	result.setWidth(w);
	result.setHeight(h);
	result.setTransform(kj::arrayPtr(objectToHomScreen.data(), 16));
	
	return result;
}
	
Temporary<HFCamProjection> toroidalProjection(
	uint32_t w, uint32_t h, // Screen space
	double phi, double rTarget, double zTarget,
	double inclination, double horzInclination, double distance,
	double viewportHeight, double fieldOfView,
	HFCamProjection::Builder result
) {
	using std::sin;
	using std::cos;
	
	double phiO = phi + horzInclination;
	
	Vec3d camEx(-sin(phiO), cos(phiO), 0);	
	Vec3d camEy(cos(phiO) * -sin(inclination), sin(phiO) * -sin(inclination), cos(inclination));
	
	// Adjust ration between ex and ey to be w/h ratio
	camEx *= ((double) w) / h;
		
	Vec3d target(cos(phi) * rTarget, sin(phi) * rTarget, zTarget);
	Vec3d origin = target + distance * Vec3d(cos(phiO) * cos(inclination), sin(phiO) * cos(inclination), sin(inclination));
	
	double invScaling = 0.5 * viewportHeight;
	double projectivity = invScaling * sin(fieldOfView) / cos(fieldOfView);
	
	return createProjection(
		w, h,
		camEx, camEy,
		origin, target,
		invScaling, projectivity,
		result
	);
}

struct HFProjectionStruct {
	Mat4d transform;
	double width;
	double height;
	
	double minDepth = 0;
	
	void load(HFCamProjection::Reader input) {
		auto tData = input.getTransform();
		for(auto i : kj::range(0, 16))
			transform.data()[i] = tData[i];
		
		width = input.getWidth();
		height = input.getHeight();
	}
};

Vec3d applyProjection(const HFProjectionStruct& projection, Vec3d position) {
	using Eigen::all;
	using Eigen::seq;
	auto xyz = seq(0, 2);
	
	Vec4d transformInput;
	transformInput(xyz) = position;
	transformInput(3) = 1;
	
	Vec4d transformResult = projection.transform * transformInput;
	double divisor = transformResult(3);
	
	Vec3d finalResult(
		0.5 * projection.width * (1 + transformResult(0) / divisor),
		0.5 * projection.height * (1 + transformResult(1) / divisor),
		transformResult(2)
	);
		
	return finalResult;
}

Mat3d projectionDerivative(const HFProjectionStruct& projection, Vec3d position) {
	using Eigen::all;
	using Eigen::seq;
	auto xyz = seq(0, 2);
	
	Vec4d transformInput;
	transformInput(xyz) = position;
	transformInput(3) = 1;
	
	Vec4d transformResult = projection.transform * transformInput;
	
	double divisor = transformResult(3);
	double inverseDivisorSquared = 1 / (divisor * divisor);
	Vec3d  divisorDerivative = projection.transform(3, xyz);
	
	Mat3d derivative;
	
	derivative(0, all) = 0.5 * projection.width * (
		projection.transform(0, xyz).transpose() / divisor
		- transformResult[0] * divisorDerivative * inverseDivisorSquared
	);
	
	derivative(1, all) = 0.5 * projection.height * (
		projection.transform(1, xyz).transpose() / divisor
		- transformResult[1] * divisorDerivative * inverseDivisorSquared
	);
	
	derivative(2, all) = projection.transform(2, xyz);
	
	return derivative;
}

void rasterizeTriangle(const HFProjectionStruct& projection, Eigen::MatrixXd& depthBuffer, Eigen::MatrixXd& determinantBuffer, const Vec3d& p1, const Vec3d& p2, const Vec3d& p3, double edgeTolerance, double depthTolerance) {
	using Eigen::all;
	using Eigen::seq;
	auto xyz = seq(0, 2);
	
	Vec3d tp1 = applyProjection(projection, p1);
	Vec3d tp2 = applyProjection(projection, p2);
	Vec3d tp3 = applyProjection(projection, p3);
	
	/*std::cout << "Triangle" << std::endl;
	std::cout << p1.transpose() << " -> " << tp1.transpose() << std::endl;
	std::cout << p2.transpose() << " -> " << tp2.transpose() << std::endl;
	std::cout << p3.transpose() << " -> " << tp3.transpose() << std::endl;*/
	
	double xMin = std::min(std::min(tp1[0], tp2[0]), tp3[0]) - edgeTolerance;
	double xMax = std::max(std::max(tp1[0], tp2[0]), tp3[0]) + edgeTolerance;
	double yMin = std::min(std::min(tp1[1], tp2[1]), tp3[1]) - edgeTolerance;
	double yMax = std::max(std::max(tp1[1], tp2[1]), tp3[1]) + edgeTolerance;
	
	if(yMax < 0 || xMax < 0)
		return;
		
	int32_t iMin = std::max((int32_t) floor(xMin), (int32_t) 0);
	int32_t jMin = std::max((int32_t) floor(yMin), (int32_t) 0);
	// Remember: Eigen has column-major loadout, but on python side it's row major. indexing into buffers is reversed
	int32_t iMax = std::min((int32_t) ceil(xMax), (int32_t) depthBuffer.cols() - 1);
	int32_t jMax = std::min((int32_t) ceil(yMax), (int32_t) depthBuffer.rows() - 1);
	
	xMin = iMin;
	xMax = iMax;
	yMin = jMin;
	yMax = jMax;
	
	// Map from triangle- into 3D space
	Eigen::Matrix<double, 3, 2> triToObject;
	triToObject(all, 0) = p2 - p1;
	triToObject(all, 1) = p3 - p1;
	
	double realspaceDet = (p2 - p1).cross(p3 - p1).norm();
	
	if(iMin > iMax || jMin > jMax)
		return;
	
	// Check if triangle partially clips behind camera
	auto clipsBehindCamera = [&](const Vec3d& p) -> bool {
		return projection.transform(2, seq(0, 2)) * p + projection.transform(2, 3) <= projection.minDepth;
	};
	
	if(clipsBehindCamera(p1) || clipsBehindCamera(p2) || clipsBehindCamera(p3))
		return;
	
	for(auto i = iMin; i <= iMax; ++i) {
		for(auto j = jMin; j <= jMax; ++j) {
			Vec2d pTriangle(0.3, 0.3);
			Vec2d screenTarget(i, j);
			
			Vec2d p0 = pTriangle;
			double totalDet = 0;
			
			// KJ_DBG(i, j);
			
			// Do 10 Newton iterations to find screen position on triangle space
			for(auto iter : kj::range(0, 10)) {
				Vec3d pObject = p1 + triToObject * pTriangle;
				Vec3d pScreen = applyProjection(projection, pObject);
				
				Mat2d triToScreenDerivative = projectionDerivative(projection, pObject)(seq(0,1), all) * triToObject;
				
				Vec2d delta = triToScreenDerivative.inverse() * (screenTarget - pScreen(seq(0,1)));
				
				pTriangle += delta;
				
				// Prevent excessive movement out of triangle domain
				
				for(auto k : kj::range(0, 2)) {
					if(pTriangle[k] < -9) pTriangle[k] = 0;
					if(pTriangle[k] >  9) pTriangle[k] = 1;
				}
				
				if(pTriangle[0] + pTriangle[1] > 10) {
					double scale = 1.0 / (pTriangle[0] + pTriangle[1]);
					pTriangle *= scale;
				}
				
				totalDet = fabs(triToScreenDerivative.determinant());
				
				//if(delta.norm() < 1e-3)
				//	break;
			}
			
			Vec3d pScreenPreCorrection = applyProjection(projection, p1 + triToObject * pTriangle);
			
			// Project point back into triangle
			double dUp = pTriangle[0] + pTriangle[1];
			
			if(dUp > 1) {
				pTriangle[0] -= 0.5 * (dUp - 1);
				pTriangle[1] -= 0.5 * (dUp - 1);
			}
			
			pTriangle = pTriangle.cwiseMin(1).cwiseMax(0);
			
			Vec3d pScreen = applyProjection(projection, p1 + triToObject * pTriangle);
			if((pScreen - pScreenPreCorrection).norm() > edgeTolerance) {
				// KJ_DBG("Edge tolerance violated");
				continue;
			}
			if((pScreen(seq(0, 1)) - screenTarget).norm() > edgeTolerance) {
				// KJ_DBG("Edge tolerance violated");
				continue;
			}
			
			if(pScreenPreCorrection(2) <= projection.minDepth)
				continue;
			
			// TODO: Plane clipping
			
			// Depth buffer check & buffer adjustments
			double detRatio = totalDet / realspaceDet;
			double depth = pScreenPreCorrection(2);
			
			// Get rid of NaN
			if(depth != depth)
				continue;
			
			if(detRatio != detRatio)
				continue;
			
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

void rasterizeGeometry(const HFProjectionStruct& projection, Eigen::MatrixXd& depthBuffer, Eigen::MatrixXd& determinantBuffer, Mesh::Reader mesh, double edgeTolerance, double depthTolerance) {
	auto vertices = mesh.getVertices();
	auto vertexData = vertices.getData();
	auto indices = mesh.getIndices();
	
	auto getVertex = [&](uint32_t i) {
		auto idx = indices[i];	
		return Vec3d(vertexData[3 * idx + 0], vertexData[3 * idx + 1], vertexData[3 * idx + 2]);
	};
	
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
			
			for(auto iPoly : kj::range(0, polys.size() - 1)) {
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

void rasterizePoint(Vec3d pObj, const HFProjectionStruct& projection, const Eigen::MatrixXd& depthBuffer, const Eigen::MatrixXd& determinantBuffer, double depthTol, double r, Eigen::MatrixXd& output) {
	Vec3d pScreen = applyProjection(projection, pObj);
	
	if(pScreen(2) < projection.minDepth)
		return;
	
	int32_t iCenter = (int32_t) floor(pScreen[0]);
	int32_t jCenter = (int32_t) floor(pScreen[1]);
	
	if(iCenter < 0 || iCenter >= determinantBuffer.cols())
		return;
	
	if(jCenter < 0 || jCenter >= determinantBuffer.rows())
		return;
	
	double minDepth = depthBuffer(jCenter, iCenter) + depthTol;
	double det = determinantBuffer(jCenter, iCenter);
	
	if(pScreen(2) > minDepth + depthTol)
		return;
	
	if(det == 0)
		return;
	
	double sigma = r * sqrt(det);
	double invSigma = 1 / sigma;
	
	if(sigma < 0.1) {
		output(jCenter, iCenter) += det;
		return;
	}
	
	auto clampI = [&](double i) {
		return std::max(0, std::min((int32_t) round(i), (int32_t) determinantBuffer.cols() - 1));
	};
	auto clampJ = [&](double j) {
		return std::max(0, std::min((int32_t) round(j), (int32_t) determinantBuffer.rows() - 1));
	};
	
	int32_t i1 = clampI(iCenter - 3 * sigma);
	int32_t i2 = clampI(iCenter + 3 * sigma);
	int32_t j1 = clampJ(jCenter - 3 * sigma);
	int32_t j2 = clampJ(jCenter + 3 * sigma);
		
	double cumWeight = 0;
	for(auto i = i1; i <= i2; ++i) {
		for(auto j = j1; j <= j2; ++j) {
			auto dj = j - jCenter;
			auto di = i - iCenter;
			
			double d = sqrt(di * di + dj * dj);
			double dNorm = (d * invSigma) * (d * invSigma);
			double weight = exp(-0.5 * dNorm);
			cumWeight += weight;
		}
	}
	
	double invCumWeight = 1 / cumWeight;
	
	for(auto i = i1; i <= i2; ++i) {
		for(auto j = j1; j <= j2; ++j) {
			auto dj = j - jCenter;
			auto di = i - iCenter;
			
			double d = sqrt(di * di + dj * dj);
			double dNorm = (d * invSigma) * (d * invSigma);
			double weight = exp(-0.5 * dNorm);
			output(j, i) += det * weight * invCumWeight;
		}
	}
}

struct HFCamImpl : public HFCam::Server {
	Eigen::MatrixXd detBuffer;
	Eigen::MatrixXd depthBuffer;
	Eigen::MatrixXd accumBuffer;
	
	Temporary<HFCamProjection> projection;
	
	HFCamImpl(const HFCamImpl& other) :
		HFCamImpl(other.detBuffer, other.depthBuffer, other.projection)
	{
		accumBuffer = other.accumBuffer;
	}
	
	HFCamImpl(const Eigen::MatrixXd& detBuffer, const Eigen::MatrixXd& depthBuffer, HFCamProjection::Reader projection) :
		detBuffer(detBuffer),
		depthBuffer(depthBuffer),
		projection(projection),
		accumBuffer(detBuffer.rows(), detBuffer.cols())
	{
		accumBuffer.setConstant(0);
	}
	
	Promise<void> clear(ClearContext ctx) {
		accumBuffer.setConstant(0);
		return READY_NOW;
	}
	
	Promise<void> clone(CloneContext ctx) {
		ctx.initResults().setCam(kj::heap<HFCamImpl>(*this));
		return READY_NOW;
	}
	
	Promise<void> addPoints(AddPointsContext ctx) {
		auto params = ctx.getParams();
		auto points = params.getPoints();
		auto pShape = points.getShape();
		auto pData = points.getData();
		
		KJ_REQUIRE(pShape.size() >= 1);
		KJ_REQUIRE(pShape[0] == 3);
		KJ_REQUIRE(pData.size() % 3 == 0);
		
		auto nPoints = pData.size() / 3;
		
		double dTol = params.getDepthTolerance();
		double r = params.getR();
		
		HFProjectionStruct proj;
		proj.load(projection.asReader());
		
		for(auto iPoint : kj::range(0, nPoints)) {
			Vec3d p(pData[0 * nPoints + iPoint], pData[1 * nPoints + iPoint], pData[2 * nPoints + iPoint]);
			rasterizePoint(
				p,
				proj, depthBuffer, detBuffer,
				dTol, r,
				accumBuffer
			);
		}
		
		return READY_NOW;
	}
	
	Promise<void> get(GetContext ctx) {
		auto image = ctx.initResults().getImage();
		size_t nRows = accumBuffer.rows();
		size_t nCols = accumBuffer.cols();
		image.setShape({nCols, nRows});
		image.setData(kj::ArrayPtr<const double>(accumBuffer.data(), accumBuffer.size()));
		
		return READY_NOW;
	}
	
	Promise<void> getData(GetDataContext ctx) {
		auto data = ctx.initResults();
		data.setProjection(projection.asReader());
		
		auto detBuf = ctx.getResults().getDeterminantBuffer();
		detBuf.setShape({(size_t ) detBuffer.cols(), (size_t) detBuffer.rows()});
		detBuf.setData(kj::ArrayPtr<const double>(detBuffer.data(), detBuffer.size()));
		
		auto dBuf = ctx.getResults().getDepthBuffer();
		dBuf.setShape({(size_t) depthBuffer.cols(), (size_t) depthBuffer.rows()});
		dBuf.setData(kj::ArrayPtr<const double>(depthBuffer.data(), depthBuffer.size()));
		
		return READY_NOW;
	}
};

struct CamProvider : public HFCamProvider::Server {;
	using Parent = HFCamProvider::Server;
	
	Promise<void> makeToroidalProjection(MakeToroidalProjectionContext context) override {
		auto params = context.getParams();
		
		toroidalProjection(
			params.getW(), params.getH(),
			params.getPhi(), params.getRTarget(), params.getZTarget(),
			params.getInclination(), params.getHorizontalInclination(), params.getDistance(),
			params.getViewportHeight(), params.getFieldOfView(),
			context.initResults()
		);
		
		return READY_NOW;
	}
	
	Promise<void> makeCamera(MakeCameraContext context) override {
		using Mat = Eigen::MatrixXd;
		
		// Postprocess using local geometry library
		auto geoLib = newGeometryLib();
		auto mergeRequest = geoLib.mergeRequest();
		mergeRequest.setNested(context.getParams().getGeometry());
		
		auto projection = context.getParams().getProjection();
		
		// Create buffers
		auto detBuf = heapHeld<Mat>(projection.getHeight(), projection.getWidth());
		detBuf -> setZero();
		auto dBuf = heapHeld<Mat>(projection.getHeight(), projection.getWidth());
		dBuf -> setConstant(std::numeric_limits<double>::infinity());
		
		auto mergeResultRef = mergeRequest.send().getRef();
		return getActiveThread().dataService().download(mergeResultRef)
		.then([context, dBuf, detBuf](LocalDataRef<MergedGeometry> localRef) mutable {
			auto mergedGeometry = localRef.get();
			
			HFProjectionStruct projection;
			projection.load(context.getParams().getProjection());
			
			kj::Vector<Promise<void>> promises;
			auto entries = mergedGeometry.getEntries();
			for(auto iMesh : kj::indices(entries)) {
				Promise<void> processMesh = kj::evalLater([iMesh, dBuf, detBuf, projection, entries, context]() mutable {
					KJ_LOG(INFO, "Processing mesh", iMesh);
					rasterizeGeometry(
						projection,
						*dBuf, *detBuf,
						entries[iMesh].getMesh(),
						context.getParams().getEdgeTolerance(), context.getParams().getDepthTolerance()
					);
				}).attach(cp(localRef));
				
				promises.add(mv(processMesh));
			}
			
			return joinPromises(promises.releaseAsArray());
		})
		.then([detBuf, dBuf, context]() mutable {			
			context.initResults().setCam(kj::heap<HFCamImpl>(*detBuf, *dBuf, context.getParams().getProjection()));
		})
		.attach(detBuf.x(), dBuf.x());
	}
};

} // anonymous namespace

HFCamProvider::Client newHFCamProvider() {
	return kj::heap<CamProvider>();
}

} // namespace fsc