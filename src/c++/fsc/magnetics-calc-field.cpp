#include "magnetics.h"
#include "magnetics-internal.h"

#include "kernels/launch.h"
#include "kernels/tensor.h"
#include "kernels/message.h"
#include "kernels/karg.h"

#include "geometry.h"

#include "magnetics-kernels.h"

namespace fsc { namespace internal {
	
namespace {

struct FieldCalculation {
	using MagKernelContext = kernels::MagKernelContext;
	
	using FieldValues = ::fsc::kernels::FieldValues;
	using Field = ::fsc::kernels::Field;
	
	using MFilament = ::fsc::kernels::MFilament;
	
	constexpr static unsigned int GRID_VERSION = 7;
	
	Own<DeviceBase> _device;
	
	DeviceMappingType<FieldValues> field;
	DeviceMappingType<FieldValues> points;
	
	// This promise makes sure we only schedule a
	// calculation once the previous is finished
	Promise<void> calculation = READY_NOW;
	
	FieldCalculation(FieldValues&& pointsIn, DeviceBase& device) :
		_device(device.addRef()),
		field(mapToDevice(FieldValues(pointsIn.dimension(0), 3), device, true)),
		points(mapToDevice(mv(pointsIn), device, true))
	{
		field -> getHost().setZero();
		
		field -> updateDevice();
		points -> updateDevice();
	}
	
	~FieldCalculation() {}
	
	kernels::MagKernelContext makeContext() {
		kernels::MagKernelContext result(points -> get(), field -> get());
		
		return result;
	}
	
	Own<kernels::MagKernelContext> mapCtx(MagKernelContext ctx) {
		return kj::attachVal(ctx, field -> addRef(), points -> addRef());
	}
	
	void addComputed(const MagKernelContext& ctx, Float64Tensor::Reader otherFieldIn, ToroidalGridStruct otherGrid) {
		auto shape = otherFieldIn.getShape();
		
		KJ_REQUIRE(shape.size() == 4);
		KJ_REQUIRE(shape[3] == 3);
		
		auto data = otherFieldIn.getData();
		
		// Write field into native format
		// for(auto i : otherFieldIn.getShape())
		// 	KJ_DBG("Shape", i);
	
		// KJ_DBG(3, otherGrid.nR, otherGrid.nZ, otherGrid.nPhi);
		Field otherField(3, otherGrid.nR, otherGrid.nZ, otherGrid.nPhi);
		KJ_REQUIRE(otherField.size() == data.size());
		
		for(int i = 0; i < otherField.size(); ++i) {
			otherField.data()[i] = data[i];
		}
				
		calculation = calculation.then([this, ctx, otherField = mv(otherField), otherGrid]() mutable {
			return FSC_LAUNCH_KERNEL(
				kernels::addFieldInterpKernel,
				*_device,
				field -> getHost().dimension(0),
				
				mapCtx(ctx),
				
				FSC_KARG(otherField, ALIAS_IN), otherGrid
			);
		});
		
	}
	
	void biotSavart(const MagKernelContext& ctx, Float64Tensor::Reader input, BiotSavartSettings::Reader settings) {
		auto shape = input.getShape();
		
		KJ_REQUIRE(shape.size() == 2);
		KJ_REQUIRE(shape[1] == 3);
		KJ_REQUIRE(shape[0] >= 2);
		
		int n_points = (int) shape[0];
		MFilament filament(3, n_points);
		
		// Copy filament into native buffer
		auto data = input.getData();
		for(int i = 0; i < n_points; ++i) {
			filament(0, i) = data[3 * i + 0];
			filament(1, i) = data[3 * i + 1];
			filament(2, i) = data[3 * i + 2];
		}
		
		double coilWidth = settings.getWidth();
		double stepSize  = settings.getStepSize();
		
		KJ_REQUIRE(stepSize != 0, "Please specify a step size in the Biot-Savart settings");
		
		calculation = calculation.then([this, ctx, filament = mv(filament), coilWidth, stepSize]() mutable {
			KJ_LOG(INFO, "Processing coil", coilWidth, stepSize, filament.dimension(1));
			
			// Launch calculation
			return FSC_LAUNCH_KERNEL(
				kernels::biotSavartKernel,
				*_device,
				field -> getHost().dimension(0),
				
				mapCtx(ctx),
				
				FSC_KARG(filament, ALIAS_IN), coilWidth, stepSize
			);
		});
	}
	
	void dipoles(const MagKernelContext& ctx, DipoleCloud::Reader cloud) {
		Tensor<double, 2> positions;
		Tensor<double, 2> moments;
		
		readTensor(cloud.getPositions(), positions);
		readTensor(cloud.getMagneticMoments(), moments);
		
		size_t nPoints = positions.dimension(0);
		KJ_REQUIRE(moments.dimension(0) == nPoints);
		KJ_REQUIRE(moments.dimension(1) == 3);
		KJ_REQUIRE(positions.dimension(1) == 3);
		
		auto radii = cloud.getRadii();
		KJ_REQUIRE(radii.size() == nPoints);
		
		auto radiiNative = kj::heapArray<double>(nPoints);
		for(auto i : kj::indices(radii))
			radiiNative[i] = radii[i];
		
		calculation = calculation.then([this, ctx, nPoints, positions = mv(positions), moments = mv(moments), radiiNative = mv(radiiNative)]() mutable {
			return FSC_LAUNCH_KERNEL(
				kernels::dipoleFieldKernel,
				*_device,
				field -> getHost().dimension(0),
				
				mapCtx(ctx),
				
				FSC_KARG(mv(positions), ALIAS_IN), FSC_KARG(mv(moments), ALIAS_IN), FSC_KARG(mv(radiiNative), ALIAS_IN)
			);
		});
	}
	
	void equilibrium(const MagKernelContext& ctx, AxisymmetricEquilibrium::Reader equilibrium) {		
		calculation = calculation.then([this, ctx, equilibrium = Temporary<AxisymmetricEquilibrium>(equilibrium)]() mutable {
			auto mapped = FSC_MAP_BUILDER(fsc, AxisymmetricEquilibrium, mv(equilibrium), *_device, true);
			
			return FSC_LAUNCH_KERNEL(
				kernels::eqFieldKernel,
				*_device,
				field -> getHost().dimension(0),
				
				mapCtx(ctx),
				
				FSC_KARG(mapped, ALIAS_IN)
			);
		});
	}
	
	Promise<void> finish(Float64Tensor::Builder out) {
		calculation = calculation
		.then([this]() {
			field -> updateHost();
			return _device -> barrier();
		})
		.then([this, out]() {
			writeTensor(field->getHost(), out);
		});
		
		return mv(calculation);
	}
};

}
	
//! Processes a root node of a magnetic field (creates calculator)
Promise<void> FieldCalculatorImpl::processRoot(MagneticField::Reader node, Eigen::Tensor<double, 2>&& points, Float64Tensor::Builder out) {
	Shared<FieldCalculation> newCalculator(mv(points), *device);
	
	auto calcDone = processField(*newCalculator, node, newCalculator -> makeContext());
	
	return calcDone.then([newCalculator, out, this]() mutable {				
		return newCalculator -> finish(out).eagerlyEvaluate(nullptr);
	})
	.attach(cp(newCalculator));
}
	
Promise<void> FieldCalculatorImpl::processFilament(FieldCalculation& calculator, Filament::Reader node, BiotSavartSettings::Reader settings, const MagKernelContext& ctx) {
	if(ctx.scale == 0)
		return READY_NOW;
	
	while(node.isNested()) {
		node = node.getNested();
	}
	
	switch(node.which()) {
		case Filament::INLINE:
			// The biot savart operation is chained by the calculator
			calculator.biotSavart(ctx, node.getInline(), settings);
			return READY_NOW;				
			
		case Filament::REF:
			return getActiveThread().dataService().download(node.getRef()).then([&calculator, settings, ctx, this](LocalDataRef<Filament> local) mutable {
				return processFilament(calculator, local.get(), settings, ctx).attach(cp(local));
			});
		
		case Filament::SUM: {
			auto sum = node.getSum();
			auto arrBuilder = kj::heapArrayBuilder<Promise<void>>(sum.size());
			
			for(auto i : kj::indices(sum)) {
				arrBuilder.add(processFilament(calculator, sum[i], settings, ctx));
			}
			
			return kj::joinPromises(arrBuilder.finish());
		}
		default:
			KJ_FAIL_REQUIRE("Unresolved filament node encountered during magnetic field calculation", node);
	}
}
	
Promise<void> FieldCalculatorImpl::processField(FieldCalculation& calculator, MagneticField::Reader node, const MagKernelContext& ctx) {	
	if(ctx.scale == 0)
		return READY_NOW;
	
	while(node.isNested()) {
		node = node.getNested();
	}
	
	switch(node.which()) {
		case MagneticField::SUM: {
			auto builder = kj::heapArrayBuilder<Promise<void>>(node.getSum().size());
			
			for(auto newNode : node.getSum()) {
				builder.add(processField(calculator, newNode, ctx));
			}
			
			return kj::joinPromises(builder.finish());
		}
		case MagneticField::REF: {
			// First download the ref
			auto ref = node.getRef();
			return getActiveThread().dataService().download(ref)
			.then([&calculator, ctx, this](LocalDataRef<MagneticField> local) {
				// Then process it like usual
				return processField(calculator, local.get(), ctx).attach(cp(local));
			});
		}
		case MagneticField::COMPUTED_FIELD: {
			auto cField = node.getComputedField();
			
			constexpr unsigned int GRID_VERSION = 7;
			ToroidalGridStruct grid = readGrid(cField.getGrid(), GRID_VERSION);
			
			// Then download data				
			return getActiveThread().dataService().download(cField.getData())
			.then([&calculator, ctx, grid](LocalDataRef<Float64Tensor> field) {					
				calculator.addComputed(ctx, field.get(), grid);
			});
		}
		case MagneticField::FILAMENT_FIELD: {
			// Process Biot-Savart field
			auto fField = node.getFilamentField();
			
			return processFilament(calculator, fField.getFilament(), fField.getBiotSavartSettings(), ctx.scaleBy(fField.getCurrent() * fField.getWindingNo()));
		}
		case MagneticField::SCALE_BY: {
			auto scaleBy = node.getScaleBy();
			
			return processField(calculator, scaleBy.getField(), ctx.scaleBy(scaleBy.getFactor()));
		}
		case MagneticField::INVERT: {
			return processField(calculator, node.getInvert(), ctx.scaleBy(-1));
		}
		case MagneticField::NESTED: {
			return processField(calculator, node.getNested(), ctx);
		}
		case MagneticField::CACHED: {
			auto cached = node.getCached();
			
			// Check if all points fit
			auto hostPoints = calculator.points -> getHost();
			auto nPoints = hostPoints.dimension(0);
			
			MagKernelContext hostCtx = ctx;
			hostCtx.points = calculator.points -> getHost();
			
			ToroidalGridStruct grid;
			try {
				constexpr unsigned int GRID_VERSION = 7;
				grid = readGrid(cached.getComputed().getGrid(), GRID_VERSION);
			} catch(kj::Exception& e) {
				goto recalculate;
			}
			
			for(auto iPoint : kj::range(0, nPoints)) {
				Vec3d xyz = ctx.getPosition(iPoint);
				double x = xyz(0);
				double y = xyz(1);
				double z = xyz(2);
				
				constexpr double tol = 1e-6;
								
				double r = sqrt(x*x + y*y);
				if(r < grid.rMin - tol || r > grid.rMax + tol) {
					goto recalculate;
				}
				if(z < grid.zMin - tol || z > grid.zMax + tol) {
					goto recalculate;
				}
			}
						
			return getActiveThread().dataService().download(cached.getComputed().getData())
			.then([&calculator, ctx, grid](LocalDataRef<Float64Tensor> field) {					
				calculator.addComputed(ctx, field.get(), grid);
			});
			
			// Jump to this label when we can not use the cached field
			recalculate:
			return processField(calculator, cached.getNested(), ctx);
		}
		case MagneticField::AXISYMMETRIC_EQUILIBRIUM: {
			calculator.equilibrium(ctx, node.getAxisymmetricEquilibrium());
			return READY_NOW;
		}
		case MagneticField::DIPOLE_CLOUD: {
			calculator.dipoles(ctx, node.getDipoleCloud());
			return READY_NOW;
		}
		case MagneticField::TRANSFORMED: {
			return processTransform(calculator, node.getTransformed(), ctx);
		}
		default:
			KJ_FAIL_REQUIRE("Unresolved magnetic field node encountered during field calculation.", node);
	}
}

Promise<void> FieldCalculatorImpl::processTransform(FieldCalculation& calculator, Transformed<MagneticField>::Reader node, const MagKernelContext& ctx) {
	using T = Transformed<MagneticField>;
	
	switch(node.which()) {
		case T::LEAF: return processField(calculator, node.getLeaf(), ctx);
		case T::TURNED: {
			auto turned = node.getTurned();
			auto inAxis = turned.getAxis();
			auto inCenter = turned.getCenter();
			double ang = angle(turned.getAngle());
			
			KJ_REQUIRE(inAxis.size() == 3);
			KJ_REQUIRE(inCenter.size() == 3);
			
			Vec3d axis   { inAxis[0], inAxis[1], inAxis[2] };
			Vec3d center { inCenter[0], inCenter[1], inCenter[2] };
			
			// Make a rotation AGAINST the angle
			Mat3d pointTransform = rotationAxisAngle(center, axis, -ang)(Eigen::seq(Eigen::fix<0>, Eigen::fix<2>), Eigen::seq(Eigen::fix<0>, Eigen::fix<2>));
			
			auto newCtx = ctx;
			newCtx.transformed = true;
			newCtx.transform = pointTransform * ctx.transform;
			
			return processTransform(calculator, turned.getNode(), newCtx);
		}
		
		case T::SHIFTED: {
			auto shift = node.getShifted().getShift();
			KJ_REQUIRE(shift.size() == 3);
			
			auto newCtx = ctx;
			newCtx.transformed = true;
			newCtx.transform(0, 3) -= shift[0];
			newCtx.transform(1, 3) -= shift[1];
			newCtx.transform(2, 3) -= shift[2];
			
			return processTransform(calculator, node.getShifted().getNode(), newCtx);
		}
		
		case T::SCALED:
			return KJ_EXCEPTION(FAILED, "Scaling magnetic fields is not supported. If you need this, file an issue on GitHub");
		
		default:
			KJ_FAIL_REQUIRE("Unknown transform node encountered");
	}
}

} }
