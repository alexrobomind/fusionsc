#include <capnp/message.h>

#include "magnetics.h"
#include "data.h"
#include "interpolation.h"

#include "kernels/launch.h"
#include "kernels/tensor.h"
#include "kernels/message.h"
#include "kernels/karg.h"

#include "magnetics-kernels.h"

namespace fsc {
	
namespace {
	
struct FieldCalculation {
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
	
	void addComputed(double scale, Float64Tensor::Reader otherFieldIn, ToroidalGridStruct otherGrid) {
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
				
		calculation = calculation.then([this, otherField = mv(otherField), otherGrid, scale]() mutable {
			return FSC_LAUNCH_KERNEL(
				kernels::addFieldInterpKernel,
				*_device, 
				field -> getHost().dimension(0),
				FSC_KARG(field, NOCOPY), FSC_KARG(points, NOCOPY),
				FSC_KARG(otherField, ALIAS_IN), otherGrid,
				scale
			);
		});
		
	}
	
	void biotSavart(double current, Float64Tensor::Reader input, BiotSavartSettings::Reader settings) {
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
		
		calculation = calculation.then([this, filament = mv(filament), coilWidth, stepSize, current]() mutable {
			KJ_LOG(INFO, "Processing coil", current, coilWidth, stepSize, filament.dimension(1));
			
			// Launch calculation
			return FSC_LAUNCH_KERNEL(
				kernels::biotSavartKernel,
				*_device,
				field -> getHost().dimension(0), FSC_KARG(points, NOCOPY),
				FSC_KARG(filament, ALIAS_IN), current, coilWidth, stepSize, FSC_KARG(field, NOCOPY)
			);
		});
	}
	
	void dipoles(double scale, DipoleCloud::Reader cloud) {
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
		
		calculation = calculation.then([this, nPoints, scale, positions = mv(positions), moments = mv(moments), radiiNative = mv(radiiNative)]() mutable {
			return FSC_LAUNCH_KERNEL(
				kernels::dipoleFieldKernel,
				*_device,
				field -> getHost().dimension(0), FSC_KARG(points, NOCOPY),
				FSC_KARG(mv(positions), ALIAS_IN), FSC_KARG(mv(moments), ALIAS_IN), FSC_KARG(mv(radiiNative), ALIAS_IN),
				scale, FSC_KARG(field, NOCOPY)
			);
		});
	}
	
	void equilibrium(double scale, AxisymmetricEquilibrium::Reader equilibrium) {		
		calculation = calculation.then([this, equilibrium = Temporary<AxisymmetricEquilibrium>(equilibrium), scale]() mutable {
			auto mapped = FSC_MAP_BUILDER(fsc, AxisymmetricEquilibrium, mv(equilibrium), *_device, true);
			
			return FSC_LAUNCH_KERNEL(
				kernels::eqFieldKernel,
				*_device,
				field -> getHost().dimension(0),
				FSC_KARG(points, NOCOPY),
				FSC_KARG(mapped, ALIAS_IN), scale, FSC_KARG(field, NOCOPY)
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

template<typename Device>
struct CalculationSession : public FieldCalculator::Server {	
	// Device device;
	Own<DeviceBase> device;
	
	CalculationSession(Own<DeviceBase> device) :
		device(mv(device))
	{}
	
	Promise<void> evaluateXyz(EvaluateXyzContext context) {
		KJ_LOG(INFO, "Initiating magnetic field evaluation");
		
		auto field = context.getParams().getField();
		auto pointsIn = context.getParams().getPoints();
		
		// Validate shape of input tensor
		validateTensor(pointsIn);
		auto pointsShape = pointsIn.getShape();
		KJ_REQUIRE(pointsIn.getShape().size() >= 1);
		KJ_REQUIRE(pointsIn.getShape()[0] == 3);
		
		size_t nPoints = 1;
		for(auto i : kj::range(1, pointsShape.size()))
			nPoints *= pointsShape[i];
		
		auto pointData = pointsIn.getData();
		
		Eigen::Tensor<double, 2> points(nPoints, 3);
		for(auto i : kj::range(0, nPoints)) {
			points(i, 0) = pointData[0 * nPoints + i];
			points(i, 1) = pointData[1 * nPoints + i];
			points(i, 2) = pointData[2 * nPoints + i];
		}
		
		return processRoot(field, mv(points), context.getResults().initValues())
		.then([context, pointsShape]() mutable {
			// Adjust field shape to match points shape
			context.getResults().getValues().setShape(pointsShape);
		});
	}
	
	Promise<void> evaluatePhizr(EvaluatePhizrContext context) {
		auto xyzReq = thisCap().evaluateXyzRequest();
		auto params = context.getParams();
		
		xyzReq.setField(params.getField());
		xyzReq.setPoints(params.getPoints());
		
		auto points = xyzReq.getPoints().getData();
		auto nPoints = points.size() / 3;
		
		for(auto i : kj::range(0, nPoints)) {
			double phi = points[0 * nPoints + i];
			double z = points[1 * nPoints + i];
			double r = points[2 * nPoints + i];
			
			double x = std::cos(phi) * r;
			double y = std::sin(phi) * r;
			
			points.set(0 * nPoints + i, x);
			points.set(1 * nPoints + i, y);
			points.set(2 * nPoints + i, z);
		}
		
		return xyzReq.send().then([params, nPoints, context](auto response) mutable {
			auto results = context.initResults();
			results.setValues(response.getValues());
			
			auto points = params.getPoints().getData();
			auto values = results.getValues().getData();
			
			KJ_REQUIRE(values.size() == 3 * nPoints);
			
			for(auto i : kj::range(0, nPoints)) {
				double bX = values[0 * nPoints + i];
				double bY = values[1 * nPoints + i];
				double bZ = values[2 * nPoints + i];
				
				double phi = points[0 * nPoints + i];
				
				double bR = bX * std::cos(phi) + bY * std::sin(phi);
				double bPhi = bY * std::cos(phi) - bX * std::sin(phi);
				
				values.set(0 * nPoints + i, bPhi);
				values.set(1 * nPoints + i, bZ);
				values.set(2 * nPoints + i, bR);
			}
		});
	}
	
	//! Handles compute request
	Promise<void> compute(ComputeContext context) {
		KJ_LOG(INFO, "Initiating magnetic field computation");
		
		constexpr unsigned int GRID_VERSION = 7;
		auto grid = readGrid(context.getParams().getGrid(), GRID_VERSION);
		
		// Calculate all grid points
		Tensor<double, 4> points(grid.nPhi, grid.nZ, grid.nR, 3);
		for(auto iR : kj::range(0, grid.nR)) {
			for(auto iPhi : kj::range(0, grid.nPhi)) {
				for(auto iZ : kj::range(0, grid.nZ)) {
					points(iPhi, iZ, iR, 0) = grid.phi(iPhi);
					points(iPhi, iZ, iR, 1) = grid.z(iZ);
					points(iPhi, iZ, iR, 2) = grid.r(iR);
				}
			}
		}
		
		// Submit evaluation request request
		auto calcReq = thisCap().evaluatePhizrRequest();
		calcReq.setField(context.getParams().getField());
		writeTensor(points, calcReq.getPoints());
		
		// Parameters no longer required
		context.releaseParams();
		
		auto refPromise = calcReq.send().then([grid](auto response) -> DataRef<Float64Tensor>::Client {
			Tensor<double, 4> values;
			readTensor(response.getValues(), values);
			
			Temporary<Float64Tensor> holder;
			
			// The computation request uses the format (phi, z, r, 3) (column major)
			// We need to transpose to (3, r, z, phi)
			{
				using A = Eigen::array<Eigen::Index, 4>;
				Tensor<double, 4> tmp = values.shuffle<A>({3, 2, 1, 0});
				values = tmp;
			}
			
			writeTensor(values, holder.asBuilder());
			
			return getActiveThread().dataService().publish(holder.asReader());
		});
			
		auto cf = context.initResults().getComputedField();
		writeGrid(grid, cf.initGrid());
		cf.setData(mv(refPromise));
		
		return READY_NOW;
	}
	
	Promise<void> interpolateXyz(InterpolateXyzContext ctx) {
		auto params = ctx.getParams();
		
		auto req = thisCap().evaluateXyzRequest();
		
		req.getField().setComputedField(params.getField());
		req.setPoints(params.getPoints());
		
		return ctx.tailCall(mv(req));
	}
	
	Promise<void> calculateFourierMoments(calculateFourierComponents ctx) {
		auto params = ctx.getParams();
		KJ_REQUIRE(params.getNTor() >= 2 * params.getNMax() + 1);
		KJ_REQUIRE(params.getMPol() >= 2 * params.getMPol() + 1);
		
		using FP = nudft::FourierPoint<2, 2>;
		using FM = nudft::FourierMode<2, 2>;
		
		kj::Vector<FM> modes;
		kj::Vector<FP> points;
		
		Tensor(nPoints, 3) inputPoints;
		Tensor(nPoints, 3) radialComponents;
		
		radialComponents.setZero();
		
		for(auto iPhi : kj::range(0, nPhi)) {
			for(auto iTheta : kj::range(0, nTheta)) {
				double phi = 2 * np.pi / nSym * iPhi;
				double theta = 2 * np.pi / nTheta * iTheta;
				
				size_t iPoint = points.size();
				
				FP point;
				point.angles[0] = phi;
				point.angles[1] = theta;
				inputPoints.append(point);
				
				for(auto iM : kj::range(
				inputPoints(0)
			}
		}
	}
	
	//! Processes a root node of a magnetic field (creates calculator)
	Promise<void> processRoot(MagneticField::Reader node, Eigen::Tensor<double, 2>&& points, Float64Tensor::Builder out) {		
		auto newCalculator = heapHeld<FieldCalculation>(mv(points), *device);
		
		auto calcDone = processField(*newCalculator, node, 1);
		
		return calcDone.then([newCalculator, out, this]() mutable {				
			return newCalculator -> finish(out).eagerlyEvaluate(nullptr);
		})
		.attach(newCalculator.x());
	}
	
	Promise<void> processFilament(FieldCalculation& calculator, Filament::Reader node, BiotSavartSettings::Reader settings, double scale) {
		if(scale == 0)
			return READY_NOW;
		
		while(node.isNested()) {
			node = node.getNested();
		}
		
		switch(node.which()) {
			case Filament::INLINE:
				// The biot savart operation is chained by the calculator
				calculator.biotSavart(scale, node.getInline(), settings);
				return READY_NOW;				
				
			case Filament::REF:
				return getActiveThread().dataService().download(node.getRef()).then([&calculator, settings, scale, this](LocalDataRef<Filament> local) mutable {
					return processFilament(calculator, local.get(), settings, scale).attach(cp(local));
				});
			
			case Filament::SUM: {
				auto sum = node.getSum();
				auto arrBuilder = kj::heapArrayBuilder<Promise<void>>(sum.size());
				
				for(auto i : kj::indices(sum)) {
					arrBuilder.add(processFilament(calculator, sum[i], settings, scale));
				}
				
				return kj::joinPromises(arrBuilder.finish());
			}
			default:
				KJ_FAIL_REQUIRE("Unknown filament node encountered. This either indicates that a device-specific node was not resolved, or a generic node from a future library version was presented", node);
		}
	}
	
	Promise<void> processField(FieldCalculation& calculator, MagneticField::Reader node, double scale) {	
		if(scale == 0)
			return READY_NOW;
		
		while(node.isNested()) {
			node = node.getNested();
		}
		
		switch(node.which()) {
			case MagneticField::SUM: {
				auto builder = kj::heapArrayBuilder<Promise<void>>(node.getSum().size());
				
				for(auto newNode : node.getSum()) {
					builder.add(processField(calculator, newNode, scale));
				}
				
				return kj::joinPromises(builder.finish());
			}
			case MagneticField::REF: {
				// First download the ref
				auto ref = node.getRef();
				return getActiveThread().dataService().download(ref)
				.then([&calculator, scale, this](LocalDataRef<MagneticField> local) {
					// Then process it like usual
					return processField(calculator, local.get(), scale).attach(cp(local));
				});
			}
			case MagneticField::COMPUTED_FIELD: {
				auto cField = node.getComputedField();
				
				constexpr unsigned int GRID_VERSION = 7;
				ToroidalGridStruct grid = readGrid(cField.getGrid(), GRID_VERSION);
				
				// Then download data				
				return getActiveThread().dataService().download(cField.getData())
				.then([&calculator, scale, grid](LocalDataRef<Float64Tensor> field) {					
					calculator.addComputed(scale, field.get(), grid);
				});
			}
			case MagneticField::FILAMENT_FIELD: {
				// Process Biot-Savart field
				auto fField = node.getFilamentField();
				
				return processFilament(calculator, fField.getFilament(), fField.getBiotSavartSettings(), scale * fField.getCurrent() * fField.getWindingNo());
			}
			case MagneticField::SCALE_BY: {
				auto scaleBy = node.getScaleBy();
				
				return processField(calculator, scaleBy.getField(), scale * scaleBy.getFactor());
			}
			case MagneticField::INVERT: {
				return processField(calculator, node.getInvert(), -scale);
			}
			case MagneticField::NESTED: {
				return processField(calculator, node.getNested(), scale);
			}
			case MagneticField::CACHED: {
				auto cached = node.getCached();
				return processField(calculator, cached.getNested(), scale);
			}
			case MagneticField::AXISYMMETRIC_EQUILIBRIUM: {
				calculator.equilibrium(scale, node.getAxisymmetricEquilibrium());
				return READY_NOW;
			}
			case MagneticField::DIPOLE_CLOUD: {
				calculator.dipoles(scale, node.getDipoleCloud());
				return READY_NOW;
			}
			default:
				KJ_FAIL_REQUIRE("Unknown magnetic field node encountered. This either indicates that a device-specific node was not resolved, or a generic node from a future library version was presented", node);
		}
	//});
	}
};

struct FieldCache : public FieldResolverBase {
	ID target;
	Temporary<ComputedField> compField;
	
	FieldCache(ID target, Temporary<ComputedField> compField) :
		target(mv(target)),
		compField(mv(compField))
	{}
		
	Promise<void> processField(MagneticField::Reader input, MagneticField::Builder output, ResolveFieldContext context) override {
		return ID::fromReaderWithRefs(input)
		.then([this, input, output, context](ID id) mutable -> Promise<void> {
			if(id == target) {
				auto cached = output.initCached();
				cached.setComputed(compField);
				cached.setNested(input);
				
				return READY_NOW;
			}
		
			return FieldResolverBase::processField(input, output, context);
		});
	}
};

}

bool isBuiltin(MagneticField::Reader field) {
	switch(field.which()) {
		case MagneticField::SUM:
		case MagneticField::REF:
		case MagneticField::COMPUTED_FIELD:
		case MagneticField::FILAMENT_FIELD:
		case MagneticField::SCALE_BY:
		case MagneticField::INVERT:
		case MagneticField::CACHED:
		case MagneticField::NESTED:
			return true;
		default:
			return false;
	}
}

bool isBuiltin(Filament::Reader filament) {
	switch(filament.which()) {
		case Filament::INLINE:
		case Filament::REF:
		case Filament::NESTED:
		case Filament::SUM:
			return true;
		
		default:
			return false;
	}
}

FieldResolver::Client newCache(MagneticField::Reader field, ComputedField::Reader computed) {
	Temporary<ComputedField> cf(computed);
	auto clientPromise = ID::fromReaderWithRefs(field).then([cf = mv(cf)](ID id) mutable -> FieldResolver::Client {
		return kj::heap<FieldCache>(mv(id), mv(cf));
	});
	
	return clientPromise;
}
	
ToroidalGridStruct readGrid(ToroidalGrid::Reader in, unsigned int maxOrdinal) {
	KJ_REQUIRE(hasMaximumOrdinal(in, maxOrdinal));
	
	ToroidalGridStruct out;
	
	out.rMin = in.getRMin();
	out.rMax = in.getRMax();
	out.zMin = in.getZMin();
	out.zMax = in.getZMax();
	out.nR = (int) in.getNR();
	out.nZ = (int) in.getNZ();
	out.nSym = (int) in.getNSym();
	out.nPhi = (int) in.getNPhi();
	
	KJ_REQUIRE(out.isValid(), in);
	return out;
}

void writeGrid(const ToroidalGridStruct& in, ToroidalGrid::Builder out) {
	KJ_REQUIRE(in.isValid());
	
	out.setRMin(in.rMin);
	out.setRMax(in.rMax);
	out.setZMin(in.zMin);
	out.setZMax(in.zMax);
	out.setNR(in.nR);
	out.setNZ(in.nZ);
	out.setNSym(in.nSym);
	out.setNPhi(in.nPhi);
}

Promise<void> FieldResolverBase::resolveField(ResolveFieldContext context) {
	auto input = context.getParams().getField();
	auto output = context.initResults();
	
	return processField(input, output, context);
}

Promise<void> FieldResolverBase::resolveFilament(ResolveFilamentContext ctx) {
	auto input = ctx.getParams().getFilament();
	
	auto req = thisCap().resolveFieldRequest();
	req.getField().initFilamentField().setFilament(input);
	req.setFollowRefs(ctx.getParams().getFollowRefs());
	
	return req.send()
	.then([ctx](auto field) mutable {
		KJ_REQUIRE(field.isFilamentField());
		
		ctx.setResults(field.getFilamentField().getFilament());
	});
}

Promise<void> FieldResolverBase::processField(MagneticField::Reader input, MagneticField::Builder output, ResolveFieldContext context) {
	switch(input.which()) {
		case MagneticField::SUM: {
			auto inSum = input.getSum();
			auto outSum = output.initSum(inSum.size());
			
			auto subTasks = kj::heapArrayBuilder<Promise<void>>(inSum.size());
			for(unsigned int i = 0; i < inSum.size(); ++i) {
				subTasks.add(processField(inSum[i], outSum[i], context));
			}
			
			return kj::joinPromises(subTasks.finish()).attach(thisCap());
		}
		case MagneticField::REF: {
			if(!context.getParams().getFollowRefs()) {
				output.setRef(input.getRef());
				return kj::READY_NOW;
			}
			
			auto tmpMessage = kj::heap<capnp::MallocMessageBuilder>();
			MagneticField::Builder newOutput = tmpMessage -> initRoot<MagneticField>();
			
			return getActiveThread().dataService().download(input.getRef())
			.then([this, newOutput, context] (LocalDataRef<MagneticField> ref) mutable {
				return processField(ref.get(), newOutput, context);
			}).then([this, output, newOutput]() mutable {
				return output.setRef(getActiveThread().dataService().publish(newOutput));
			}).attach(mv(tmpMessage), thisCap());
		}
		case MagneticField::COMPUTED_FIELD: {
			output.setComputedField(input.getComputedField());
			return kj::READY_NOW;
		}
		case MagneticField::FILAMENT_FIELD: {
			auto filIn  = input.getFilamentField();
			auto filOut = output.initFilamentField();
			
			filOut.setCurrent(filIn.getCurrent());
			filOut.setBiotSavartSettings(filIn.getBiotSavartSettings());
			filOut.setWindingNo(filIn.getWindingNo());
			
			return processFilament(filIn.getFilament(), filOut.initFilament(), context);
		}
		case MagneticField::SCALE_BY: {
			output.initScaleBy().setFactor(input.getScaleBy().getFactor());
			return processField(input.getScaleBy().getField(), output.getScaleBy().initField(), context);
		}
		case MagneticField::INVERT: {
			return processField(input.getInvert(), output.initInvert(), context);
		}
		case MagneticField::NESTED: {
			return processField(input.getNested(), output, context);
		}
		case MagneticField::CACHED: {
			auto cachedIn = input.getCached();
			auto cachedOut = output.getCached();
			
			cachedOut.setComputed(cachedIn.getComputed());
			
			return processField(cachedIn.getNested(), cachedOut.getNested(), context);
		}
		default: {
			output.setNested(input);
			return READY_NOW;
		}
	}
}

Promise<void> FieldResolverBase::processFilament(Filament::Reader input, Filament::Builder output, ResolveFieldContext context) {
	switch(input.which()) {
		case Filament::INLINE: {
			output.setInline(input.getInline());
			return kj::READY_NOW;
		}
		case Filament::REF: {
			if(!context.getParams().getFollowRefs()) {
				output.setRef(input.getRef());
				return kj::READY_NOW;
			}
			
			auto tmpMessage = kj::heap<capnp::MallocMessageBuilder>();
			Filament::Builder newOutput = tmpMessage -> initRoot<Filament>();
			
			return getActiveThread().dataService().download(input.getRef())
			.then([this, newOutput, context] (LocalDataRef<Filament> ref) mutable {
				return processFilament(ref.get(), newOutput, context);
			}).then([this, output, newOutput]() mutable {
				return output.setRef(getActiveThread().dataService().publish(newOutput));
			}).attach(mv(tmpMessage), thisCap());
		}
		case Filament::NESTED: {
			return processFilament(input.getNested(), output, context);
		}
		case Filament::SUM: {
			auto sumIn = input.getSum();
			auto sumOut = output.initSum(sumIn.size());
			
			auto arrBuilder = kj::heapArrayBuilder<Promise<void>>(sumIn.size());
			
			for(auto i : kj::indices(sumIn)) {
				auto in = sumIn[i];
				auto out = sumOut[i];
				
				arrBuilder.add(processFilament(in, out, context));
			}
			
			return kj::joinPromises(arrBuilder.finish());
		}
		default: {
			output.setNested(input);
			return READY_NOW;
		}
	}
}

FieldCalculator::Client newFieldCalculator(Own<DeviceBase> dev) {
	return kj::heap<CalculationSession<Eigen::ThreadPoolDevice>>(mv(dev));
}

namespace {
	void buildCoil(double rMaj, double rMin, double phi, Filament::Builder out) {
		const size_t nTheta = 4;
		
		Tensor<double, 2> result(3, nTheta);
		for(auto i : kj::range(0, nTheta)) {
			double theta = 2 * pi / nTheta * i;
			
			double r = rMaj + rMin * std::cos(theta);
			double z =        rMin * std::sin(theta);
			
			double x = std::cos(phi) * r;
			double y = std::sin(phi) * r;
			
			result(0, i) = x;
			result(1, i) = y;
			result(2, i) = z;
		}
		
		writeTensor(result, out.initInline());
	}
	
	void buildAxis(double rMaj, Filament::Builder out) {
		const size_t nPhi = 200;
		
		Tensor<double, 2> result(3, nPhi);
		for(auto i : kj::range(0, nPhi)) {
			double phi = 2 * pi / nPhi * i;
			
			double x = std::cos(phi) * rMaj;
			double y = std::sin(phi) * rMaj;
			double z = 0;
			
			result(0, i) = x;
			result(1, i) = y;
			result(2, i) = z;
		}
		
		writeTensor(result, out.initInline());
	}
}

void simpleTokamak(MagneticField::Builder output, double rMajor, double rMinor, unsigned int nCoils, double Ip) {
	nCoils = 20;
	auto fields = output.initSum(nCoils + 1);
	
	for(auto i : kj::range(0, nCoils)) {
		auto filamentField = fields[i].initFilamentField();
		filamentField.getBiotSavartSettings().setStepSize(0.01);
		
		double phi = 2 * pi / nCoils * i;
		
		buildCoil(rMajor, rMinor, phi, filamentField.getFilament());	
	}
	
	{
		auto filamentField = fields[nCoils].initFilamentField();
		filamentField.setCurrent(Ip);
		filamentField.getBiotSavartSettings().setStepSize(0.01);
		
		buildAxis(rMajor, filamentField.getFilament());
	}
}

}