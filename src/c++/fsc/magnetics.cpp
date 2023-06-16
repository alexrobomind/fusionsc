#include <capnp/message.h>

#include "magnetics.h"
#include "data.h"
#include "kernels.h"
#include "magnetics-kernels.h"

namespace fsc {
	
namespace {
	
struct FieldCalculation {
	using Field = ::fsc::kernels::Field;
	using MFilament = ::fsc::kernels::MFilament;
	
	constexpr static unsigned int GRID_VERSION = 7;
	
	Own<DeviceBase> _device;
	ToroidalGridStruct grid;
	ToroidalGrid::Reader gridReader;
	DeviceMappingType<Field> field;
	
	// This promise makes sure we only schedule a
	// calculation once the previous is finished
	Promise<void> calculation = READY_NOW;
	
	FieldCalculation(/*ToroidalGridStruct*/ToroidalGrid::Reader in, DeviceBase& device) :
		_device(device.addRef()),
		grid(readGrid(in, GRID_VERSION)),
		gridReader(in),
		field(mapToDevice(Field(3, grid.nR, grid.nZ, grid.nPhi), device, true))
	{
		field -> getHost().setZero();
		field -> updateDevice();
	}
	
	~FieldCalculation() {}
	
	void add(double scale, Float64Tensor::Reader input) {
		auto shape = input.getShape();
		
		KJ_REQUIRE(shape.size() == 4);
		KJ_REQUIRE(shape[3] == 3);
		
		auto data = input.getData();
		
		// Write field into native format
		Field newField(3, grid.nR, grid.nZ, grid.nPhi);
		for(int i = 0; i < newField.size(); ++i) {
			newField.data()[i] = data[i];
		}
		
		calculation = calculation.then([this, newField = mv(newField), scale]() mutable {
			return FSC_LAUNCH_KERNEL(
				kernels::addFieldKernel,
				*_device, 
				field -> getHost().size(),
				FSC_KARG(field, NOCOPY), FSC_KARG(newField, ALIAS_IN), scale
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
			// Launch calculation
			return FSC_LAUNCH_KERNEL(
				kernels::biotSavartKernel,
				*_device,
				field -> getHost().size() / 3,
				grid, FSC_KARG(filament, ALIAS_IN), current, coilWidth, stepSize, FSC_KARG(field, NOCOPY)
			);
		});
	}
	
	void equilibrium(double scale, AxisymmetricEquilibrium::Reader equilibrium) {		
		calculation = calculation.then([this, equilibrium = Temporary<AxisymmetricEquilibrium>(equilibrium), scale]() mutable {
			auto mapped = mapToDevice(
				cuBuilder<AxisymmetricEquilibrium, cu::AxisymmetricEquilibrium>(mv(equilibrium)),
				*_device, true
			);
			
			return FSC_LAUNCH_KERNEL(
				kernels::eqFieldKernel,
				*_device,
				field -> getHost().size() / 3,
				grid, FSC_KARG(mapped, ALIAS_IN), scale, FSC_KARG(field, NOCOPY)
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
			KJ_DBG("Calculation finished");
			writeTensor(field->getHost(), out);
		});
		
		return mv(calculation);
	}
};

template<typename Device>
struct CalculationSession : public FieldCalculator::Server {	
	// Device device;
	Own<DeviceBase> device;
	
	// ToroidalGridStruct grid;
	// Cache<ID, LocalDataRef<Float64Tensor>> cache;
	
	CalculationSession(Own<DeviceBase> device/*, ToroidalGrid::Reader newGrid*/) :
		device(mv(device))/*,
		grid(readGrid(newGrid, GRID_VERSION))*/
	{}
	
	//! Handles compute request
	Promise<void> compute(ComputeContext context) {
		// context.allowCancellation(); // NOTE: In capnproto 0.11, this has gone away
		KJ_REQUIRE("Processing compute request");
		
		// Copy input field (so that call context can be released)
		auto field = heapHeld<Temporary<MagneticField>>(context.getParams().getField());
		auto grid  = heapHeld<Temporary<ToroidalGrid>>(context.getParams().getGrid());
		context.releaseParams();
		
		// Fill in computed grid struct
		auto compField = context.initResults().initComputedField();
		// writeGrid(grid, compField.initGrid());
		compField.setGrid(*grid);
		
		// Start calculation lazily
		auto data = processRoot(*field, *grid)
		/*.then([this, field, context, grid](LocalDataRef<Float64Tensor> tensorRef) mutable {
			// Cache field if not present, use existing if present
			return ID::fromReaderWithRefs(field->asBuilder())
			.then([this, tensorRef = mv(tensorRef)](ID id) mutable -> DataRef<Float64Tensor>::Client {
				auto insertResult = cache.insert(id, mv(tensorRef));
				return attach(mv(insertResult.element), mv(insertResult.ref));
			});
		})*/
		.attach(thisCap(), field.x(), grid.x()).eagerlyEvaluate(nullptr);
		
		compField.setData(mv(data));
		
		KJ_REQUIRE("Compute request finished");
		return READY_NOW;
	}
	
	//! Processes a root node of a magnetic field (creates calculator)
	Promise<LocalDataRef<Float64Tensor>> processRoot(MagneticField::Reader node, ToroidalGrid::Reader grid) {		
		auto newCalculator = heapHeld<FieldCalculation>(grid, *device);
		
		auto calcDone = processField(*newCalculator, node, 1);
		
		return calcDone.then([newCalculator, this]() mutable {	
			auto result = heapHeld<Temporary<Float64Tensor>>();	
			
			auto publish = newCalculator->finish(*result).then([result, this]() mutable {
				return getActiveThread().dataService().publish(result->asReader());
			})
			.eagerlyEvaluate(nullptr)
			.attach(result.x());
			
			return publish;
		})
		.attach(newCalculator.x());
	}
	
	Promise<void> processFilament(FieldCalculation& calculator, Filament::Reader node, BiotSavartSettings::Reader settings, double scale) {
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
				// First check grid compatibility
				auto cField = node.getComputedField();
				auto grid = cField.getGrid();
				
				KJ_REQUIRE(
					ID::fromReader(grid) == ID::fromReader(calculator.gridReader),
					"The field you are using as input and the field you are trying to calculate "
					"have differing grids. Currently, interpolation between different magnetic "
					"grids is not yet implemented. If you really need this, please contact the "
					"authors and request implementation of cross-grid interpolation. Following "
					"are the two grids as interpreted by this code.",
					grid, calculator.gridReader
				);
				
				// Then download data				
				return getActiveThread().dataService().download(cField.getData())
				.then([&calculator, scale](LocalDataRef<Float64Tensor> field) {
					return calculator.add(scale, field.get());
				});
			}
			case MagneticField::FILAMENT_FIELD: {
				// Process Biot-Savart field
				auto fField = node.getFilamentField();
				
				return processFilament(calculator, fField.getFilament(), fField.getBiotSavartSettings(), scale * fField.getCurrent() * fField.getWindingNo());
			}
			case MagneticField::SCALE_BY: {
				auto scaleBy = node.getScaleBy();
				
				if(scaleBy.getFactor() == 0)
					return READY_NOW;
				
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
				
				auto myGrid = ID::fromReader(calculator.gridReader);
				auto cachedGrid = ID::fromReader(cached.getComputed().getGrid());
				
				if(myGrid == cachedGrid) {
					return getActiveThread().dataService().download(cached.getComputed().getData())
					.then([&calculator, scale](LocalDataRef<Float64Tensor> field) {
						return calculator.add(scale, field.get());
					});
				}
				
				return processField(calculator, cached.getNested(), scale);
			}
			case MagneticField::AXISYMMETRIC_EQUILIBRIUM: {
				calculator.equilibrium(scale, node.getAxisymmetricEquilibrium());
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
		const size_t nPhi = 4;
		
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