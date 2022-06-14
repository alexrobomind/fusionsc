#include <capnp/message.h>

#include "magnetics.h"
#include "data.h"
#include "kernels.h"
#include "kernels-biotsavart.h"

namespace fsc {
	
namespace {
	
template<typename Device>
struct FieldCalculation {
	using Field = ::fsc::kernels::Field;
	using MFilament = ::fsc::kernels::MFilament;
	
	Device& _device;
	ToroidalGridStruct grid;
	Field field;
	MapToDevice<Field, Device> mappedField;
	
	// This promise makes sure we only schedule a
	// calculation once the previous is finished
	Promise<void> calculation = READY_NOW;
	
	FieldCalculation(ToroidalGridStruct in, Device& device) :
		_device(device),
		grid(in),
		field(3, grid.nPhi, grid.nZ, grid.nR),
		mappedField(field, _device)
	{
		field.setZero();
		mappedField.updateDevice();
	}
	
	~FieldCalculation() {}
	
	void add(double scale, Float64Tensor::Reader input) {
		auto shape = input.getShape();
		
		KJ_REQUIRE(shape.size() == 4);
		KJ_REQUIRE(shape[3] == 3);
		
		auto data = input.getData();
		
		// Write field into native format
		auto newField = kj::heap<Field>(3, grid.nPhi, grid.nZ, grid.nR);
		for(int i = 0; i < newField->size(); ++i) {
			newField->data()[i] = data[i];
		}
		
		calculation = FSC_LAUNCH_KERNEL(
			kernels::addFieldKernel,
			_device, 
			mv(calculation),
			field.size(),
			FSC_KARG(mappedField, NOCOPY), FSC_KARG(*newField, IN), scale
		);
		calculation = calculation.attach(mv(newField));
	}
	
	void biotSavart(double current, Float64Tensor::Reader input, BiotSavartSettings::Reader settings) {
		auto shape = input.getShape();
		
		KJ_REQUIRE(shape.size() == 2);
		KJ_REQUIRE(shape[1] == 3);
		KJ_REQUIRE(shape[0] >= 2);
		
		int n_points = (int) shape[0];
		auto filament = kj::heap<MFilament>(3, n_points);
		
		// Copy filament into native buffer
		auto data = input.getData();
		for(int i = 0; i < n_points; ++i) {
			(*filament)(0, i) = data[3 * i + 0];
			(*filament)(1, i) = data[3 * i + 1];
			(*filament)(2, i) = data[3 * i + 2];
		}
		
		double coilWidth = settings.getWidth();
		double stepSize  = settings.getStepSize();
		
		KJ_REQUIRE(stepSize != 0, "Please specify a step size in the Biot-Savart settings");
		
		// Launch calculation
		calculation = FSC_LAUNCH_KERNEL(
			kernels::biotSavartKernel,
			_device,
			mv(calculation),
			field.size() / 3,
			grid, FSC_KARG(*filament, IN), current, coilWidth, stepSize, FSC_KARG(mappedField, NOCOPY)
		);
		calculation = calculation.attach(mv(filament));
	}
	
	Promise<void> finish(Float64Tensor::Builder out) {
		calculation = calculation
		.then([this]() {
			mappedField.updateHost();
			return hostMemSynchronize(_device);
		})
		.then([this, out]() {
			writeTensor(field, out);
		});
		
		return mv(calculation);
	}
};

template<typename Device>
struct CalculationSession : public FieldCalculator::Server {
	constexpr static unsigned int GRID_VERSION = 7;
	
	// Device device;
	Device& device;
	
	ToroidalGridStruct grid;
	LibraryThread lt;
	Cache<ID, LocalDataRef<Float64Tensor>> cache;
	
	CalculationSession(Device& device, ToroidalGrid::Reader newGrid, LibraryThread& lt) :
		device(device),
		grid(readGrid(newGrid, GRID_VERSION)),
		lt(lt->addRef())
	{}
	
	//! Handles compute request
	Promise<void> compute(ComputeContext context) {
		context.allowCancellation();
		
		// Copy input field (so that call context can be released)
		auto field = heapHeld<Temporary<MagneticField>>(context.getParams().getField());
		context.releaseParams();
		
		// Start calculation lazily
		auto data = processRoot(*field)
		.then([this, field](LocalDataRef<Float64Tensor> tensorRef) mutable {
			// Cache field if not present, use existing if present
			return ID::fromReaderWithRefs(field->asBuilder())
			.then([this, tensorRef = mv(tensorRef)](ID id) mutable -> DataRef<Float64Tensor>::Client {
				auto insertResult = cache.insert(id, mv(tensorRef));
				return attach(mv(insertResult.element), mv(insertResult.ref));
			});
		}).attach(thisCap(), field.x());
		
		// Fill in computed grid struct
		auto compField = context.initResults().initComputedField();
		writeGrid(grid, compField.initGrid());
		compField.setData(mv(data));
		
		return READY_NOW;
	}
	
	//! Processes a root node of a magnetic field (creates calculator)
	Promise<LocalDataRef<Float64Tensor>> processRoot(MagneticField::Reader node) {
		auto newCalculator = heapHeld<FieldCalculation<Device>>(grid, device);
		auto calcDone = processField(*newCalculator, node, 1);
		
		return calcDone.then([newCalculator, this]() mutable {		
			auto result = kj::heap<Temporary<Float64Tensor>>();					
			auto readout = newCalculator->finish(*result);
			auto publish = readout.then([result=result.get(), this]() {
				return lt->dataService().publish(lt->randomID(), result->asReader());
			});
			return publish.attach(mv(result));
		}).attach(thisCap(), newCalculator.x());
	}
	
	Promise<void> processFilament(FieldCalculation<Device>& calculator, Filament::Reader node, BiotSavartSettings::Reader settings, double scale) {
		switch(node.which()) {
			case Filament::INLINE:
				// The biot savart operation is chained by the calculator
				calculator.biotSavart(scale, node.getInline(), settings);
				return READY_NOW;
				
			case Filament::REF:
				return lt->dataService().download(node.getRef()).then([&calculator, settings, scale, this](LocalDataRef<Filament> local) mutable {
					return processFilament(calculator, local.get(), settings, scale).attach(cp(local));
				});
			default:
				KJ_FAIL_REQUIRE("Unknown filament node encountered. This either indicates that a device-specific node was not resolved, or a generic node from a future library version was presented");
		}
	}
	
	Promise<void> processField(FieldCalculation<Device>& calculator, MagneticField::Reader node, double scale) {
		return ID::fromReaderWithRefs(node).then([this, &calculator, node, scale](ID id) -> Promise<void> {
			// Check if the node is in the cache
			KJ_IF_MAYBE(pFieldRef, cache.find(id)) {
				calculator.add(scale, pFieldRef->get());
				return READY_NOW;
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
					return lt -> dataService().download(ref)
					.then([&calculator, scale, this](LocalDataRef<MagneticField> local) {
						// Then process it like usual
						return processField(calculator, local.get(), scale).attach(cp(local));
					});
				}
				case MagneticField::COMPUTED_FIELD: {
					// First check grid compatibility
					auto cField = node.getComputedField();
					auto grid = cField.getGrid();
					
					Temporary<ToroidalGrid> myGrid;
					writeGrid(this->grid, myGrid);
					KJ_REQUIRE(ID::fromReader(grid) == ID::fromReader(myGrid.asReader()));
					
					// Then download data				
					return lt->dataService().download(cField.getData()).
					then([&calculator, scale](LocalDataRef<Float64Tensor> field) {
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
				default:
					KJ_FAIL_REQUIRE("Unknown magnetic field node encountered. This either indicates that a device-specific node was not resolved, or a generic node from a future library version was presented");
			}
		});
	}
};

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
	
	KJ_REQUIRE(out.isValid());
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

FieldResolverBase::FieldResolverBase(LibraryThread& lt) : lt(lt->addRef()) {}

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
			
			return lt->dataService().download(input.getRef())
			.then([this, newOutput, context] (LocalDataRef<MagneticField> ref) mutable {
				return processField(ref.get(), newOutput, context);
			}).then([this, output, newOutput]() mutable {
				return output.setRef(lt->dataService().publish(lt->randomID(), newOutput));
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
		default:
			return kj::READY_NOW;	
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
			
			return lt->dataService().download(input.getRef())
			.then([this, newOutput, context] (LocalDataRef<Filament> ref) mutable {
				return processFilament(ref.get(), newOutput, context);
			}).then([this, output, newOutput]() mutable {
				return output.setRef(lt->dataService().publish(lt->randomID(), newOutput));
			}).attach(mv(tmpMessage), thisCap());
		}
		default:
			return kj::READY_NOW;
	}
}

FieldCalculator::Client newFieldCalculator(LibraryThread& lt, ToroidalGrid::Reader grid, Own<Eigen::ThreadPoolDevice> dev) {
	auto& devRef = *dev;
	return FieldCalculator::Client(
		kj::heap<CalculationSession<Eigen::ThreadPoolDevice>>(devRef, grid, lt).attach(mv(dev))
	);
}

#ifdef FSC_WITH_CUDA

#include <cuda_runtime_api.h>

FieldCalculator::Client newFieldCalculator(LibraryThread& lt, ToroidalGrid::Reader grid, Own<Eigen::GpuDevice> dev) {
	auto& devRef = *dev;
	return FieldCalculator::Client(
		kj::heap<CalculationSession<Eigen::GpuDevice>>(devRef, grid, lt).attach(mv(dev))
	);
}

#endif

}