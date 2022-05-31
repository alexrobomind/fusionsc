# pragma once

#include "magnetics.h"
#include "kernels.h"
#include "kernels-biotsavart.h"

namespace fsc { namespace internal {
	
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
struct CalculationSession : public FieldCalculationSession::Server {	
	constexpr static unsigned int GRID_VERSION = 7;
	
	// Device device;
	Device& device;
	
	ToroidalGridStruct grid;
	LibraryThread lt;
	kj::TreeMap<ID, kj::ForkedPromise<LocalDataRef<Float64Tensor>>> fieldCache;
	
	CalculationSession(Device& device, ToroidalGrid::Reader newGrid, LibraryThread& lt) :
		device(device),
		grid(readGrid(newGrid, GRID_VERSION)),
		lt(lt->addRef())
	{}
	
	Promise<void> compute(ComputeContext context) {
		return processRoot(context.getParams().getField())
		.then([this, context](LocalDataRef<Float64Tensor> tensorRef) mutable {
			auto compField = context.getResults().initComputedField();
			
			writeGrid(grid, compField.initGrid());
			compField.setData(tensorRef);
		}).attach(thisCap());
	}
	
	Promise<LocalDataRef<Float64Tensor>> processRoot(MagneticField::Reader node) {
		auto newCalculator = kj::heap<FieldCalculation<Device>>(grid, device);
		auto calcDone = processField(*newCalculator, node, 1);
		
		return calcDone.then([newCalculator = mv(newCalculator), this]() mutable {		
			auto result = kj::heap<Temporary<Float64Tensor>>();					
			auto readout = newCalculator->finish(*result);
			auto publish = readout.then([result=result.get(), this]() {
				return lt->dataService().publish(lt->randomID(), result->asReader());
			});
			return publish.attach(mv(result), thisCap(), mv(newCalculator));
		});
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
					// Then check if there is a calculation for it (running or finished)
					ID id = local.getID();
					
					auto addComputed = [&calculator, scale](auto& cacheEntry) {
						return cacheEntry.addBranch().then([&calculator, scale](auto localRef) {
							calculator.add(scale, localRef.get());
						});
					};
					
					KJ_IF_MAYBE(entry, fieldCache.find(id)) {
						// If yes, add field as soon as calculation is finished
						return addComputed(*entry);
					}
					
					// Otherwise, we need to schedule a new calculation ...
					auto result = kj::evalLater([this, local]() mutable {
						return processRoot(local.get()).attach(cp(local));
					});
					
					// ... and then reference its result
					auto& entry = fieldCache.insert(id, result.fork());
					return addComputed(entry.value);
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
				return lt->dataService().download(cField.getData()).then([&calculator, scale](LocalDataRef<Float64Tensor> field) {
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
	}
};

template<typename Device>
struct FieldCalculatorImpl : public FieldCalculator::Server {
	LibraryThread lt;
	Own<Device> device;
	
	FieldCalculatorImpl(LibraryThread& lt, Own<Device> device) :
		lt(lt -> addRef()),
		device(mv(device))
	{}
	
	Promise<void> get(GetContext context) {		
		FieldCalculationSession::Client newClient(
			kj::heap<internal::CalculationSession<Device>>(
				*device,
				context.getParams().getGrid(),
				lt
			).attach(thisCap())
		);
		context.getResults().setSession(newClient);
		return READY_NOW;
	}
};
	
}}