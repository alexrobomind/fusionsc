#include "magnetics.h"
#include "tensor.h"
#include "data.h"
#include <cmath>

#include <kj/map.h>
#include <kj/refcount.h>


namespace fsc { 

using Vec3d = Vec3<double>;

using Field = Eigen::Tensor<double, 4>;
using FieldRef = Eigen::TensorMap<Field>;

using MFilament = Eigen::Tensor<double, 2>;
using FilamentRef = Eigen::TensorMap<MFilament>;
	
constexpr double pi = 3.14159265358979323846;
	
struct GridData {
	double rMin; double rMax; int nR;
	double zMin; double zMax; int nZ;
	int nSym; int nPhi;
	
	GridData() = default;
	GridData(ToroidalGrid::Reader in) {
		rMin = in.getRMin();
		rMax = in.getRMax();
		zMin = in.getZMin();
		zMax = in.getZMax();
		nR = (int) in.getNR();
		nZ = (int) in.getNZ();
		nSym = (int) in.getNSym();
		nPhi = (int) in.getNPhi();
		
		KJ_REQUIRE(nR >= 2);
		KJ_REQUIRE(nZ >= 2);
	}
	
	Vec3<double> xyz(int i_phi, int i_z, int i_r) {
		double r = rMin + (rMax - rMin) / (nR - 1) * i_r;
		double z = zMin + (zMax - zMin) / (nZ - 1) * i_z;
		double phi = 2 * pi / nSym / nPhi;
		
		double x = r * cos(phi);
		double y = r * sin(phi);
		
		Vec3<double> result;
		result(0) = x;
		result(1) = y;
		result(2) = z;
		
		return result;
	}
	
	double phi(int i_phi) {
		return 2 * pi / nSym / nPhi;
	}
	
	Vec3<double> phizr(int i_phi, int i_z, int i_r) {
		double r = rMin + (rMax - rMin) / (nR - 1) * i_r;
		double z = zMin + (zMax - zMin) / (nZ - 1) * i_z;
		double vphi = phi(i_phi);
		
		Vec3<double> result;
		result(0) = vphi;
		result(1) = z;
		result(2) = r;
		
		return result;
	}
};

// Computational kernels

EIGEN_DEVICE_FUNC void biotSavartKernel(unsigned int idx, GridData grid, FilamentRef filament, double current, double coilWidth, double stepSize, FieldRef out) {
	int midx[3];
	
	// Decode index using column major layout
	// in which the first index has stride 1
	for(int i = 0; i < 3; ++i) {
		midx[i] = idx % out.dimension(i);
		idx /= out.dimension(i);
	}
	
	int i_phi = midx[0];
	int i_z   = midx[1];
	int i_r   = midx[2];

	Vec3d x_grid = grid.xyz(i_phi, i_z, i_r);

	Vec3d field_cartesian;
	field_cartesian.setZero();
	
	auto n_points = filament.dimension(1);	
	for(int i_fil = 0; i_fil < n_points - 1; ++i_fil) {
		// Extract current filament
		Vec3d x1 = filament.chip(i_fil, 1);
		Vec3d x2 = filament.chip(i_fil + 1, 1);
		
		// Calculate step and no. of steps
		auto dxtot = (x2 - x1).eval();
		double dxnorm = norm(dxtot);
		int n_steps = (int) (dxnorm / stepSize + 1);
		Vec3d dx = dxtot * (1.0 / n_steps);
		
		for(int i_step = 0; i_step < n_steps; ++i_step) {
			auto x = x1 + (x1 - x1) * ((double) i_step) * dx;
			
			auto dr = (x_grid - x).eval();
			auto distance = dr.square().sum().sqrt();
			auto useDistance = distance.cwiseMax(distance.constant(coilWidth)).eval();
			TensorFixedSize<double, Eigen::Sizes<>> dPow3 = useDistance * useDistance * useDistance;
			
			constexpr double mu0over4pi = 1e-7;
			field_cartesian += mu0over4pi * cross(dx, dr) / dPow3();
		}
	}
	
	double phi = grid.phi(i_phi);
	double fieldR   = field_cartesian(0) * cos(phi) + field_cartesian(1) * sin(phi);
	double fieldZ   = field_cartesian(2);
	double fieldPhi = field_cartesian(1) * cos(phi) - field_cartesian(0) * sin(phi);
	
	double* outData = out.data();
	outData[3 * idx + 0] += current * fieldPhi;
	outData[3 * idx + 1] += current * fieldZ;
	outData[3 * idx + 2] += current * fieldR;
}

// Kernel launcher

void launchBiotSavart(Eigen::ThreadPoolDevice& device, GridData grid, FilamentRef filament, double current, double coilWidth, double stepSize, FieldRef out) {
	# pragma omp parallel for
	for(unsigned int i = 0; i < out.size() / 3; ++i) {
		biotSavartKernel(i, grid, filament, current, coilWidth, stepSize, out);
	}
}

template<typename Device>
struct FieldCalculation {
	Device& _device;
	GridData grid;
	Field field;
	MappedTensor<Field, Device> mappedField;
	
	FieldCalculation(ToroidalGrid::Reader in, Device& device) :
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
		Field newField(3, grid.nPhi, grid.nZ, grid.nR);
		for(int i = 0; i < newField.size() / 3; ++i) {
			newField.data()[i] = data[i];
		}
		
		// Map field onto GPU
		MappedTensor<Field, Device> mField(newField, _device);
		mField.updateDevice();
		
		// Call addition
		field.device(_device) += mField * scale;
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
		
		using i2 = Eigen::array<int, 2>;
		using i1 = Eigen::array<int, 1>;
		
		// Map filament onto device
		MappedTensor<decltype(filament), Device> mappedFilament(filament, _device);
		mappedFilament.updateDevice();
		
		launchBiotSavart(_device, grid, mappedFilament, current, coilWidth, stepSize, mappedField);
	}
	
	void finish(Float64Tensor::Builder out) {
		mappedField.updateHost();
		
		auto shape = out.initShape(4);
		shape.set(0, grid.nPhi);
		shape.set(1, grid.nZ);
		shape.set(2, grid.nR);
		shape.set(3, 3);
		
		auto data = out.initData(3 * field.size());
		auto fData = field.data();
		
		for(int i = 0; i < field.size(); ++i) {
			for(int j = 0; j < 3; ++j) {
				data.set(3 * i + j, fData[3 * i + j]);
			}
		}
	}
};


struct CalculationSession : public FieldCalculationSession::Server, kj::Refcounted {
	using Device = Eigen::ThreadPoolDevice;
	
	// Device device;
	Eigen::ThreadPool pool;
	Eigen::ThreadPoolDevice device;
	
	Temporary<ToroidalGrid> grid;
	LibraryThread lt;
	kj::TreeMap<ID, kj::ForkedPromise<LocalDataRef<Float64Tensor>>> fieldCache;
	
	CalculationSession(ToroidalGrid::Reader newGrid, LibraryThread& lt) :
		pool(numThreads()),
		device(&pool, numThreads()),
		grid(newGrid),
		lt(lt->addRef())
	{}
	
	Promise<void> compute(ComputeContext context) {
		return processRoot(context.getParams().getField())
		.then([this, context](LocalDataRef<Float64Tensor> tensorRef) mutable {
			auto compField = context.getResults().initComputedField();
			
			compField.setGrid(grid);
			compField.setData(tensorRef);
		});
	}
	
	Promise<LocalDataRef<Float64Tensor>> processRoot(MagneticField::Reader node) {
		auto newCalculator = kj::heap<FieldCalculation<Device>>(grid, device);
		auto calcDone = processField(*newCalculator, node, 1);
		
		return calcDone.then([newCalculator = mv(newCalculator), this]() mutable {							
			Temporary<Float64Tensor> result;
			newCalculator->finish(result);
			return lt->dataService().publish(lt->randomID(), result.asReader());
		});
	}
	
	Promise<void> processFilament(FieldCalculation<Device>& calculator, Filament::Reader node, BiotSavartSettings::Reader settings, double scale) {
		switch(node.which()) {
			case Filament::INLINE:
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
				for(auto newNode : node.getSum()) {
					processField(calculator, newNode, scale);
				}
				return READY_NOW;
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
				
				KJ_REQUIRE(ID::fromReader(grid) == ID::fromReader(this->grid.asReader()));
				
				// Then download data				
				return lt->dataService().download(cField.getData()).then([&calculator, scale](LocalDataRef<Float64Tensor> field) {
					calculator.add(scale, field.get());
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
			default:
				KJ_FAIL_REQUIRE("Unknown magnetic field node encountered. This either indicates that a device-specific node was not resolved, or a generic node from a future library version was presented");
		}
	}
};

struct FieldCalculatorImpl : public FieldCalculator::Server {
	LibraryThread lt;
	
	FieldCalculatorImpl(LibraryThread& lt) :
		lt(lt -> addRef())
	{}
	
	Promise<void> get(GetContext context) {
		FieldCalculationSession::Client newClient(
			kj::refcounted<CalculationSession>(
				context.getParams().getGrid(),
				lt
			)
		);
		context.getResults().setSession(newClient);
		return READY_NOW;
	}
};

FieldCalculator::Client newFieldCalculator(LibraryThread& lt) {
	return FieldCalculator::Client(
		kj::heap<FieldCalculatorImpl>(lt)
	);
}

}