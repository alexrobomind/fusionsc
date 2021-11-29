#include "magnetics.h"
#include "tensor.h"
#include "data.h"
#include <cmath>

#include <kj/map.h>


namespace fsc { 
	
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
	
	Vec3<double> phizr(int i_phi, int i_z, int i_r) {
		double r = rMin + (rMax - rMin) / (nR - 1) * i_r;
		double z = zMin + (zMax - zMin) / (nZ - 1) * i_z;
		double phi = 2 * pi / nSym / nPhi;
		
		Vec3<double> result;
		result(0) = phi;
		result(1) = z;
		result(2) = r;
		
		return result;
	}
};

struct FieldCalculation {
	using Vec3d = Vec3<double>;

	using Field = Eigen::Tensor<Vec3d, 3>;
	using FieldRef = Eigen::TensorMap<Field>;
	
	FieldCalculation(ToroidalGrid::Reader in, OffloadDevice& device) :
		_device(device),
		grid(in),
		field(grid.nPhi, grid.nZ, grid.nR),
		mappedField(field, _device)
	{
		Vec3d defaultVal;
		defaultVal.setZero();
		field.setConstant(defaultVal);
		mappedField.updateDevice();
	}
	
	~FieldCalculation() {}
	
	auto& eigenDevice() { return _device.eigenDevice(); }
	
	void add(double scale, Float64Tensor::Reader input) {
		auto shape = input.getShape();
		
		KJ_REQUIRE(shape.size() == 4);
		KJ_REQUIRE(shape[3] == 3);
		
		auto op = [&](int idx) {
			auto offset = 3 * idx;
			
			Vec3d result;
			auto data = input.getData();
			for(int i = 0; i < 3; ++i)
				result[i] = data[offset + i];
			return result;
		};
		
		Vec3d vScale;
		vScale.setConstant(scale);
		field.device(eigenDevice()) += field.nullaryExpr(op) * vScale;
	}
	
	void biotSavart(double current, Float64Tensor::Reader input, BiotSavartSettings::Reader settings) {
		auto shape = input.getShape();
		
		KJ_REQUIRE(shape.size() == 2);
		KJ_REQUIRE(shape[1] == 3);
		KJ_REQUIRE(shape[0] >= 2);
		
		int n_points = (int) shape[0];
		Eigen::Tensor<double, 2> filament(n_points, 3);
		
		// Copy data into native buffer
		auto data = input.getData();
		for(int i = 0; i < n_points; ++i) {
			filament(i, 0) = data[3 * i + 0];
			filament(i, 1) = data[3 * i + 1];
			filament(i, 2) = data[3 * i + 2];
		}
		
		double coilWidth = settings.getWidth();
		double stepSize  = settings.getStepSize();
		
		using i2 = Eigen::array<int, 2>;
		using i1 = Eigen::array<int, 1>;
		
		// Map filament onto device
		MappedTensor<decltype(filament)> mappedFilament(filament, _device);
		mappedFilament.updateDevice();
		
		/*# pragma omp teams distribute parallel for collapse(3)
		for(int i_r   = 0; i_r   < grid.nR  ; ++i_r  ) {
		for(int i_z   = 0; i_z   < grid.nZ  ; ++i_z  ) {
		for(int i_phi = 0; i_phi < grid.nPhi; ++i_phi) {*/

		auto op = [&](int idx) {
			int midx[3];
			// Decode index using column major layout
			// in which the first index has stride 1
			for(int i = 0; i < 3; ++i) {
				midx[i] = idx % mappedField.dimension(i);
				idx /= mappedField.dimension(i);
			}
			
			int i_phi = midx[0];
			int i_z   = midx[1];
			int i_r   = midx[2];

			Vec3d field_cartesian;
			field_cartesian.setZero();

			Vec3d x_grid = grid.xyz(i_phi, i_z, i_r);
			
			for(int i_fil = 0; i_fil < n_points - 1; ++i_fil) {
				// Extract current filament
				Vec3d x1 = mappedFilament.chip(i_fil, 0);
				Vec3d x2 = mappedFilament.chip(i_fil + 1, 0);
				
				// Calculate step and no. of steps
				auto dxtot = (x2 - (Vec3d) x1).eval();
				double dxnorm = norm(dxtot);
				int n_steps = (int) (dxnorm / stepSize + 1);
				Vec3d dx = dxtot * (1.0 / n_steps);
				
				for(int i_step = 0; i_step < n_steps; ++i_step) {
					auto x = x1 + (x2 - x1) * ((double) i_step) * dx;
					
					auto dr = (x_grid - x).eval();
					auto distance = dr.square().sum().sqrt();
					auto useDistance = distance.cwiseMax(distance.constant(coilWidth)).eval();
					
					constexpr double mu0over4pi = 1e-7;
					field_cartesian += current * mu0over4pi * cross(dx, dr) / (useDistance * useDistance * useDistance);
				}
			}
			
			Vec3d phizr = grid.phizr(i_phi, i_z, i_r);
			double phi = phizr(0);
			double fieldR   = field_cartesian(0) * cos(phi) + field_cartesian(1) * sin(phi);
			double fieldZ   = field_cartesian(2);
			double fieldPhi = field_cartesian(1) * cos(phi) - field_cartesian(0) * sin(phi);
			
			Vec3d result;
			result(0) = fieldPhi;
			result(1) = fieldZ;
			result(2) = fieldR;
			return result;
		};

		mappedField.device(eigenDevice()) += field.nullaryExpr(op);
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
				data.set(3 * i + j, (fData + i)->operator()(j));
			}
		}
	}
	
	OffloadDevice& _device;
	Field field;
	MappedTensor<Field> mappedField;
	GridData grid;
};


struct CalculationSession : public FieldCalculationSession::Server, kj::Refcounted {
	OffloadDevice device;
	
	Temporary<ToroidalGrid> grid;
	LibraryThread lt;
	kj::TreeMap<ID, kj::ForkedPromise<LocalDataRef<Float64Tensor>>> fieldCache;
	
	Promise<LocalDataRef<Float64Tensor>> processRoot(MagneticField::Reader node) {
		auto newCalculator = kj::heap<FieldCalculation>(grid, device);
		auto calcDone = processField(*newCalculator, node, 1);
		
		return calcDone.then([newCalculator = mv(newCalculator), this]() mutable {							
			Temporary<Float64Tensor> result;
			newCalculator->finish(result);
			return lt->dataService().publish(lt->randomID(), result.asReader());
		});
	}
	
	Promise<void> processFilament(FieldCalculation& calculator, Filament::Reader node, BiotSavartSettings::Reader settings, double scale) {
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
	
	Promise<void> processField(FieldCalculation& calculator, MagneticField::Reader node, double scale) {
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

void dummyFieldcalc() {
}

}