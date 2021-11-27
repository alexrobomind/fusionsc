#include "magnetics.h"
#include "tensor.h"
#include <cmath>


namespace fsc { namespace {
	
constexpr double pi = 3.14159265358979323846;
	
struct GridData {
	double rMin; double rMax; int nR;
	double zMin; double zMax; int nZ;
	int nSym; int nPhi;
	
	void setFrom(ToroidalGrid::Reader in) {
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

class FieldCalculation {
	using Field = Eigen::Tensor<double, 4>;
	
	FieldCalculation(ToroidalGrid::Reader in) {
		// Initialize grid and field
		grid.setFrom(in);
		field = Field(grid.nPhi, grid.nZ, grid.nR, 3);
		
		// Initialize field on device
		# pragma omp target enter data map(alloc: field) map(to: grid)
		# pragma omp target
		{
			field.setZero();
		}
	}
	
	Field& getField() {
		# pragma omp target update from(field)
		return field;
	}
	
	~FieldCalculation() {
		# pragma omp target exit data map(release: field, grid)
	}
	
	void run(double current, Float64Tensor::Reader input, BiotSavartSettings::Reader settings) {
		using Vec3d = Vec3<double>;
		
		{
			auto shape = input.getShape();
			KJ_REQUIRE(shape.size() == 2);
			KJ_REQUIRE(shape[1] == 3);
			KJ_REQUIRE(shape[0] >= 2);
		}
		
		int n_points = (int) input.getShape()[0];
		Eigen::Tensor<double, 2> filament(n_points, 3);
		
		// Copy data into machine buffer
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
				
		// Start offloading onto device
		# pragma omp target data map(to: filament) 
		{
			# pragma omp teams distribute parallel for collapse(3)
			for(int i_r   = 0; i_r   < grid.nR  ; ++i_r  ) {
			for(int i_z   = 0; i_z   < grid.nZ  ; ++i_z  ) {
			for(int i_phi = 0; i_phi < grid.nPhi; ++i_phi) {
				Vec3d field_cartesian; field_cartesian.setZero();
				Vec3d x_grid = grid.xyz(i_phi, i_z, i_r);
				
				// TODO: Specify that this loop is the same in every thread
				for(int i_fil = 0; i_fil < filament.dimension(0) - 1; ++i_fil) {
					// Calculate no. of substeps
					auto filamentPoint = [&](int i) -> Vec3d {
						return filament.slice(i2({i, 0}), i2({1, 3})).reshape(i1({3}));
					};
					Vec3d x1 = filamentPoint(i_fil);
					Vec3d x2 = filamentPoint(i_fil + 1);
					
					auto dxtot = x2 - x1;
					double dxnorm = norm(dxtot);
					int n_steps = (int) (dxnorm / stepSize + 1);
					Vec3d dx = dxtot * (1.0 / n_steps);
					
					// TODO: Specify that this loop is the same in every thread
					for(int i_step = 0; i_step < n_steps; ++i_step) {
						Vec3d x = x1 + (x2 - x1) * ((double) i_step) * dx;
						
						auto dr = x_grid - x;
						auto distance = dr.square().sum().sqrt();
						auto useDistance = distance.cwiseMax(distance.constant(coilWidth));
						
						// TODO: Factor out scalars?
						constexpr double mu0over4pi = 1e-7;
						field_cartesian += current * mu0over4pi * cross(dx, dr) / (useDistance * useDistance * useDistance);
					}
				}
				
				Vec3d phizr = grid.phizr(i_phi, i_z, i_r);
				double phi = phizr(0);
				double fieldR   = field_cartesian(0) * cos(phi) + field_cartesian(1) * sin(phi);
				double fieldZ   = field_cartesian(2);
				double fieldPhi = field_cartesian(1) * cos(phi) - field_cartesian(0) * sin(phi);
				
				field(i_phi, i_z, i_r, 0) += fieldPhi;
				field(i_phi, i_z, i_r, 1) += fieldZ;
				field(i_phi, i_z, i_r, 2) += fieldR;
			}}}
		}
	}
	
private:
	Field field;
	GridData grid;
};

}}