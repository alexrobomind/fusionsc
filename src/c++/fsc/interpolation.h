#pragma once

#include "tensor.h"

namespace fsc {

/** Function defined over an R-Z-phi grid */
template<typename Num, typename Data>
struct ToroidalInterpolator {
private:
	static constexpr size_t wrap_slices = 1; // Number of phi-entries added by wrapping

	template<typename In>
	static auto wrap_phi(const In& input) {
		TensorRef<Tensor<Num, 3>> wrappedInput = input;
		size_t nr = wrappedInput.dimensions()[0];
		size_t nz = wrappedInput.dimensions()[1];

		using Idx = typename Eigen::Tensor<Num, 3>::Index;
		using Dims = typename Eigen::Tensor<Num, 3>::Dimensions;

		const Dims pos(0, 0, 0);
		const Dims len(nr, nz, 1);
		auto phi_slice = input.slice(pos, len);

		return input.concatenate(phi_slice, 2);
	}

	Num reduce_angle(Num in) {
		const Num pi = 3.14159265358979323846;
		const Num wrap = 2 * pi / mtor;

		const Num base = wrap * floor(in / wrap);
		return (in - base) / wrap;
	}
	
	size_t mtor;

	Num r_min;
	Num r_max;
	Num z_min;
	Num z_max;

	using WrappedExpr = decltype(wrap_phi(std::declval<Data>()));
	WrappedExpr data;
	
public:
	ToroidalInterpolator(
		size_t mtor, Num r_min, Num r_max, Num z_min, Num z_max, const Data& ndata
	) :
		mtor(mtor), r_min(r_min), r_max(r_max), z_min(z_min), z_max(z_max),
		data(wrap_phi(ndata))
	{}

	Num eval_phizr(Num phi, Num z, Num r) {
		// If we have a NAN input, return a NAN output
		if(phi != phi || z != z || r != r)
			return std::nan("");
		
		TensorRef<Tensor<Num, 3>> data = this->data;

		size_t nr = data.dimensions()[0];
		size_t nz = data.dimensions()[1];
		size_t nphi = data.dimensions()[2] - wrap_slices;

		Num cr = (r - r_min) / (r_max - r_min) * (nr - 1);
		Num cz = (z - z_min) / (z_max - z_min) * (nr - 1);
		Num cphi = reduce_angle(phi) * nphi;

		if(cr < 0 || cz < 0)
			return std::nan("");

		if(cphi < 0)
			throw std::logic_error(
				"Internal error: cphi (" + std::to_string(cphi) + ") < 0" +
				", phi value is " + std::to_string(phi) +
				", mtor is " + std::to_string(mtor) +
				", reduced angle is " + std::to_string(reduce_angle(phi))
			);

		int64_t i_r = (int64_t) floor(cr);
		int64_t i_z = (int64_t) floor(cz);
		int64_t i_phi = (int64_t) floor(cphi);

		if(i_r >= nr - 1 || i_z >= nz - 1)
			return std::nan("");

		if(i_phi >= nphi + wrap_slices - 1)
			throw std::logic_error(
				"Internal error: i_phi (" + std::to_string(i_phi) + ") exceeded grid maximum " + std::to_string(nphi + wrap_slices - 1) +
				", phi value is " + std::to_string(phi) +
				", cphi is " + std::to_string(cphi) +
				", mtor is " + std::to_string(mtor) +
				", reduced angle is " + std::to_string(reduce_angle(phi))
			);

		Num lr = cr - i_r;
		Num lz = cz - i_z;
		Num lphi = cphi - i_phi;

		return
			(1 - lr) * (1 - lz) * (1 - lphi) * data(i_r    , i_z    , i_phi    ) + 
			lr       * (1 - lz) * (1 - lphi) * data(i_r + 1, i_z    , i_phi    ) +
			(1 - lr) * lz       * (1 - lphi) * data(i_r    , i_z + 1, i_phi    ) +
			lr       * lz       * (1 - lphi) * data(i_r + 1, i_z + 1, i_phi    ) +
			(1 - lr) * (1 - lz) * lphi       * data(i_r    , i_z    , i_phi + 1) + 
			lr       * (1 - lz) * lphi       * data(i_r + 1, i_z    , i_phi + 1) +
			(1 - lr) * lz       * lphi       * data(i_r    , i_z + 1, i_phi + 1) +
			lr       * lz       * lphi       * data(i_r + 1, i_z + 1, i_phi + 1)
		;
	}

	Num eval_xyz(Num x, Num y, Num z) {
		Num r = std::sqrt(x*x + y*y);
		Num phi = atan2(y, x);

		return eval_phizr(phi, z, r);
	}

	Num eval_xyz(const std::array<Num, 3>& x) {
		return eval_xyz(x[0], x[1], x[2]);
	}
	
	Num operator()(const Vec3<Num>& xyz) {
		return eval_xyz(xyz[0], xyz[1], xyz[2]);
	}
};

template<typename Num, typename Data>
ToroidalInterpolator<Num, Data> interpolateToroidal(size_t mtor, Num r_min, Num r_max, Num z_min, Num z_max, const Data& data) {
	return ToroidalInterpolator<Num, Data>(mtor, r_min, r_max, z_min, z_max, data);
}


template<typename Num, typename Expr, bool normalize>
struct SlabCoordinateField {
	Expr expr;
	
	size_t mtor;

	Num r_min;
	Num r_max;
	Num z_min;
	Num z_max;

	SlabCoordinateField(
		size_t mtor, Num r_min, Num r_max, Num z_min, Num z_max,
		const Expr& expr
	):
		expr(expr), mtor(mtor), r_min(r_min), r_max(r_max), z_min(z_min), z_max(z_max)
	{}

	// Returns the magnetic field (optionally normalized) at a given phi, r, z position in cartesian components
	Vec3<Num> eval_phizr(Num phi, Num z, Num r) {
		using std::sin;
		using std::cos;
		
		auto Bphi = interpolateToroidal(mtor, r_min, r_max, z_min, z_max, expr.chip(0, 0));
		auto Bz =   interpolateToroidal(mtor, r_min, r_max, z_min, z_max, expr.chip(0, 1));
		auto Br =   interpolateToroidal(mtor, r_min, r_max, z_min, z_max, expr.chip(0, 2));

		Num Bval_r   = Br  .eval_phizr(phi, z, r);
		Num Bval_z   = Bz  .eval_phizr(phi, z, r);
		Num Bval_phi = Bphi.eval_phizr(phi, z, r);

		if(normalize) {
			Num norm = 1 / std::sqrt(Bval_r * Bval_r + Bval_z * Bval_z + Bval_phi * Bval_phi);
	
			Bval_r *= norm;
			Bval_z *= norm;
			Bval_phi *= norm;
		}

		Vec2<Num> e_r = {cos(phi), sin(phi)};
		Vec2<Num> e_phi = {-sin(phi), cos(phi)};

		return {
			Bval_r * e_r[0] + Bval_phi * e_phi[0],
			Bval_r * e_r[1] + Bval_phi * e_phi[1],
			Bval_z
		};
	}

	Vec3<Num> eval_xyz(Num x, Num y, Num z) {
		Num r = std::sqrt(x*x + y*y);
		Num phi = std::atan2(y, x);

		return eval_phizr(phi, z, r);
	}
	
	Vec3<Num> operator()(const Vec3<Num>& xyz) {
		return eval_xyz(xyz[0], xyz[1], xyz[2]);
	}
};

template<bool normalize, typename Num, typename Expr>
SlabCoordinateField<Num, Expr, normalize> slabCoordinateField(
	size_t mtor, Num r_min, Num r_max, Num z_min, Num z_max,
	const Expr& expr
) {
	return SlabCoordinateField<Num, Expr, normalize>(mtor, r_min, r_max, z_min, z_max, expr);
}

}
