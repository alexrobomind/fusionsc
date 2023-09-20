#include <cupnp/cupnp.h>

namespace fsc {

//! Implements the popular 32 bit Mersenne twister 19937 pseudo-random number generator
struct MT19937 {
	static constexpr int N = 624;
	static constexpr int M = 397;
	
	uint32_t state[N];
	uint16_t index = 0;
	
	CUPNP_FUNCTION MT19937(cu::MT19937State::Reader input) :
		index(input.getIndex())
	{
		for(int i = 0; i < N; ++i) {
			state[i] = input.getVector()[i];
		}
	}
	
	CUPNP_FUNCTION void save(cu::MT19937State::Builder output) {
		output.setIndex(index);
		for(int i = 0; i < N; ++i)
			output.mutateVector().set(i, state[i]);
	}
	
	// Based on DE wikipedia for mersenne twister
	CUPNP_FUNCTION void update() {
		static const uint32_t  A[2] = { 0, 0x9908B0DF };
		int                    i = 0;

		for (; i < N-M; i++)
			state[i] = state[i+(M  )] ^ (((state[i  ] & 0x80000000) | (state[i+1] & 0x7FFFFFFF)) >> 1) ^ A[state[i+1] & 1];
		
		for (; i < N-1; i++)
			state[i] = state[i+(M-N)] ^ (((state[i  ] & 0x80000000) | (state[i+1] & 0x7FFFFFFF)) >> 1) ^ A[state[i+1] & 1];
		
		state[N-1] = state[M-1]     ^ (((state[N-1] & 0x80000000) | (state[0  ] & 0x7FFFFFFF)) >> 1) ^ A[state[0  ] & 1];
	}
	
	CUPNP_FUNCTION uint32_t operator()() {
		if(index >= N) {
			update();
			index = 0;
		}
		
		uint32_t value = state[index++];
		
		// Tempering of the distribution
		value ^= (value >> 11);
		value ^= (value <<  7) & 0x9D2C5680;
		value ^= (value << 15) & 0xEFC60000;
		value ^= (value >> 18);
		
		return value;
	}
	
	CUPNP_FUNCTION double uniform() {
		constexpr double scale = ((double) 1) / ( (((uint64_t) 1) << 32) + 1 );
		return static_cast<double>((*this)() + 1) * scale;
	}
	
	CUPNP_FUNCTION double exponential() {
		double u = uniform();
		if(u == 0) return 0;
		
		return -std::log(u);
	}
	
	CUPNP_FUNCTION void normalPair(double& n1, double& n2) {
		constexpr double pi = 3.14159265358979323846;
		
		double angle = uniform() * 2 * pi;
		double magnitude = sqrt(2 * exponential());
		
		n1 = cos(angle) * magnitude;
		n2 = sin(angle) * magnitude;
	}
	
	static void seed(uint32_t seed, MT19937State::Builder state) {
		constexpr uint32_t mult = 1812433253ul;
		
		state.setIndex(0);
		auto vector = state.initVector(N);
		
		for(auto i : kj::indices(vector)) {
			vector.set(i, seed);
			seed = mult * (seed ^ (seed >> 30)) + (i+1);
		}
	}
};

}