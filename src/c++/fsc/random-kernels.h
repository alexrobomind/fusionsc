#include <cupnp/cupnp.h>

namespace fsc {

struct MT19937 {
	static constexpr unsigned int N = 624;
	static constexpr unsigned int M = 397;
	
	uint32_t state[N];
	uint16_t index = 0;
	
	CUPNP_FUNCTION MT19937(const cu::MT19937State input) :
		index(input.getIndex())
	{
		for(unsigned int i = 0; i < N; ++i) {
			state[i] = input.getVector()[i];
		}
	}
	
	CUPNP_FUNCTION void save(cu::MT19937State output) {
		output.setIndex(index);
		for(unsigned int i = 0; i < N; ++i)
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
	
	CUPNP_FUNCTION uint32_t operator() {
		if(index >= N) {
			update();
			index = 0;
		}
		
		return state[index++];
	}
	
	CUPNP_FUNCTION double uniform() {
		constexpr double scale = 1 / ( 1 << 32 + 1 );
		return static_cast<double>(operator() + 1) * scale;
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
};

}