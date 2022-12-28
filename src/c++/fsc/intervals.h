#pragma once

namespace fsc {

struct UnbalancedIntervalSplit {
	inline UnbalancedIntervalSplit(size_t nTotal, size_t blockSize) :
		nTotal(nTotal), blockSize(blockSize)
	{}
	
	inline size_t blockCount() {
		return (nTotal + blockSize - 1) / blockSize;
	}
	
	inline size_t edge(size_t i) {
		return std::min(nTotal, i * blockSize);
	}
	
	inline size_t interval(size_t i) {
		return i / blockSize;
	}
	
private:
	size_t nTotal;
	size_t blockSize;
};

struct BalancedIntervalSplit {
	inline BalancedIntervalSplit(size_t nTotal, size_t blockSize) :
		nTotal(nTotal), blockSize(blockSize)
	{
		compute();
	}
	
	inline size_t blockCount() {
		return (nTotal + blockSize - 1) / blockSize;
	}
	
	inline size_t interval(size_t i) {
		if(i < blockSize * remainder)
			return i / blockSize;
		
		i -= blockSize * remainder;
		
		return remainder + i / (blockSize - 1);
	}
	
	inline size_t edge(size_t i) {
		if(i < remainder)
			return blockSize * i;
		
		i -= remainder;
		
		return remainder * blockSize + i * (blockSize - 1);
	}
	
private:
	size_t nTotal;
	size_t blockSize;
	size_t remainder;
	
	inline void compute() {
		remainder = blockCount() - (blockSize * blockCount() - nTotal);
	}
};

}