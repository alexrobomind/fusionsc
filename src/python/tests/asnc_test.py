import fusionsc as fsc
import time

from pytest import approx, fixture


@fixture
def fiberPool():
	return fsc.asnc.FiberPool(1024 * 1024)

# Timer currently unsupported
"""
@fsc.asnc.asyncFunction
async def test_timer():
	t1 = time.time()
	await fsc.asnc.delay(0.1)
	t2 = time.time()
	
	assert (t2 - t1) == approx(0.1, abs=0.1)
	"""

@fsc.asnc.asyncFunction
async def test_nested():	
	async def nested():
		return None
	
	@fsc.asnc.asyncFunction
	async def nestedAF():
		return None
		
	await nested()
	await nestedAF.asnc()
	
	return None

# Fibers are currently not available
"""
@fsc.asnc.asyncFunction
async def test_fiberPool(fiberPool):
	await fiberPool.startFiber(test_nested)
"""