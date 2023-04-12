from . import asnc

import time

from pytest import approx, fixture

@fixture
def fiberPool():
	return asnc.FiberPool(1024 * 1024)

@asnc.asyncFunction
async def test_timer():
	t1 = time.time()
	await asnc.delay(0.1)
	t2 = time.time()
	
	assert (t2 - t1) == approx(0.1, abs=0.1)

@asnc.asyncFunction
async def test_fiberPool(fiberPool):
	await fiberPool.startFiber(test_timer)

@asnc.asyncFunction
async def test_await():
	await test_timer.asnc()