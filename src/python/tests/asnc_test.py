import fusionsc as fsc
import time

from pytest import approx, fixture

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
