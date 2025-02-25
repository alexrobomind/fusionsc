import fusionsc as fsc
from fusionsc.devices import w7x

import pytest, asyncio

from threading import Thread

def test_xthread():
	h = fsc.xthread.export({"A" : "B"})
	
	def check():
		fsc.asnc.startEventLoop()
		
		assert h.get()["A"] == "B"
		print(h.get())
		
		fsc.asnc.stopEventLoop()
	
	t = Thread(target = check)
	t.start()
	
	while t.is_alive():
		asyncio.run(asyncio.sleep(0.001))
		t.join(0.001)
	
	# If this is not done, there are funky error message
	# at the final cleanup later on
	import gc
	gc.collect()
