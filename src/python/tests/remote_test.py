import fusionsc as fsc
from fusionsc.devices import w7x, jtext

import pytest
import numpy as np

def test_serve_connect():
	"""openPort = fsc.remote.serve(fsc.backends.activeBackend())
	
	portNo = openPort.getPort()
	connected = fsc.remote.connect(f"http://localhost:{portNo}")
	
	fsc.asnc.wait(connected)
	print("Connect OK")
	del connected
	print("Disconnected")
	
	openPort.stopListening()
	print("Stopped")
	
	#Note: This currently tends to crash when closed
	del openPort
	print("Finished")
	#openPort.closeAll()
	#print("Closed")
	#openPort.drain()
	#print("Drained")"""