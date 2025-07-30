import fusionsc as fsc

import threading
import asyncio

def test_newThread():
	def innerFunc():
		# Create a new asyncio loop
		asyncio.set_event_loop(asyncio.new_event_loop())
		
		fsc.backends.connectLocal()
		assert fsc.backends.isLocalConnected()
		
		fsc.backends.disconnectLocal()
		assert not fsc.backends.isLocalConnected()
		
		fsc.asnc.stopEventLoop()
	
	t = threading.Thread(target = innerFunc)
	t.start()
	t.join()

def test_backend_switch():
	ba = fsc.backends.activeBackend()
	
	fsc.backends.reconfigureLocalBackend(fsc.service.LocalConfig.newMessage())
	
	with fsc.backends.useBackend(fsc.backends.localBackend()):
		print(fsc.backends.backendInfo())
	
	fsc.backends.alwaysUseBackend(ba)

def test_local_reconnect():
	fsc.backends.disconnectLocal()
	
	print(fsc.backends.backendInfo())
	