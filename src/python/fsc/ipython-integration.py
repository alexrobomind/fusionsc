from .native.timer import delay

hasIPython = False

try:
	import IPython
	hasIPython = True
except:
	pass

if hasIPython:
	from IPython.kernel.zmq.eventloops import register_integration
	
	@register_integration('fsc')
	def fsc_integration(kernel):
		@asyncFunction
		async def loop():
			while True:
				await delay(kernel._poll_interval)
				kernel.do_one_iteration()
		
		# Run infinite loop
		loop()