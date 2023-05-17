"""Integration with IPython and Jupyter"""

from .native.timer import delay
from .asnc import FiberPool, asyncFunction

from .asnc import asyncFunction

hasIPython = False

import asyncio
import nest_asyncio

try:
	import IPython
	hasIPython = True
except:
	pass

if hasIPython:
	from ipykernel.eventloops import register_integration
	from IPython.terminal.pt_inputhooks import register
	
	# The IPython kernel integrates by having us call its event loop
	# repeatedly. Such a call has to come from inside our own event loop,
	# where the usual wait scope is not available. Instead, we need a
	# wait scope that - instead of blocking the thread - saves the stack
	# and continues with other events on the loop.
	#
	# Fortunately, this type of quasi-synchronous programming is available
	# with fibers. Therefore, we run the kernel inside a fiber scope.
	
	@register_integration('fsc')
	def kernel_integration(kernel):
		"""IPython kernel integration (use with `%gui fsc`)"""
		
		# Modify the event loop to allow nesting
		nest_asyncio.apply()
		
		def runKernel():
			loop = asyncio.get_event_loop()
			try:
				loop.run_until_complete(kernel.do_one_iteration())
			except Exception:
				kernel.log.exception("Error in message handler")
		
		@asyncFunction
		async def loop():
			# Reserve 8MB of stack size
			stackSize = 8 * 1024 * 1024
			fiberPool = FiberPool(stackSize)
			
			while True:
				await fiberPool.startFiber(runKernel)
				await delay(kernel._poll_interval)
		
		# Run infinite loop
		loop()
	
	# The IPython terminal integration works different. Instead of asking
	# us to call the event loop, our event loop is the one that gets called.
	# In this case, integration is easier since we are always running from the
	# root scope.
	
	def terminal_inputhook(context):
		"""IPython terminal integration (use with `%gui fsc`)"""
		
		while not context.input_is_ready():
			# Cycle the event loop for 0.05s
			delay(0.05).wait()
	
	register('fsc', terminal_inputhook)