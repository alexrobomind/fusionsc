import fsc
from fsc.devices import w7x

tracer = fsc.tracer()
fsc.importOfflineData("/mnt/c/Users/alexr/Downloads/w7x.fsc")

w7x.preheat(tracer)

fsc.native.delay(60).wait()