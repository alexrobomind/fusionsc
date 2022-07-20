import fsc
from fsc.devices import w7x

tracer = fsc.tracer()
fsc.importOfflineData("/mnt/c/Users/Knieps/Downloads/w7x.fsc")

w7x.preheat(tracer)