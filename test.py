import fsc
from fsc.devices import w7x

fsc.resolve.importOfflineData("/mnt/c/Users/Knieps/Downloads/w7x.fsc")

grid = fsc.native.ToroidalGrid.newMessage()
grid.nR = 128
grid.nZ = 128
grid.nPhi = 1
grid.rMin = 4
grid.rMax = 6
grid.zMin = -1.5
grid.zMax = 1.5

config = w7x.standard()
tracer = fsc.tracer()
compField = tracer.computeField(config, grid)

print(compField)