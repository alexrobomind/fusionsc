import numpy as np

import fsc
from fsc.devices import w7x

coils = fsc.data.readArchive("examples/cadCoils.fsc").wait()

tracer = fsc.tracer()

grid = fsc.native.ToroidalGrid.newMessage()
grid.nR = 128
grid.nZ = 128
grid.nPhi = 32
grid.rMin = 4
grid.rMax = 7
grid.zMin = -1.5
grid.zMax = 1.5
grid.nSym = 5

config = w7x.standard(coils = coils)

startPoints = np.asarray([
	[5.71, 0, 0],
	[5.8 , 0, 0]
]).T
# startPoints = np.ascontiguousarray(startPoints)

pcPoints = tracer.poincareInPhiPlanes(startPoints, [0.0], 100, config, grid, distanceLimit = 1e5)
print("------------------------ DONE --------------------------")
print(pcPoints)