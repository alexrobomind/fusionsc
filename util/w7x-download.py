import argparse
from tqdm import tqdm

import fusionsc as fsc

from fusionsc.devices import w7x
	
def download(args = None):
	if args is None:
		args = downloadArgs(argparse.ArgumentParser()).parse_args()
			
	# Connect to webservices
	w7x.connectCoilsDB(args.coilsdb)
	w7x.connectComponentsDB(args.componentsDB)
	
	coils = args.coil
	assemblies = args.assembly
	meshes = args.mesh
	
	# Extract parts from multi-part geometries
	def addParts(reader):
		if isinstance(reader, fsc.geometry.Geometry):
			addParts(reader.geometry)
			
		if reader.which() == 'combined':
			for el in reader.combined:
				addParts(el)
		
		if reader.which() == 'componentsDbMesh':
			meshes.append(reader.componentsDBMesh)
		
		if reader.which() == 'componentsDbAssembly':
			assemblies.append(reader.componentsDBAssembly)
	
	c = args.campaign
	
	# Add default components
	if args.default:
		coils.extend(range(160, 230)) # CAD main coils
		coils.extend(range(230, 240)) # CAD control coils
		coils.extend([350, 241, 351, 352, 535]) # CAD trim coils
		
		meshes.append(164) # Boundary
		
		addParts(w7x.divertor(c))
		addParts(w7x.baffles(c))
		addParts(w7x.heatShield(c))
		addParts(w7x.pumpSlits())
		addParts(w7x.vessel())
	
	# Perform downloads
	geometriesOut = []
	coilsOut = []
	
	for id in tqdm(assemblies, "Downloading assemblies"):
		key = w7x.assembly(id)
		geometriesOut.append({"key" : key.geometry, "val" : key.resolve().geometry})
	
	for id in tqdm(meshes, "Downloading meshes"):
		key = w7x.component(id)
		geometriesOut.append({"key" : key.geometry, "val" : key.resolve().geometry})
	
	for id in tqdm(coils, "Downloading coils"):
		key = w7x.coilsDBCoil(id)
		coilsOut.append({"key" : key.filament, "val" : key.resolve().filament})
	
	print("Preparing root")
	
	# Prepare output
	output = service.OfflineData.newMessage()
	output.geometries = geometriesOut
	output.coils = coilsOut
	
	print("Writing data")
	
	# Write output
	data.writeArchive(output, args.output)
	
	print("Done")
	
	return 0

if __name__ == "__main__":
	rootParser = argparse.ArgumentParser()
	
	# Declare arguments	
	rootParser.add_argument("--coil", type=int, action='append', default = [])
	rootParser.add_argument("--assembly", type=int, action='append', default = [])
	rootParser.add_argument("--mesh", type=int, action='append', default = [])
	
	rootParser.add_argument("--campaign", default="OP12")
	rootParser.add_argument("--default", action = "store_true")
	
	rootParser.add_argument("--coilsdb", default = "http://esb.ipp-hgw.mpg.de:8280/services/CoilsDBRest")
	rootParser.add_argument("--componentsdb", default = "http://esb.ipp-hgw.mpg.de:8280/services/ComponentsDbRest")
	
	rootParser.add_argument("output", default = "w7x.fsc")
	
	args = rootParser.parse_args()
	download(args)