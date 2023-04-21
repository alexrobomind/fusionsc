import argparse
from tqdm import tqdm

from fusionsc.asnc import asyncFunction

import fusionsc

from fusionsc import native, data, devices

from . import divertor, baffles, heatShield, pumpSlits, vessel, singleComponent, singleAssembly, connectCoilsDB, connectComponentsDB

def downloadArgs(parser):
	# Declare arguments	
	parser.add_argument("--coil", type=int, action='append', default = [])
	parser.add_argument("--assembly", type=int, action='append', default = [])
	parser.add_argument("--mesh", type=int, action='append', default = [])
	
	parser.add_argument("--campaign", default="OP12")
	parser.add_argument("--default", action = "store_true")
	
	parser.add_argument("--coilsdb", default = "http://esb.ipp-hgw.mpg.de:8280/services/CoilsDBRest")
	parser.add_argument("--componentsdb", default = "http://esb.ipp-hgw.mpg.de:8280/services/ComponentsDbRest")
	
	parser.add_argument("output", default = "w7x.fsc")
	
	return parser
	
def download(args = None):
	if args is None:
		args = downloadArgs(argparse.ArgumentParser()).parse_args()
			
	# Connect to webservices
	connectCoilsDB(args.coilsdb)
	connectComponentsDB(args.componentsDB)
	
	coils = args.coil
	assemblies = args.assembly
	meshes = args.mesh
	
	# Extract parts from multi-part geometries
	def addParts(reader):
		if isinstance(reader, fsc.Geometry):
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
		
		addParts(divertor(c))
		addParts(baffles(c))
		addParts(heatShield(c))
		addParts(pumpSlits())
		addParts(vessel())
	
	# Perform downloads
	geometriesOut = []
	coilsOut = []
	
	for id in tqdm(assemblies, "Downloading assemblies"):
		key = singleAssembly(id)
		geometriesOut.append({"key" : key.geometry, "val" : key.resolve().geometry})
	
	for id in tqdm(meshes, "Downloading meshes"):
		key = singleMesh(id)
		geometriesOut.append({"key" : key.geometry, "val" : key.resolve().geometry})
	
	for id in tqdm(coils, "Downloading coils"):
		key = fusionsc.service.Filament.newMessage()
		key.initW7x().coilsDb = id
		
		coilsOut.append({"key" : key, "val" : fusionsc.resolve.resolveFilament(key)})
	
	# Prepare output
	output = native.OfflineData.newMessage()
	output.geometries = geometriesOut
	output.coils = coilsOut
	
	# Write output
	data.writeArchive(output, args.output)
	print("Done")
	
	return 0

if __name__ == "__main__":
	rootParser = argparse.ArgumentParser()
	
	subparsers = rootParser.add_subparsers(dest = 'cmd', required = True)
	downloadParser = downloadArgs(subparsers.add_parser('download'))
	
	args = rootParser.parse_args()
	if args.cmd == 'download':
		download(args)
	
