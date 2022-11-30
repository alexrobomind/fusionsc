import argparse
from tqdm import tqdm

from fsc.asnc import eager

import fsc.native as native
import fsc.native.devices.w7x as w7xnative

def downloadArgs(parser):
	# Declare arguments	
	parser.add_argument("--coil", type=int, action='append', default = [])
	parser.add_argument("--config", type=int, action='append', default = [])
	parser.add_argument("--assembly", type=int, action='append', default = [])
	parser.add_argument("--mesh", type=int, action='append', default = [])
	
	parser.add_argument("--default", action = "store_true")
	
	parser.add_argument("--coilsdb", default = "http://esb.ipp-hgw.mpg.de:8280/services/CoilsDBRest")
	parser.add_argument("--componentsdb", default = "http://esb.ipp-hgw.mpg.de:8280/services/ComponentsDbRest")
	
	parser.add_argument("output", default = "w7x.fsc")
	
	return parser
	
@eager
async def download(args = None):
	if args is None:
		args = downloadArgs(argparse.ArgumentParser()).parse_args()
	
	configs = args.config
	coils = args.coil
	assemblies = args.assembly
	meshes = args.mesh
	
	# Add default components
	if args.default:
		coils.extend(range(160, 230)) # CAD main coils
		coils.extend(range(230, 240)) # CAD control coils
		coils.extend([350, 241, 351, 352, 535]) # CAD trim coils
		
		meshes.append(164) # Boundary
		meshes.extend(range(165, 170)) # OP1.2 test divertor unit
		meshes.extend(range(320, 335)) # OP1.2 baffles, baffle covers, heat shield
		meshes.extend(range(450, 455)) # Pump splits
		
		assemblies.append(8) # OP1.2 steel panels
		assemblies.append(9) # OP1.2 heat shield
	
	# Connect to webservices
	coilsDB = w7xnative.webserviceCoilsDB(args.coilsdb)
	componentsDB = w7xnative.webserviceComponentsDB(args.componentsdb)
	
	# Download configs and assemblies
	# We need to do that first as these give us additional coils and meshes
	async def getConfig(id):
		config = await coilsDB.getConfig(id)
		for coilId in config.coils:
			coils.append(coilId)
		
		return config
	
	async def getAssembly(id):
		assembly = await componentsDB.getAssembly(id)
		for meshId in assembly.components:
			meshes.append(meshId)
		
		return assembly
	
	async def getGeometry(reader):
		if reader.which() == 'combined':
			for piece in reader.combined:
				getGeometry(piece)
		
		else if reader.which() == 'componentsDBMeshes':
			meshes.extend(reader.componentsDBMeshes)
	
	configs = {
		id : await getConfig(id)
		for id in tqdm(set(configs), "Downloading configurations")
	}
	
	assemblies = {
		id : await getAssembly(id)
		for id in tqdm(set(assemblies), "Downloading assemblies")
	}
	
	# Now download coils and meshes
	coils = {
		id : await coilsDB.getCoil(id)
		for id in tqdm(set(coils), "Downloading coils")
	}
	
	meshes = {
		id : await componentsDB.getMesh(id)
		for id in tqdm(set(meshes), "Download components")
	}
	
	print('Writing', args.output)
	
	# Prepare output
	output = native.OfflineData.newMessage()
	
	output.initW7xConfigs(len(configs))
	for out, (id, config) in zip(output.w7xConfigs, configs.items()):
		out.id = id
		out.config = config
	del configs
	
	output.initW7xAssemblies(len(assemblies))
	for out, (id, assembly) in zip(output.w7xAssemblies, assemblies.items()):
		out.id = id
		out.assembly = assembly
	del assemblies
	
	output.initW7xCoils(len(coils))
	for out, (id, coil) in zip(output.w7xCoils, coils.items()):
		out.id = id
		out.filament = coil
	del coils
	
	output.initW7xComponents(len(meshes))
	for out, (id, mesh) in zip(output.w7xComponents, meshes.items()):
		out.id = id
		out.component = mesh
	del meshes
	
	# Write output
	await native.writeArchive(output, args.output)
	print("Done")
	
	return 0

if __name__ == "__main__":
	rootParser = argparse.ArgumentParser()
	
	subparsers = rootParser.add_subparsers(dest = 'cmd', required = True)
	downloadParser = downloadArgs(subparsers.add_parser('download'))
	
	args = rootParser.parse_args()
	if args.cmd == 'download':
		download(args)
	