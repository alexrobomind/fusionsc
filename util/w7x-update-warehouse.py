import argparse
from tqdm import tqdm, trange

import fusionsc as fsc

from fusionsc.devices import w7x
	
def download(args = None):
	if args is None:
		args = downloadArgs(argparse.ArgumentParser()).parse_args()
			
	# Connect to webservices
	w7x.connectCoilsDB(args.coilsdb)
	w7x.connectComponentsDB(args.componentsdb)
	
	coils = args.coil
	assemblies = args.assembly
	meshes = args.mesh
	
	# The OP2 geometries need to be run through a first-step
	# resolver to get the initial part lists
	baseResolver = fsc.native.devices.w7x.geometryResolver()
	
	# Extract parts from multi-part geometries
	@fsc.asnc.asyncFunction
	async def addParts(reader):
		if isinstance(reader, fsc.geometry.Geometry):
			geo = reader.data
			resolved = await baseResolver.resolveGeometry(geo)
			await addParts.asnc(resolved)
			return
			
		if reader.which_() == 'combined':
			for el in reader.combined:
				await addParts.asnc(el)
			return
		
		if reader.which_() == 'nested':
			await addParts.asnc(reader.nested)
			return
		
		if reader.which_() == 'ref':
			await addParts.asnc(fsc.data.download(reader.ref))
			return
				
		if reader.which_() == 'w7x':
			w7x = reader.w7x
		
			if w7x.which_() == 'componentsDbMesh':
				meshes.append(w7x.componentsDbMesh)
			
			if w7x.which_() == 'componentsDbAssembly':
				assemblies.append(w7x.componentsDbAssembly)
			
			return
		
		print(reader)
	
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
		addParts(w7x.steelPanels())
		addParts(w7x.toroidalClosure())
	
	if args.all:
		coils.extend(range(0, 2342))
		meshes.extend(range(0, 22350))
		assemblies.extend(range(0, 22))
	
	if args.coilRange:
		coils.extend(range(*args.coilRange))
	if args.meshRange:
		meshes.extend(range(*args.meshRange))
	if args.assemblyRange:
		assemblies.extend(range(*args.assemblyRange))
		
	# Open database
	wh = fsc.warehouse.open(args.output)
	
	# Perform downloads
	def update(key):
		keyData = key.data
		
		for i in range(1000):
			try:
				valData = key.resolve().data
				break
			except KeyboardInterrupt:
				raise
			except Exception as e:
				tqdm.write(str(e))
				tqdm.write('Failed to resolve, trying again')
				
				if i == 3:
					raise

		fsc.resolve.updateWarehouse(wh, {keyData : valData})
		fsc.asnc.cycle()

	for id in tqdm(assemblies, "Downloading assemblies"):
		tqdm.write(str(id))
		key = w7x.assembly(id)
		update(key)
	
	for id in tqdm(meshes, "Downloading meshes"):
		tqdm.write(str(id))
		key = w7x.component(id)
		update(key)
	
	for id in tqdm(coils, "Downloading coils"):
		tqdm.write(str(id))
		key = w7x.coilsDBCoil(id)
		update(key)
	
	rootData = wh.get("resolveIndex").download()
	#print(rootData)
	
	return 0

if __name__ == "__main__":
	rootParser = argparse.ArgumentParser()
	
	# Declare arguments	
	rootParser.add_argument("--coil", type=int, action='append', default = [])
	rootParser.add_argument("--assembly", type=int, action='append', default = [])
	rootParser.add_argument("--mesh", type=int, action='append', default = [])
	
	rootParser.add_argument("--all", action = "store_true")
	
	rootParser.add_argument("--coilRange", type=int, nargs=2, default = [])
	rootParser.add_argument("--meshRange", type=int, nargs=2, default = [])
	rootParser.add_argument("--assemblyRange", type=int, nargs=2, default = [])
	
	rootParser.add_argument("--campaign", default="OP12")
	rootParser.add_argument("--default", action = "store_true")
	
	rootParser.add_argument("--coilsdb", default = "http://esb.ipp-hgw.mpg.de:8280/services/CoilsDBRest")
	rootParser.add_argument("--componentsdb", default = "http://esb.ipp-hgw.mpg.de:8280/services/ComponentsDbRest")
	
	rootParser.add_argument("output", default = "w7x.fsc")
	
	args = rootParser.parse_args()
	download(args)

