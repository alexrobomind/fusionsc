# NOTE
#
# To use this script, please download a dump of the ComponentsDB metadata from
#    http://esb.ipp-hgw.mpg.de:8280/services/ComponentsDbRest/components
# and store it as "components.json" next to the runtime of this script.

import json
import pandas as pd
import numpy as np
from intspan import intspan

import fusionsc as fsc
from fusionsc.devices import w7x

print("Loading...")

# Load whole component list
with open('components.json') as f:
	componentsRaw = json.load(f)
	
components = pd.DataFrame.from_dict(componentsRaw)
components['compID'] = np.arange(len(components))

print("No of components: ", len(components))

# Baffle geometry
closure = fsc.geometry.Geometry()
print("Processing closure (not added to index)")

query = (
	components.name.str.contains('closure')
)

rows = components[query]

print("\tProcessing", len(rows), "entries")
geoList = closure.data.initCombined(len(rows))
entries = intspan()

for row, output in zip(rows.itertuples(), geoList):	
	entries.add(row.compID)

print("Entries: ", entries)

# New entries start at 599

# Baffle geometry
baffles = fsc.geometry.Geometry()
print("Processing baffles")

query = (
	components.name.str.endswith('LOD-holes-removed_sag-0.25_2022-09-01') &
	components.name.str.contains('Baffle')
)

rows = components[query]

print("\tProcessing", len(rows), "entries")
geoList = baffles.data.initCombined(len(rows))
entries = intspan()

for row, output in zip(rows.itertuples(), geoList):
	module, bafRow, bafID = row.location.split(', ')
	_, moduleNo, ul = module.split(' ')
	_, tileNo = bafID.split(' ')
		
	output.initW7x().componentsDbMesh = row.compID
	tags = output.initTags(4)
	tags[0].name = 'moduleNo'
	tags[0].value.uInt64 = int(moduleNo)
	tags[1].name = 'upper'
	tags[1].value.uInt64 = 1 if ul == 'upper' else 0
	tags[2].name = 'baffleRow'
	tags[2].value.text = bafRow
	tags[3].name = 'tileNo'
	tags[3].value.uInt64 = int(tileNo)
	
	entries.add(row.compID)

print("Entries: ", entries)

baffles.save("baffles.fsc")

# Baffle geometry
heatShield = fsc.geometry.Geometry()
print("Processing heat shield")
entries = intspan()

query = (
	components.name.str.endswith('LOD-holes-removed_sag-0.25_2022-09-01') &
	components.name.str.contains('WSZ')
)

rows = components[query]

print("\tProcessing", len(rows), "entries")
geoList = heatShield.data.initCombined(len(rows))

for row, output in zip(rows.itertuples(), geoList):
	#if(row.compID > 22107):
	#	continue
		
	try:
		_, bafRow, bafID = row.location.split(', ')
		
		_, _, hm, _, _ = row.name.split(' # ')
		_, bafNo = bafID.split(' ')
		_, groupNo = bafRow.split(' ')
		
		assert hm[0:2] == 'HM'
		assert hm[4:6] == '.1'
		hm = int(hm[2:4])
		
		moduleNo = hm // 10
		
		if groupNo == '':
			# There are some entries missing in the components DB
			# Fill them in by hand
			if row.compID in range(22108, 22138):
				groupNo = 11
			else:
				continue
		
		
		output.initW7x().componentsDbMesh = row.compID
		tags = output.initTags(4)
		tags[0].name = 'moduleNo'
		tags[0].value.uInt64 = int(moduleNo)
		tags[1].name = 'halfModuleNo'
		tags[1].value.uInt64 = hm
		tags[2].name = 'wszGroupNo'
		tags[2].value.uInt64 = int(groupNo)
		tags[3].name = 'tileNo'
		tags[3].value.uInt64 = int(bafNo)
		
		entries.add(row.compID)
	except Exception as e:
		raise Exception("When processing " + str(row.compID)) from e

print("Entries: ", entries)
	
heatShield.save("heatShield.fsc")

divertor = fsc.geometry.Geometry()
print("Processing divertor")
entries = intspan()

query = (
	components.name.str.endswith('0.25_2022-09-01') &
	components.name.str.contains('Divertor')
)

rows = components[query]

print("\tProcessing", len(rows), "entries")
geoList = divertor.data.initCombined(len(rows))

compTypes = {}

for row, output in zip(rows.itertuples(), geoList):
	mod, section, piece = row.location.split(', ')
	
	_, modNo, ul = mod.split(' ')
	compType, compId = piece.split(' ')
	
	#print(compType, ':', compId, '-', modNo, '-', ul, '-', section)
	
	compTypes[compType] = compType
	
	output.initW7x().componentsDbMesh = row.compID
	tags = output.initTags(5)
	tags[0].name = 'moduleNo'
	tags[0].value.uInt64 = int(modNo)
	tags[1].name = 'upper'
	tags[1].value.uInt64 = 1 if ul == 'upper' else 0
	tags[2].name = 'section'
	tags[2].value.text = section
	tags[3].name = 'componentType'
	tags[3].value.text = compType
	tags[4].name = 'componentNo'
	tags[4].value.uInt64 = int(compId)
	
	entries.add(row.compID)

print("Entries: ", entries)

divertor.save("divertor.fsc")

import contextlib
import time

@contextlib.contextmanager
def timer():
	c1 = time.perf_counter()
	yield None
	c2 = time.perf_counter()
	
	print("Time [s]:", c2 - c1)
	
with timer():
	geo = fsc.geometry.Geometry.load("divertor.fsc")
	print(geo.data.which())