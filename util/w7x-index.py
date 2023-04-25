import json
import pandas as pd
import numpy as np

import fusionsc as fsc
from fusionsc.devices import w7x

print("Loading...")

# Load whole component list
with open('components.json') as f:
	componentsRaw = json.load(f)
	
components = pd.DataFrame.from_dict(componentsRaw)
components['compID'] = np.arange(len(components))

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
geoList = baffles.geometry.initCombined(len(rows))

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

print(type(baffles.geometry))
baffles.save("baffles.fsc")

# Baffle geometry
heatShield = fsc.geometry.Geometry()
print("Processing heat shield")

query = (
	components.name.str.endswith('LOD-holes-removed_sag-0.25_2022-09-01') &
	components.name.str.contains('WSZ')
)

rows = components[query]

print("\tProcessing", len(rows), "entries")
geoList = heatShield.geometry.initCombined(len(rows))

for row, output in zip(rows.itertuples(), geoList):
	_, bafRow, bafID = row.location.split(', ')
	
	_, _, hm, _, _ = row.name.split(' # ')
	_, bafNo = bafID.split(' ')
	_, groupNo = bafRow.split(' ')
	
	assert hm[0:2] == 'HM'
	assert hm[4:6] == '.1'
	hm = int(hm[2:4])
	
	moduleNo = hm // 10
	
	output.initW7x().componentsDbMesh = row.compID
	tags = output.initTags(4)
	tags[0].name = 'moduleNo'
	tags[0].value.uInt64 = int(moduleNo)
	tags[1].name = 'halfModuleNo'
	tags[1].value.uInt64 = hm
	tags[2].name = 'wszGroupNo'
	tags[2].value.uInt64 = int(groupNo)
	tags[3].name = 'tileNo'
	tags[3].name.uInt64 = int(bafNo)
	
heatShield.save("heatShield.fsc")

divertor = fsc.geometry.Geometry()
print("Processing divertor")

query = (
	components.name.str.endswith('0.25_2022-09-01') &
	components.name.str.contains('Divertor')
)

rows = components[query]

print("\tProcessing", len(rows), "entries")
geoList = divertor.geometry.initCombined(len(rows))

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

divertor.save("divertor.fsc")

print(compTypes)