"""Helpers for exporting calculation results (without requiring reimport)"""

from . import service

from typing import Literal

def exportTrace(trace: dict, filename: str, format: Literal[None, "matlab", "netcdf4", "json"] = None, indent = None, allow_nan = True):
	"""
	Exports a fieldline trace result (from the fsc.flt.trace function) to a file
	for third party code use.
	
	Parameters:
	
	- trace: Tracing result
	- filename: Filename for the target file.
	- format: Format specifier. Must be "matlab", "netcdf4", "json", or None (in which case
	  the file format is inferred from the filename)
	- indent: JSON only - Whether to indent the file.
	- allow_nan: JSON only - whether to allow "inf" and "nan" (which are not json spec conformant) in the output.
	"""
	
	if format is None:
		format = _determineFormat(filename)
	
	assert format in ["matlab", "netcdf4", "json"]
	
	if format == "netcdf4":
		return _dumpTraceNc(trace, filename)
	
	if format == "matlab":
		return _dumpTraceMatlab(trace, filename)
	
	if format == "json":
		return _dumpTraceJson(trace, filename, indent = indent, allow_nan = allow_nan)
	
	raise ValueError(f"Unknown format '{format}'")

def _determineFormat(filename):
	if filename.endswith(".nc"):
		return "netcdf4"
	if filename.endswith(".mat"):
		return "matlab"
	if filename.endswith(".json"):
		return "json"
	
	raise ValueError(f"Could not determine format from filename {filename}. Known endings are .nc (netcdf4), .mat (matlab), and .json (json).")

def _dumpTraceMatlab(trace, filename):
	import scipy.io
	import numpy as np
	
	def tagToPy(tag):		
		if tag.which_() == "uInt64":
			return tag.uInt64
		
		if tag.which_() == "text":
			return str(tag.text)
		
		return "<NotSet>"
		
	tagToPy = np.vectorize(tagToPy, otypes=[object])
	
	matlabDict = {
		"endPoints" : trace["endPoints"],
		"poincareHits" : trace["poincareHits"],
		"stopReasons" : np.vectorize(str, otypes=[object])(trace["stopReasons"]),
		"fieldLines" : trace["fieldLines"],
		"fieldStrengths" : trace["fieldStrengths"],
		"endTags" : {
			name : tagToPy(values)
			for name, values in trace["endTags"].items()
		},
		"responseSize" : trace["responseSize"]
	}
	
	scipy.io.savemat(filename, matlabDict)

def _dumpTraceJson(trace, filename, **jsonOptions):
	import json
	jsonDict = {}
	
	pointsShape = trace["endPoints"].shape[1:]
	
	# End points
	endPoints = trace["endPoints"]
	jsonDict["endPoints"] = {
		"shape" : list(pointsShape),
		"x" : list(endPoints[0].flatten()),
		"y" : list(endPoints[1].flatten()),
		"z" : list(endPoints[2].flatten()),
		"len" : list(endPoints[3].flatten())
	}
	
	# Poincare hits
	pcHits = trace["poincareHits"]
	jsonDict["pcHits"] = {
		"shape" : list(pcHits.shape[1:]),
		"x" : list(pcHits[0].flatten()),
		"y" : list(pcHits[1].flatten()),
		"z" : list(pcHits[2].flatten()),
		"lcBwd" : list(pcHits[3].flatten()),
		"lcFwd" : list(pcHits[4].flatten())
	}
	
	# Stop reasons
	jsonDict["stopReasons"] = [
		str(reason)
		for reason in trace["stopReasons"].flatten()
	]
	
	# Field lines
	fieldLines = trace["fieldLines"]
	jsonDict["fieldLines"] = {
		"shape" : list(fieldLines.shape[1:]),
		"x" : list(fieldLines[0].flatten()),
		"y" : list(fieldLines[1].flatten()),
		"z" : list(fieldLines[2].flatten()),
		"field" : list(trace["fieldStrengths"].flatten())
	}
	
	# End tags
	def processTag(x):
		if x.which_() == "notSet":
			return None
		if x.which_() == "text":
			return str(x.text)
		if x.which_() == "uInt64":
			return x.uInt64
		return "<unknown>"
	
	jsonDict["endTags"] = {
		name : [processTag(tag) for tag in values.flatten()]
		for name, values in trace["endTags"].items()
	}
	
	with open(filename, "w") as f:
		json.dump(jsonDict, f, **jsonOptions)

def _dumpTraceNc(trace, filename):
	import netCDF4 as nc
	import numpy as np
	
	root = nc.Dataset(filename, "w", format="NETCDF4")
	
	# Points dimension
	pointsShape = trace["endPoints"].shape[1:]
	pointDims = [
		root.createDimension(f"points{i}", dim)
		for i, dim in enumerate(pointsShape)
	]
	
	# Neccessary types		
	stopReasonsEnum = root.createEnumType(np.uint16, "StopReason", {
		str(value) : value.raw
		for value in service.FLTStopReason.values
	})
	tagWhichEnum = root.createEnumType(np.uint8, "WhichKind", {
		"unknown" : 0,
		"notSet" : 1,
		"text" : 2,
		"uint64" : 3
	})
	
	# End points
	endPoints = trace["endPoints"]
	dimEndpoints = root.createDimension("xyzLen", 4)
	varEndPoints = root.createVariable("endPoints", np.float64, [dimEndpoints] + pointDims)
	varEndPoints[:] = endPoints
	
	# Poincare hits
	pcHits = trace["poincareHits"]
	dimPoincare = root.createDimension("xyzLcFwdLcBwd", 5)
	dimPhiPlanes = root.createDimension("phiPlanes", pcHits.shape[1])
	dimTurns = root.createDimension("turns", pcHits.shape[-1])
	varPcHits = root.createVariable("poincareHits", np.float64, [dimPoincare, dimPhiPlanes] + pointDims + [dimTurns])
	varPcHits[:] = pcHits
	
	# Stop reasons
	
	def getRaw(x):
		raw = x.raw
		if raw >= len(service.FLTStopReason.values):
			return 0 # 0 means Unknown
			
		return raw
	
	stopReasons = trace["stopReasons"]
	varStopReasons = root.createVariable("stopReasons", stopReasonsEnum, pointDims)
	varStopReasons[:] = np.vectorize(getRaw)(stopReasons)
	
	# Field lines
	fieldLines = trace["fieldLines"]
	dimFieldlines = root.createDimension("nPoints", fieldLines.shape[-1])
	dimXyz = root.createDimension("xyz", 3)
	
	varFieldLines = root.createVariable("fieldLines", np.float64, [dimXyz] + pointDims + [dimFieldlines])
	varFieldLines[:] = fieldLines
	
	# Field strengths
	fieldStrengths = trace["fieldStrengths"]
	varFieldStrengths = root.createVariable("fieldStrengths", np.float64, pointDims + [dimFieldlines])
	varFieldStrengths[:] = fieldStrengths
	
	# End tags
	groupEndTags = root.createGroup("endTags")
	
	@np.vectorize
	def getWhich(x):
		if x.which_() == "notSet":
			return 1
		if x.which_() == "text":
			return 2
		if x.which_() == "uInt64":
			return 3
		
		return 0
	
	@np.vectorize
	def getText(x):
		if x.which_() == "text":
			return str(x.text)
		
		return ""
	
	@np.vectorize
	def getUint64(x):
		if x.which_() == "uInt64":
			return x.uInt64
		
		return 0
		
	for name, values in trace["endTags"].items():
		subGroup = groupEndTags.createGroup(name)
		
		varWhich = subGroup.createVariable("type", tagWhichEnum, pointDims)
		varWhich[:] = getWhich(values)
		
		varText = subGroup.createVariable("text", str, pointDims)
		varText[:] = getText(values)
		
		varUint64 = subGroup.createVariable("uint64", np.uint64, pointDims)
		varUint64[:] = getUint64(values)
	
	# Response size
	varResponseSize = root.createVariable("responseSize", np.uint64, ())
	varResponseSize[:] = trace["responseSize"]
	
	root.close()