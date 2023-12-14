from . import service

def _determineFormat(filename):
	if filename.endswith(".nc"):
		return "netcdf4"
	if filename.endswith(".mat"):
		return "matlab"
	
	raise ValueError(f"Could not determine format from filename {filename}")

def exportTrace(trace, filename, format = None):
	if format is None:
		format = _determineFormat(filename)
	
	assert format in ["matlab", "netcdf4"]
	
	if format == "netcdf4":
		return _dumpTraceNc(trace, filename)
	
	if format == "matlab":
		return _dumpTraceMatlab(trace, filename)

def _dumpTraceMatlab(trace, filename):
	import scipy.io
	import numpy as np
		
	def tagToPy(tag):		
		if tag.which_() == "uInt64":
			return tag.uInt64
		
		if tag.which()_ == "text":
			return str(tag.text)
		
		return None
	
	matlabDict = {
		"endPoints" : trace["endPoints"],
		"poincareHits" : trace["poincareHits"],
		"stopReasons" : np.vectorize(str)(trace["stopReasons"]),
		"fieldLines" : trace["fieldLines"],
		"fieldStrengths" : trace["fieldStrengths"],
		"endTags" : {
			name : np.vectorize(tagToPy)(values)
			for name, values in trace["endTags"].items()
		},
		"responseSize" : response.totalBytes_()
	}
	
	return scipy.io.savemat(filename, matlabDict)

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
		"text": : 2,
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
		if x >= len(enumDict):
			return 0 # 0 means Unknown
			
		return x
	
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
		if x.which_() == "uint64":
			return 3
		
		return 0
	
	@np.vectorize
	def getText(x):
		if x.which_() == "text":
			return str(x.text)
		
		return ""
	
	@np.vectorize
	def getUint64(x):
		if x.which_() == "uint64":
			return x.uint64
		
		return 0
		
	for name, values in trace["endTags"].items():
		subGroup = groupEndTags.createGroup(name)
		
		varWhich = subGroup.createVariable("type", tagWhichEnum, pointDims)
		varWhich[:] = getWhich(values)
		
		varText = subGroup.createVariable("text", str, pointDims)
		varText[:] = getText(values)
		
		varUint64 = subgroup.createVariable("uint64", np.uint64, pointDims)
		varUint64[:] = getUint64(values)
	
	# Response size
	varResponseSize = root.createVariable("responseSize", np.uint64, ())
	varResponseSize[:] = trace["responseSize"]