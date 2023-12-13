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
		return _dumpTraceMat(trace, filename)

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
		"endTags" : np.vectorize(tagToPy)(trace["endTags"]),
		"responseSize" : response.totalBytes_()
	}
	
	return scipy.io.savemat(filename, matlabDict)

def _dumpTraceNc(trace, filename):
	assert False, "Unimplemented"