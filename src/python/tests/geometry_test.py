import fusionsc as fsc
from fusionsc.devices import jtext

import pytest
import numpy as np

def test_simpleGeo():
	geo = fsc.geometry.cuboid([0, 0, 0], [1, 1, 1], {"A" : 1, "B" : "Hello"})
	
	geo = geo + geo.scale(0.5).translate([2, 0, 0]).rotate(np.radians(25), [0, 0, 1])
	geo = geo.merge()
	fsc.asnc.wait(geo)

def test_jtextGeo(tmp_path):
	geo = jtext.pfcs(0.24)
	geo = geo.reduce()
	geo.exportTo(str(tmp_path / "test.ply"))

def test_poly():
	points = [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]]
	
	polys1 = [[0, 1, 2], [0, 2, 3]]
	polys2 = [[0, 1, 2, 3]]
	
	fsc.geometry.Geometry.polyMesh(points, polys1).asPyvista()
	fsc.geometry.Geometry.polyMesh(points, polys2).asPyvista()

def test_wrap():
	r = [1, 1.1, 1.1, 1]
	z = [-0.5, -0.5, 0.5, 0.5]
	
	geo = fsc.geometry.Geometry.from2D(r, z, phi1 = np.radians(-30), phi2 = np.radians(30))
	geo.plotCut(np.radians(0))
	
	print(geo.reduce())