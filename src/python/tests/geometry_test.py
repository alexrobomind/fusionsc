import fusionsc as fsc
from fusionsc.devices import jtext

import pytest
import numpy as np

try:
	import meshio
	meshioAvailable = True
except:
	meshioAvailable = False

def test_simpleGeo():
	geo = fsc.geometry.cuboid([0, 0, 0], [1, 1, 1], {"A" : 1, "B" : "Hello"})
	
	geo = geo + geo.scale(0.5).translate([2, 0, 0]).rotate(np.radians(25), [0, 0, 1])
	geo = geo.merge()
	fsc.asnc.wait(geo)

@pytest.mark.skipif(not meshioAvailable, reason = "meshio not installed")
def test_jtextGeoOff():
	geo = jtext.pfcs(0.24)
	geo = geo.reduce()
	geo.exportTo(str(tmp_path / "test.off"))

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

def test_quad():
	rg, phig = np.meshgrid(np.linspace(5, 6, 100), np.linspace(0, 1.6 * np.pi, 100), indexing = 'ij')
	
	x = rg * np.cos(phig)
	y = rg * np.sin(phig)
	z = (rg - 5.5)**2
	
	geo = fsc.geometry.Geometry.quadMesh([x, y, z], wrapU = False, wrapV = True)
	geo.getMerged()

def test_intersect():
	p1 = [0, 0, 0]
	p2 = [2, 0, 0]
	
	jtext.pfcs(0.24).intersect(p1, p2, grid = jtext.defaultGeometryGrid())
