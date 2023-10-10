from skbuild import setup
from setuptools import find_packages

setup(
	packages = ['fusionsc', 'fusionsc.devices', 'fusionsc.devices.w7x', 'fusionsc.devices.jtext'],
	package_dir = {"" : "src/python"},
	cmake_install_target = "fsc-install-skbuild",
)