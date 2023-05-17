from skbuild import setup
from setuptools import find_packages

setup(
	name = 'fusionsc',
	version = '0.1',
	packages = ['fusionsc', 'fusionsc.devices', 'fusionsc.devices.w7x', 'fusionsc.devices.jtext'],
	package_dir = {"" : "src/python"},
	cmake_install_target = "fsc-install-skbuild",
	entry_points = {
		'console_scripts': [
			#'w7x-download = fusionsc.devices.w7x.scripts:download'
		]
	},
	
	install_requires = [
		"typing-extensions >= 3.10",
		"numpy>=1.19.5 ; python_version < '3.10'",
		"numpy>=1.23.4 ; python_version >= '3.10'",
        "nest-asyncio",
		"netCDF4"
	]
)