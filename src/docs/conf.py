# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

if 'DOC_FSCPATH' not in os.environ:
	raise 'Use the CMake target "docs" to build the documentation'

project = 'FusionSC'
copyright = '2023 - 2024, Forschungszentrum Jülich GmbH, Jülich, Germany'
author = 'Alexander Knieps'
release = '2.3.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
	'myst_nb',
	'sphinx.ext.autodoc',
	'sphinx.ext.coverage',
	'sphinx.ext.autosummary',
	'sphinx_tabs.tabs',
	#'fusionsc.sphinx-ext.capnp',
	'breathe',
]

templates_path = ['templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autosummary_ignore_module_all = False
autosummary_imported_members = False


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']

# -- Notebook options --------------------------------------------------------

nb_execution_mode = "off"

sys.path.insert(0, os.path.abspath(os.environ['DOC_FSCPATH']))

# -- Breathe configuration (C++ docs via Doxygen XML) ------------------------

breathe_projects = {}

# Main C++ API documentation
_doxygen_xml = os.environ.get('FSC_DOXYGEN_XML_DIR')
if _doxygen_xml and os.path.isdir(_doxygen_xml):
	breathe_projects["fsc"] = _doxygen_xml
else:
	# When building docs without Doxygen, we skip C++ doc generation.
	# Breathe will produce warnings but the build will still succeed.
	breathe_projects["fsc"] = os.environ.get('FSC_DOXYGEN_XML_DIR', '_doxygen/xml')

# Cap'n'Proto generated API documentation
#_genapi_xml = os.environ.get('FSC_GENAPI_XML_DIR')
#if _genapi_xml and os.path.isdir(_genapi_xml):
#	breathe_projects["fsc-genapi"] = _genapi_xml

breathe_default_project = "fsc"
breathe_default_members = ('members', 'undoc-members')

# During transition, C++ docs may be incomplete. Don't let Breathe warnings
# stop the whole build.
breathe_show_include = False
