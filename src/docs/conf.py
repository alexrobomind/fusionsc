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

extensions = ['myst_nb', 'sphinx.ext.autodoc', 'sphinx.ext.coverage', 'sphinx.ext.autosummary', 'sphinx_tabs.tabs', 'fusionsc.sphinx-ext.capnp']

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