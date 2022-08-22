# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'FSC'
copyright = '2022, Forschungszentrum Juelich GmbH, Germany'
author = 'Alexander Knieps'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['myst_nb', 'sphinx.ext.autodoc', 'sphinx.ext.coverage', 'sphinx.ext.autosummary']

templates_path = ['templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']

# -- Notebook options --------------------------------------------------------

nb_execution_mode = "off"

import os
import sys
sys.path.insert(0, os.path.abspath('../src/python'))