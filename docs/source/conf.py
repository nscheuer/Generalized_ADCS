# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
# Add the project root (one level above /docs)
sys.path.insert(0, os.path.abspath('../..'))  # add project root

autodoc_mock_imports = ["Disturbance", "Sensor", "Actuator", "RW", "GPS"]

project = 'Generalized_ADCS'
copyright = '2025, Patrick McKeen, Niclas Scheuer'
author = 'Patrick McKeen, Niclas Scheuer'
release = '2025'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    #'sphinx_autodoc_typehints',
    'myst_parser',
]

autosummary_generate = True
autodoc_typehints = 'description'
napoleon_google_docstring = True
napoleon_numpy_docstring = True


templates_path = ['_templates']
exclude_patterns = []

autodoc_default_options = {
    "imported-members": False,
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '11pt',
    'preamble': r'''
\usepackage{amsmath,amssymb}
\usepackage{bm}
''',
}
html_static_path = ['_static']
