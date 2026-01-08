# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
# Add the project root (two levels up from /docs/source)
sys.path.insert(0, os.path.abspath('../..'))

# Mock heavy or unavailable imports to prevent build errors
autodoc_mock_imports = ["matplotlib", "mpl_toolkits"]

# -----------------------------------------------------------------------------
# Project information
# -----------------------------------------------------------------------------
project = 'Generalized_ADCS'
copyright = '2025, Patrick McKeen, Niclas Scheuer'
author = 'Patrick McKeen, Niclas Scheuer'
release = '2025'

# -----------------------------------------------------------------------------
# General configuration
# -----------------------------------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',       # Core autodoc support
    'sphinx.ext.autosummary',   # Generates summary tables
    'sphinx.ext.napoleon',      # Parse NumPy/Google style docstrings
    'sphinx.ext.viewcode',      # Add [source] links
    'sphinx.ext.mathjax',       # Math rendering with LaTeX syntax
    'myst_parser',              # Markdown doc support
]

# Automatically generate summary stubs
autosummary_generate = True

# Control type hint rendering style
autodoc_typehints = 'description'

# Use both Google and NumPy docstring conventions
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Ensure autodoc doesn’t recursively expand imports
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "special-members": False,
    "inherited-members": False,
    "imported-members": False,   # prevents duplicate submodules
    "show-inheritance": True,
}

# Tame autodoc / autosummary TOC explosion
autodoc_member_order = 'groupwise'  # keeps members grouped logically
toc_object_entries_show_parents = 'hide'  # avoid deep nesting
autosummary_imported_members = False

# Optional: ignore overzealous cross-ref warnings from re-exports
nitpick_ignore_regex = [
    (r'py:class', r'.*Actuator'),
    (r'py:class', r'.*Sensor'),
    (r'py:class', r'.*Disturbance'),
    (r'py:class', r'.*Orbital_State'),
]

# You can safely leave exclude_patterns empty unless you have build directories
exclude_patterns = []

templates_path = ['_templates']

# -----------------------------------------------------------------------------
# HTML output
# -----------------------------------------------------------------------------
html_theme = 'furo'
html_theme_options = {
    "navigation_depth": 1,        # only top-level and 1 sublevel visible
    "collapse_navigation": True,  # collapse subtrees by default
    "sidebar_hide_name": False,
    "light_logo": "logo-light.png",   # optional: if you add a logo
    "dark_logo": "logo-dark.png",     # optional: if you add a logo
    "top_of_page_button": "edit",     # small aesthetic tweak
}

# Simplify navigation tree
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/scroll-end.html",
    ]
}

# Static files (optional, used for logos/css)
html_static_path = ['_static']

# -----------------------------------------------------------------------------
# Math & LaTeX output configuration
# -----------------------------------------------------------------------------
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '11pt',
    'preamble': r'''
\usepackage{amsmath,amssymb,bm}
''',
}

# -----------------------------------------------------------------------------
# Behavior improvements
# -----------------------------------------------------------------------------
# Suppress duplicate Python reference warnings (common with re-exports)
suppress_warnings = [
    'ref.python',
]

# Optional: consistent autosummary build style
autosummary_context = {
    'generate': True,
}

