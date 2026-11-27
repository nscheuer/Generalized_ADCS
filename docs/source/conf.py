import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

project = 'Generalized ADCS'
copyright = '2026, Niclas Scheuer, Patrick McKeen'
author = 'Niclas Scheuer, Patrick McKeen'
release = '2026'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'myst_parser',
    'sphinx_sitemap',
    'sphinx_design',
    'sphinx_copybutton',
]

# Copy buttons on code blocks. Strip shell and Python REPL prompts so a copied
# block can be pasted straight into a terminal or interpreter.
copybutton_prompt_text = r">>> |\.\.\. |\$ |PS> "
copybutton_prompt_is_regexp = True

autodoc_mock_imports = ["saltro_py", "matplotlib", "mpl_toolkits", "pyvista", "vtk", "rich", "trajectory_planner.build"]

autosummary_generate = False
autosummary_imported_members = False

autodoc_typehints = 'description'
autodoc_member_order = 'groupwise'

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "imported-members": False,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'furo'
html_title = 'Generalized ADCS'
html_favicon = '_static/logo-light.svg'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_js_files = ['factory-tables.js']

# Canonical URL. Needed so that pages linked from print material (the SSC26
# poster QR codes point at /ssc26/) resolve to absolute URLs in link previews
# and search results rather than to relative paths.
html_baseurl = 'https://nscheuer.github.io/Generalized_ADCS/'
sitemap_url_scheme = "{link}"

html_theme_options = {
    "sidebar_hide_name": False,
    "light_logo": "logo-light.svg",
    "dark_logo": "logo-dark.svg",
    "top_of_page_button": "edit",
    "navigation_with_keys": True,
    # Brand colours and the palette used by custom.css. Furo applies these for
    # its light, dark and auto modes, so components follow the theme toggle in
    # the page header as well as the operating-system preference.
    "light_css_variables": {
        "color-brand-primary": "#0066cc",
        "color-brand-content": "#0066cc",
        "color-brand-visited": "#0052a3",
        "adcs-tip-border": "#79aee8",
        "adcs-tip-border-strong": "#3277c8",
        "adcs-tip-bg": "#edf5ff",
        "adcs-tip-title": "#174b86",
        "adcs-tip-title-bg": "#d9eaff",
        "adcs-mission-border": "#e8a04f",
        "adcs-mission-border-strong": "#dc791c",
        "adcs-mission-bg": "#fff3e4",
        "adcs-mission-title": "#82400e",
        "adcs-mission-title-bg": "#ffe3bf",
    },
    "dark_css_variables": {
        "color-brand-primary": "#4d9eff",
        "color-brand-content": "#80c0ff",
        "color-brand-visited": "#a9d3ff",
        "adcs-tip-border": "#37679c",
        "adcs-tip-border-strong": "#37679c",
        "adcs-tip-bg": "#142841",
        "adcs-tip-title": "#c6e0ff",
        "adcs-tip-title-bg": "#1d3b5e",
        "adcs-mission-border": "#a76525",
        "adcs-mission-border-strong": "#a76525",
        "adcs-mission-bg": "#352313",
        "adcs-mission-title": "#ffd29a",
        "adcs-mission-title-bg": "#563316",
    },
}

html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/scroll-end.html",
    ]
}

toc_object_entries = False
toc_object_entries_show_parents = 'hide'

latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '11pt',
    'preamble': r'''
\usepackage{amsmath,amssymb,bm}
''',
}

suppress_warnings = ['ref.python', 'toc.not_included']


def autodoc_skip_reexports(app, what, name, obj, skip, options):
    """Index imported objects only in the module that defines them."""
    if what == "class" and isinstance(obj, type):
        # Environment interfaces expose model classes as convenience class
        # attributes. Their defining modules have the canonical API entries.
        return True

    if what != "module":
        return skip

    cur_module = app.env.temp_data.get("autodoc:module")
    obj_module = getattr(obj, "__module__", None)
    if cur_module and obj_module and obj_module != cur_module:
        return True
    return skip


def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_reexports)
