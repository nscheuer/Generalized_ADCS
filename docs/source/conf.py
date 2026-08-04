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
]

autodoc_mock_imports = ["saltro_py", "matplotlib", "mpl_toolkits", "pyvista", "vtk", "rich", "choldate", "trajectory_planner.build", "abc"]

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
html_static_path = ['_static']
html_css_files = ['custom.css']

# Canonical URL. Needed so that pages linked from print material (the SSC26
# poster QR codes point at /ssc26/) resolve to absolute URLs in link previews
# and search results rather than to relative paths.
html_baseurl = 'https://nscheuer.github.io/Generalized_ADCS/'

html_theme_options = {
    "sidebar_hide_name": False,
    "light_logo": "starlab_logo.svg",
    "dark_logo": "starlab_logo.svg",
    "top_of_page_button": "edit",
    "navigation_with_keys": True,
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


def autodoc_skip_reexports_on_package_pages(app, what, name, obj, skip, options):
    if what != "module":
        return skip

    cur_module = app.env.temp_data.get("autodoc:module")
    if not cur_module:
        return skip

    try:
        mod = __import__(cur_module, fromlist=["*"])
    except Exception:
        return skip

    if not hasattr(mod, "__path__"):
        return skip

    obj_module = getattr(obj, "__module__", None)
    if obj_module and obj_module != cur_module:
        return True

    return skip


def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_reexports_on_package_pages)
