"""Sphinx configuration for the KMNet documentation."""

import os
import sys
from datetime import datetime

# Make the package importable for autodoc (src layout).
sys.path.insert(0, os.path.abspath("../src"))

import kmnet  # noqa: E402

# -- Project information -----------------------------------------------------
project = "KMNet"
author = "Rahul Goswami"
copyright = f"{datetime.now():%Y}, the KMNet authors"
version = kmnet.__version__
release = version

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_nb",                   # Markdown pages + .ipynb notebooks
    "sphinx_design",             # grids / cards / tabs
    "sphinx_copybutton",         # copy button on code blocks
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**/.ipynb_checkpoints"]

# -- MyST-NB -----------------------------------------------------------------
# The tutorial notebook is committed with its outputs; the build only renders them.
nb_execution_mode = "off"
nb_merge_streams = True

myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist", "attrs_inline", "attrs_block"]
myst_heading_anchors = 3

# -- Autodoc -----------------------------------------------------------------
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {"members": True, "show-inheritance": True}
napoleon_numpy_docstring = True
napoleon_google_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- HTML output (pydata-sphinx-theme) ---------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = [("reveal.js", {"defer": "defer"})]
html_title = "KMNet"
html_favicon = "_static/logo.svg"
html_show_sourcelink = False
pygments_style = "friendly"

html_theme_options = {
    "logo": {
        "image_light": "_static/logo.svg",
        "image_dark": "_static/logo.svg",
        "text": "KMNet",
    },
    "github_url": "https://github.com/yuvrajiro/KMNet",
    "icon_links": [
        {"name": "PyPI", "url": "https://pypi.org/project/kmnet/", "icon": "fa-brands fa-python"},
    ],
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "header_links_before_dropdown": 5,
    "show_prev_next": True,
    "show_toc_level": 2,
    "pygments_light_style": "friendly",
    "pygments_dark_style": "github-dark",
    "footer_start": ["copyright"],
    "footer_end": ["theme-version"],
}

html_context = {"default_mode": "auto"}
html_sidebars = {"**": []}
