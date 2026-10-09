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

# -- HTML output (Furo, monochrome editorial) ---------------------------------
html_theme = "furo"
html_title = "KMNet"
html_static_path = ["_static"]
html_css_files = [
    "https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1"
    "&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap",
    "custom.css",
]
html_favicon = "_static/logo.svg"
html_show_sourcelink = False
html_show_sphinx = False
# Served at the site root: the CNAME file for the custom domain.
html_extra_path = ["_extra"]
html_baseurl = "https://kmnet.statml.in/"
pygments_style = "default"
pygments_dark_style = "github-dark"

_fonts = {
    "font-stack": "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif",
    "font-stack--monospace": "'JetBrains Mono', ui-monospace, Menlo, Consolas, monospace",
    "font-stack--headings": "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif",
}

html_theme_options = {
    "light_css_variables": {"color-brand-primary": "#111111", "color-brand-content": "#111111", **_fonts},
    "dark_css_variables": {"color-brand-primary": "#f2f2f2", "color-brand-content": "#f2f2f2"},
    "source_repository": "https://github.com/yuvrajiro/KMNet/",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/yuvrajiro/KMNet",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">'
                '<path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17'
                '.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52'
                '-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64'
                '-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0'
                ' 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87'
                ' 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016'
                ' 8c0-4.42-3.58-8-8-8z"></path></svg>'
            ),
            "class": "",
        },
    ],
}
