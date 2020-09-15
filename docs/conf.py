"""Sphinx API documentation.

Sphinx is used to create an interactive, viewable API documentation.
There are three steps to building a fresh documentation:

1. Generate new API docs:
    sphinx-apidoc -o sphinx/ deepblink -e --tocfile index -d 1
2. Rearange minor things:
    - Replace index.rst content with ".. include:: ./deepblink.rst"
    - Delete everything under "Module contents" in deepblink.rst
    - Combine both remaining toctrees in deepblink.rst and sort
    alphabetically while removing the "Submodules..." headings
3. Build the project for local inspection:
    sphinx-build sphinx dist/docs

This generates the HTML files in a build/ directory.
Inspect the index.html to make sure things look good.
"""

import os

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.ifconfig",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
]

source_suffix = ".rst"
master_doc = "index"
project = "deepblink"
year = "2020"
author = "Bastian Eichenberger"
copyright = f"{year}, {author}"
version = release = "0.0.6"

pygments_style = "trac"
templates_path = ["."]
extlinks = {
    "issue": ("https://github.com/bbquercus/deepblink/issues/%s", "#"),
    "pr": ("https://github.com/bbquercus/deepblink/pull/%s", "PR #"),
}
# on_rtd is whether we are on readthedocs.org
on_rtd = os.environ.get("READTHEDOCS", None) == "True"

if not on_rtd:  # only set the theme if we're building docs locally
    html_theme = "sphinx_rtd_theme"

html_use_smartypants = True
html_last_updated_fmt = "%b %d, %Y"
html_split_index = False
html_sidebars = {
    "**": ["searchbox.html", "globaltoc.html", "sourcelink.html"],
}
html_short_title = f"{project}-{version}"

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False
