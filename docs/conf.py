"""Configuration file for sphinx documentation.
To build locally start populating 'references' manually
via apidoc from the home dir:

    $ sphinx-apidoc -o docs/references deepblink\
                    -H References\
                    --tocfile index

Next, make sure all links are functional. Due to inconsistent behaviour,
it currently isn't included in the tox checks. Run:

    $ sphinx-build -b linkcheck docs dist/docs

Finally, check proper building using:

    $ sphinx-build docs dist/docs
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
version = release = "0.0.4"

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
