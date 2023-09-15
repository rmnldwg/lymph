# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/main/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

from pkg_resources import DistributionNotFound, get_distribution

sys.path.insert(0, os.path.abspath('../..'))

try:
    __version__ = get_distribution("lymph").version
except DistributionNotFound:
    __version__ = "unknown version"


# -- Project information -----------------------------------------------------

project = 'lymph'
copyright = '2022, Roman Ludwig'
author = 'Roman Ludwig'
gh_username = 'rmnldwg'

version = __version__
# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'myst_nb'
]

# MyST settings
myst_enable_extensions = ["colon_fence", "dollarmath"]
nb_execution_mode = "auto"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_data']

# document classes and their constructors
autoclass_content = 'class'

# sort members by source
autodoc_member_order = 'bysource'

# show type hints
autodoc_typehints = 'signature'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = 'sphinx_book_theme'
html_theme_options = {
    "repository_url": f"https://github.com/{gh_username}/{project}",
    "repository_branch": "main",
    "use_repository_button": True,
}

# import sphinx_modern_theme
# html_theme = "sphinx_modern_theme"
# html_theme_path = [sphinx_modern_theme.get_html_theme_path()]

# html_theme = "bootstrap-astropy"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['./_static']
