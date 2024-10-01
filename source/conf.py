# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from datetime import date

import dunamai

sys.path.insert(0, os.path.abspath("../pyCellPhenoX"))

import pyCellPhenoX  

project = "CellPhenoX"
copyright = "2024, Fan Zhang, Jade Young, Jun Inamo, Revanth Krishna, Zachary Caterer"
author = "Fan Zhang, Jade Young, Jun Inamo, Revanth Krishna, Zachary Caterer"
release = "0.2.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# -- Extensions ---------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",  # Automatically document Python modules
    "sphinx.ext.napoleon",  # Support for Google style docstrings
    "sphinx_copybutton",  # Adds a "copy" button to code blocks
    "m2r2",  # Support for Markdown files
    "nbsphinx",  # Support for Jupyter Notebooks
]

html_js_files = [
    "readthedocs.js",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Set the theme to the default Sphinx theme
# html_theme = "default"  # Change to 'default' for the basic theme
# html_theme = "alabaster"  # or use this if you prefer alabaster
# html_theme = "sphinx_rtd_theme"
html_static_path = []

# -- Options for HTML output -------------------------------------------------
# The theme to use for HTML and HTML Help pages.
html_theme = "furo"

# Define colors for light and dark modes
cellphenox_dark_blue = "#317EC2"
cellphenox_light_blue = "#B9DBF4"
cellphenox_pink = "#F3CDCC"  # Added missing '#' for hex code
cellphenox_red = "#C25757"

# Furo theme option colors specified here
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": cellphenox_red,
        "color-brand-content": cellphenox_red,
        "color-api-pre-name": cellphenox_red,
        "color-api-name": cellphenox_red,
    },
    "dark_css_variables": {
        "color-brand-primary": cellphenox_dark_blue,  # Use dark blue for primary
        "color-brand-content": cellphenox_dark_blue,  # Use dark blue for content
        "color-api-pre-name": cellphenox_dark_blue,  # Use dark blue for API names
        "color-api-name": cellphenox_dark_blue,  # Use dark blue for API pre-names
    },
}

# Path to the logo file
html_logo = "../logo/pycpx.svg"
