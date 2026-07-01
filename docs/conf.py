# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from datetime import datetime

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

pyproj_toml = tomllib.load(open("../pyproject.toml", "rb"))

year = datetime.now().year

project = "stablebear"
copyright = f"2024-{year}, Björn H. Wehlin"
author = "Björn H. Wehlin"
release = pyproj_toml["project"]["version"]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinxcontrib.youtube",
    "myst_parser",
    "nbsphinx",
    "sphinx_design",
    "sphinxcontrib.mermaid",
    "sphinxcontrib.bibtex",
]

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "alpha"
#'breathe',
#'exhale']

gen_cpp_docs = False

if gen_cpp_docs:
    extensions.append("breathe")
    extensions.append("exhale")

    breathe_projects = {"stablebear_internals": "./_build_doxygen/xml"}

    breathe_default_project = "stablebear_internals"

    import textwrap

    exhale_args = {
        # These arguments are required
        "containmentFolder": "./cpp_api",
        "rootFileName": "library_root.rst",
        "doxygenStripFromPath": "..",
        # Heavily encouraged optional argument (see docs)
        "rootFileTitle": "stablebear C++ API reference",
        # Suggested optional arguments
        "createTreeView": True,
        # TIP: if using the sphinx-bootstrap-theme, you need
        # "treeViewIsBootstrap": True,
        "exhaleExecutesDoxygen": True,
        "exhaleDoxygenStdin": "INPUT = ../include",
        "exhaleDoxygenStdin": textwrap.dedent("""
            INPUT      = ../include
        """),
    }


exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**/.virtual_documents"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
# html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    "header_links_before_dropdown": 4,
}
html_static_path = ["_static"]
html_css_files = ["lightbox.css"]
html_js_files = ["lightbox.js"]

# --- Create a temporary bundle of 'stablebear' with a stub C++ backend from the Python source in ../stablebear. This is only for documentation purposes (so that we don't have to keep reinstalling, including recompiling, stablebear everytime we want to update the docs). The setup has been tested on Linux and should probably work on OSX. It is unclear if it'll work on Windows.

import os
import platform
import shutil
import sys
from glob import glob

is_windows = platform.system() == "Windows"

temp_mod_dir = os.path.abspath("modules")

os.makedirs(temp_mod_dir, exist_ok=True)

print(f"Will remove {temp_mod_dir}")

shutil.rmtree(temp_mod_dir, ignore_errors=True)
os.makedirs(os.path.join(temp_mod_dir, "stablebear"))

stablebear_temp_dir = os.path.join(os.path.abspath("modules"), "stablebear")

files = glob(os.path.abspath("../stablebear/*"))
for file in files:
    basename = os.path.basename(file)
    if "__pycache__" in file:
        continue
    # Skip the real _sb_cpp.py -- we replace it with the stub below
    if basename == "_sb_cpp.py":
        continue

    target = os.path.join(stablebear_temp_dir, basename)

    if is_windows:
        if os.path.isdir(file):
            shutil.copytree(file, target)
        else:
            shutil.copy2(file, target)
    else:
        os.symlink(file, target)

# Symlink the stub _sb_cpp package (docs/_sb_cpp/) so that autodoc can
# import stablebear without the compiled C++ extension.
cpp_src = os.path.abspath("./_sb_cpp")
cpp_dest = os.path.join(stablebear_temp_dir, "_sb_cpp")

if is_windows:
    shutil.copytree(cpp_src, cpp_dest)
else:
    os.symlink(cpp_src, cpp_dest)

sys.path.insert(0, temp_mod_dir)
