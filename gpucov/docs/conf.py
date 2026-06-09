"""Sphinx configuration for GPUCov documentation."""

from datetime import datetime

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

pyproj_toml = tomllib.load(open("../pyproject.toml", "rb"))

year = datetime.now().year

project = "GPUCov"
copyright = f"2026, GPUCov contributors"
author = "GPUCov contributors"
release = pyproj_toml["project"]["version"]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_theme_options = {
    "github_url": "https://github.com/bwehlin/stablebear/tree/main/gpucov",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"
