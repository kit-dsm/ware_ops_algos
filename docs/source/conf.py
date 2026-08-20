import os
import sys


sys.path.insert(0, os.path.abspath("../../src"))

project = "ware_ops_algos"
author = "Janik Bischoff"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]
html_title = "ware_ops_algos"
html_short_title = "ware_ops_algos"
html_logo = "_static/favicon.png"
html_favicon = "_static/favicon.svg"

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
