import os
import sys

import pypandoc


sys.path.insert(0, os.path.abspath("../../src"))
pandoc_dir = os.path.dirname(pypandoc.get_pandoc_path())
os.environ["PATH"] = pandoc_dir + os.pathsep + os.environ.get("PATH", "")

project = "ware_ops_algos"
author = "Janik Bischoff"
release = "0.2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "nbsphinx",
    "nbsphinx_link",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]
html_title = "ware_ops_algos"
html_short_title = "ware_ops_algos"
html_logo = "_static/favicon.png"
html_favicon = "_static/favicon.svg"

nbsphinx_execute = "never"

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
