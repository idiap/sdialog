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
import re

from os import path
from codecs import open
from datetime import datetime

# Ensure the source directory is in sys.path for autodoc to find the modules
sys.path.insert(0, os.path.abspath('../src/'))


# Mock imports for Sphinx documentation generation
autodoc_mock_imports = [
    # Heavy / optional scientific stack
    'numpy', 'pandas', 'scipy', 'torch', 'transformers', 'sklearn', 'matplotlib', 'networkx',
    # Embedding / NLP helper libs
    'sentence_transformers', 'simpleneighbors', 'syllables',
    # LangChain ecosystem
    'langchain_core', 'langchain_ollama', 'langchain_openai', 'langchain_aws',
    'langchain_google_genai', 'langchain_huggingface',
    # Model / orchestration related
    'ollama', 'openai',
    # Utility libs
    'tqdm', 'print_color', 'pydantic', 'jinja2', 'yaml', 'graphviz', 'PIL', 'tenacity', 'joblib'
]


# -- Project version ---------------------------------------------------------

_version_re__ = r"__version__\s*=\s*['\"]([^'\"]+)['\"]"
__cwd__ = path.abspath(path.dirname(__file__))
__init_file__ = path.join(__cwd__, '../src/sdialog/util.py')
with open(__init_file__, encoding="utf-8") as __init__py:
    __version__ = re.search(_version_re__, __init__py.read()).group(1)

# -- Project information -----------------------------------------------------

project = 'SDialog'
copyright = '{year}, The SDialog Development Team'.format(
    year=datetime.now().year
)
author = 'Sergio Burdisso'

version = __version__
release = version


master_doc = 'index'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx_rtd_theme',
              'sphinx.ext.imgmath']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['cmd.py']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'  # 'nature'
# html_logo = "_static/logo.png"
# html_favicon = "_static/favicon.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
