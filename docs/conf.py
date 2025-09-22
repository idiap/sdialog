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
    'tqdm', 'print_color', 'jinja2', 'graphviz', 'PIL', 'tenacity', 'joblib'
]

autodoc_default_options = {
    'exclude-members': ','.join(['model_post_init', 'model_config'])
}

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


# -- Post-build hook: generate llm.txt from api/sdialog.html --------------
def _extract_text_from_html(html: str) -> str:
    """Extract readable text from HTML.

    Tries BeautifulSoup if available; falls back to a simple tag-strip regex.
    """
    # Prefer BeautifulSoup when available for better structure handling
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "html.parser")
        # Remove script/style to avoid noise
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
    except Exception:
        # Very naive fallback: strip tags and collapse whitespace
        import re as _re

        text = _re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", "", html, flags=_re.I)
        text = _re.sub(r"<[^>]+>", "\n", text)
        text = _re.sub(r"\s+", " ", text)
        # Re-introduce minimal line breaks
        text = text.replace(" . ", ". ")
    # Normalize consecutive blank lines
    lines = [ln.strip() for ln in text.splitlines()]
    compact = []
    last_blank = False
    for ln in lines:
        blank = (ln == "")
        if blank and last_blank:
            continue
        compact.append(ln)
        last_blank = blank
    return "\n".join(compact).strip() + "\n"


def _write_llm_txt(app, exception):  # noqa: D401
    """Sphinx build-finished hook to emit llm.txt from api/sdialog.html."""
    # Only run on successful HTML builds
    if exception is not None:
        return
    if getattr(app, "builder", None) is None or getattr(app.builder, "name", "") != "html":
        return

    from sphinx.util import logging as sphinx_logging

    logger = sphinx_logging.getLogger(__name__)
    outdir = getattr(app.builder, "outdir", None)
    if not outdir:
        logger.warning("llm.txt: output directory not found; skipping.")
        return

    html_path = os.path.join(outdir, "api", "sdialog.html")
    if not os.path.exists(html_path):
        logger.warning("llm.txt: '%s' not found; skipping.", html_path)
        return

    try:
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
        text = _extract_text_from_html(html)

        llm_txt_path = os.path.join(outdir, "llm.txt")
        with open(llm_txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info("llm.txt: generated at %s", llm_txt_path)
    except Exception as e:
        logger.warning("llm.txt: failed to generate (%s)", e)


def setup(app):  # noqa: D401
    """Sphinx entry point: register build-finished hook."""
    app.connect('build-finished', _write_llm_txt)
