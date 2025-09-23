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
import re
import sys
import m2r2

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


# -- Post-build hook: generate llm.txt (Markdown) from api/sdialog.html -----
def _extract_markdown_from_html(html: str) -> str:
    """Convert HTML to Markdown, preserving structure when possible.

    Uses `markdownify` if available; otherwise falls back to a naive tag-strip.
    """
    try:
        from markdownify import markdownify as md  # type: ignore

        # Strip noisy tags and prefer ATX (#) headings
        md_text = md(
            html,
            heading_style="ATX",
            strip=["script", "style", "a"],
        )
    except Exception:
        # Fallback: remove script/style, strip tags, collapse whitespace
        md_text = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", "", html, flags=re.I)
        md_text = re.sub(r"<[^>]+>", "\n", md_text)
        # Collapse space but keep some line breaks
        md_text = re.sub(r"\r?\n\s*\r?\n\s*\r?\n+", "\n\n", md_text)
        md_text = re.sub(r"[ \t]+", " ", md_text)

    # Normalize consecutive blank lines to max 2 and trim
    lines = [ln.rstrip() for ln in md_text.splitlines()]
    compact = []
    blank_count = 0
    for ln in lines:
        if ln.strip() == "":
            blank_count += 1
            if blank_count > 2:
                continue
            compact.append("")
        else:
            blank_count = 0
            compact.append(ln)
    out = "\n".join(compact).strip()
    return (out + "\n") if out else ""


def _write_llm_txt(app, exception):  # noqa: D401
    """Sphinx build-finished hook to emit Markdown llm.txt from api/sdialog.html."""
    # Only run on successful HTML builds
    if exception is not None:
        return
    if getattr(app, "builder", None) is None or getattr(app.builder, "name", "") != "html":
        return

    from sphinx.util import logging as sphinx_logging  # type: ignore

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
        md_text = _extract_markdown_from_html(html)

        llm_txt_path = os.path.join(outdir, "llm.txt")
        with open(llm_txt_path, "w", encoding="utf-8") as f:
            f.write(md_text)
        logger.info("llm.txt: generated at %s", llm_txt_path)
    except Exception as e:
        logger.warning("llm.txt: failed to generate (%s)", e)


def _prebuild_about_from_md(app):  # noqa: D401
    """Sphinx builder-inited hook: generate docs/about/*.rst from root *.md."""
    try:
        from sphinx.util import logging as sphinx_logging  # type: ignore
        logger = sphinx_logging.getLogger(__name__)
    except Exception:  # pragma: no cover - fallback basic logging
        import logging as _logging
        logger = _logging.getLogger(__name__)

    docs_dir = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(docs_dir, os.pardir))

    sources = {
        os.path.join(repo_root, "CHANGELOG.md"): os.path.join(docs_dir, "about", "changelog.rst"),
        os.path.join(repo_root, "CONTRIBUTING.md"): os.path.join(docs_dir, "about", "contributing.rst"),
    }

    os.makedirs(os.path.join(docs_dir, "about"), exist_ok=True)

    for src_md, dst_rst in sources.items():
        if not os.path.exists(src_md):
            logger.warning("pre-build: source Markdown not found: %s", src_md)
            # Write a tiny placeholder to avoid Sphinx errors
            placeholder = os.path.splitext(os.path.basename(dst_rst))[0].replace("_", " ").title()
            content = f"{placeholder}\n{'=' * len(placeholder)}\n\n(autogenerated placeholder)\n"
            try:
                with open(dst_rst, "w", encoding="utf-8") as f:
                    f.write(content)
            except Exception:
                pass
            continue

        try:
            with open(src_md, "r", encoding="utf-8") as f:
                md_text = f.read()
            rst_text = m2r2.convert(md_text)

            # Write only if changed
            write = True
            if os.path.exists(dst_rst):
                try:
                    with open(dst_rst, "r", encoding="utf-8") as f:
                        old = f.read()
                    write = (old != rst_text)
                except Exception:
                    write = True

            if write:
                with open(dst_rst, "w", encoding="utf-8") as f:
                    f.write(rst_text)
                logger.info("pre-build: generated %s from %s",
                            os.path.relpath(dst_rst, docs_dir), os.path.relpath(src_md, repo_root))
            else:
                logger.info("pre-build: up-to-date %s", os.path.relpath(dst_rst, docs_dir))
        except Exception as e:
            logger.warning("pre-build: failed to convert %s (%s)", src_md, e)


def setup(app):  # noqa: D401
    """Sphinx entry point: register build-finished hook."""
    # Generate about/*.rst from Markdown before sources are read
    app.connect('builder-inited', _prebuild_about_from_md)
    # Post-build: emit llm.txt markdown snapshot
    app.connect('build-finished', _write_llm_txt)
