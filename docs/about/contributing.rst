
Contributing
============

Thanks for your interest in the project — you're awesome! 😎🎉

Any kind of help is welcome (Code, Bug reports, Content, Data, Documentation, Design, Examples, Ideas, Feedback, etc.). Issues and Pull Requests are encouraged: from a tiny typo fix to a new feature. Help us make SDialog better 👍

You can use the Edit button (pencil icon) on GitHub to quickly propose changes to any file via the web UI.

We follow `Chris Beams' guidelines <https://chris.beams.io/posts/git-commit/>`_ for commit messages.

Development installation
------------------------

.. code-block:: bash

   git clone git@github.com:idiap/sdialog.git
   cd sdialog
   pip install -e .

..

   [!NOTE]
   If you plan to work with the audio capabilities of SDialog (\ ``sdialog.audio``\ ), install with audio dependencies:

   .. code-block:: bash

      pip install -e .[audio]


Running tests & style
---------------------

.. code-block:: bash

   flake8 --ignore=W503 --max-line-length=120
   pytest -v

Coverage (HTML + terminal):

.. code-block:: bash

   pytest -v --cov=src/sdialog --cov-report=term-missing --cov-report=html
   # Open htmlcov/index.html

Manual documentation build
--------------------------

Generate HTML:

.. code-block:: bash

   cd docs
   python -m sphinx -T -b html -d _build/doctrees -D language=en . ../docs_html

Regenerate API reference (only needed if new submodules are are added):

.. code-block:: bash

   cd docs
   sphinx-apidoc -f --ext-autodoc -o api ../src/sdialog

ReadTheDocs latest build list: https://app.readthedocs.org/projects/sdialog/

Release (PyPI)
--------------


#. Update version in ``src/sdialog/util.py`` (follow semver)
#. Update CHANGELOG (if present)
#. Tag & push
   .. code-block:: bash

      git commit -m "Release v0.x.x"
      git tag v0.x.x
      git push origin main --tags

#. Build & upload:
   .. code-block:: bash

      python -m build
      python -m twine upload dist/*

Guidelines
----------


* Keep functions/classes small & composable
* Add/extend tests for new features or bug fixes
* Document public APIs (docstrings + docs reference where appropriate)
* Prefer pure functions where state is not needed
* Avoid introducing heavy deps without discussion (open issue first)
* Use meaningful names; avoid abbreviations except standard ones (LLM, NLP, etc.)

Adding tutorials / notebooks
----------------------------

Place new notebooks under ``tutorials/`` and keep naming numeric + descriptive (e.g., ``8.new_feature_example.ipynb``\ ). Ensure they run top-to-bottom in Colab. Use lightweight models or small number of elements to keep runtime short.

Opening an issue
----------------

Provide:


* Summary
* Steps to reproduce (if bug)
* Expected vs actual
* Environment (Python version, OS, backend model)
* Minimal reproducible code snippet

Pull request checklist
----------------------


* [ ] Feature / bug issue linked (if applicable)
* [ ] Tests added or updated
* [ ] Docs / examples updated
* [ ] No lint errors
* [ ] Local tests pass
* [ ] Changelog updated (if user-facing change)

Communication
-------------

Use GitHub Issues / Discussions for feature proposals. For larger changes, open a draft PR early for feedback.

AI-assisted development
-----------------------

This project provides an `llm.txt file <https://sdialog.readthedocs.io/en/latest/llm.txt>`_ following the `llms.txt specification <https://llmstxt.org/>`_ for AI coding assistants. GitHub Copilot and other AI tools can fetch structured project information with: ``#fetch https://sdialog.readthedocs.io/en/latest/llm.txt``

Thanks
------

Your contributions make the project better for everyone. 🙏
