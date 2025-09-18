# Contributions

Thanks for your interest in the project, you're ![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)!!

Any kind of help is very welcome (Code, Bug reports, Content, Data, Documentation, Design, Examples, Ideas, Feedback, etc.),  Issues and/or Pull Requests are welcome for any level of improvement, from a small typo to new features, help us make SDialog better :+1:

Remember that you can use the "Edit" button ('pencil' icon) up the top to [edit any file of this repo directly on GitHub](https://help.github.com/en/github/managing-files-in-a-repository/editing-files-in-your-repository).

In case you're planning to create a **new Pull Request**, for committing to this repo, we follow the Chris Beams' "seven rules of a great Git commit message" from ["How to Write a Git Commit Message"](https://chris.beams.io/posts/git-commit/), so make sure your commits follow them as well.

## Manually Build Documentation

Generate the HTML version of it:
```bash
cd docs
python -m sphinx -T -b html -d _build/doctrees -D language=en . ../docs_html
```

In case we need to re-generate the API Reference:
```bash
cd docs/api
sphinx-apidoc -f --ext-autodoc -o api ../src/sdialog
```

Link to our ReadTheDocs [here](https://app.readthedocs.org/projects/sdialog/builds/28462329/).

## PyPI

```bash
python -m build
python -m twine upload dist/*
```

## Tests

Make sure your changes passes the style and unit tests, run the following commands in the root directory:

```bash
flake8 --ignore=W503
pytest
```

Or if you want to check the code coverture:
```bash
pytest -v --cov-report=html --cov=src/sdialog
```
And check the content of the newlly created `htmlcov` folder (open `index.html`).
