.PHONY: help install install-dev test test-verbose test-coverage lint format type-check build clean clean-all docs serve-docs pre-commit setup-dev all

# Default target
help:
	@echo "Available targets:"
	@echo "  help          Show this help message"
	@echo "  setup         Install package in current environment"
	@echo "  setup-dev     Setup development environment"
	@echo "  test          Run tests with pytest"
	@echo "  test-verbose  Run tests with verbose output"
	@echo "  test-coverage Run tests with coverage report"
	@echo "  lint          Run flake8 and ruff linting"
	@echo "  format        Format code with ruff"
	@echo "  type-check    Run type checking (if mypy is available)"
	@echo "  build         Build package distribution"
	@echo "  clean         Clean Python cache files"
	@echo "  clean-all     Clean all generated files (cache, build, dist)"
	@echo "  docs          Build documentation"
	@echo "  docs-serve    Serve documentation locally"
	@echo "  pre-commit    Run pre-commit hooks"

	@echo "  all           Run lint, test, and build"


all: lint test build
	@echo "All checks passed!"

check: lint test
	@echo "Code quality and tests passed!"

install:
	pip install -e .

setup:
	python -m venv .venv
	source .venv/bin/activate; pip install -r requirements.txt
	source .venv/bin/activate; pip install -r requirements-audio.txt

setup-dev:
	@if [ -f requirements-dev.txt ]; then source .venv/bin/activate; pip install -r requirements-dev.txt; fi
	@echo "Setting up pre-commit hooks..."
	pre-commit install
	@echo "Development environment setup complete!"

test:
	pytest -s

test-verbose:
	pytest -v -s

test-coverage:
	pytest --cov=sdialog --cov-report=html --cov-report=term-missing

lint:
	@echo "Running flake8..."
	flake8 src tests --ignore=W503 --max-line-length=120
	@echo "Running ruff check..."
	ruff check src tests --output-format=full

format:
	@echo "Formatting code with ruff..."
	ruff format src tests

type-check:
	@echo "Running ruff type checking..."
	ruff check src tests --select=E9,F63,F7,F82 --statistics  --output-format=full



build:
	@echo "Building package..."
	python -m build


# Documentation
docs:
	cd ./docs/ ; make html

docs-serve:
	@if [ -d "docs/_build/html" ]; then \
		echo "Serving documentation on http://localhost:8000"; \
		cd docs/_build/html && python -m http.server 8000; \
	else \
		echo "Documentation not built. Run 'make docs' first."; \
	fi

pre-commit:
	pre-commit run --all-files



# Clean up
clean:
	@echo "Cleaning Python cache files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyo" -delete

clean-all: clean
	@echo "Cleaning all generated files..."
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	find . -type d -name ".tox" -exec rm -rf {} +
