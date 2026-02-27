.PHONY: install
install:
	@echo "Creating virtual environment and installing dependencies using uv..."
	uv venv
	uv lock
	uv sync --all-groups
	uv run pre-commit install

.PHONY: build
build:
	@echo "Installing build dependencies..."
	uv pip install scikit-build-core numpy
	@echo "Compiling C-extensions..."
	uv pip install --no-build-isolation -e .

.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	find . -name "*.so" -not -path "./.git/*" -delete
	rm -rf _skbuild build dist *.egg-info
	rm -rf src/*.egg-info
	rm -rf .pytest_cache
	rm -rf .venv

.PHONY: rebuild
rebuild: clean install build

.PHONY: test
test:
	@echo "Running tests with pytest..."
	uv run pytest tests/ -v

.PHONY: lint
lint:
	@echo "Running linter with ruff..."
	uv run ruff format . --config pyproject.toml
	@echo "Running checks with ruff..."
	uv run ruff check . --config pyproject.toml

.PHONY: ci
ci:
	@echo "This target attempts to simulate running tests and linting"
	$(MAKE) test
	$(MAKE) lint

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  install  - Set up virtual environment and install all dependencies"
	@echo "  build    - Compile C extensions (editable install)"
	@echo "  rebuild  - Clean and recompile everything from scratch"
	@echo "  clean    - Remove all build artifacts (.so, dist, .venv)"
	@echo "  test     - Run pytest"
	@echo "  lint     - Run ruff and cython-lint"
	@echo "  ci       - Full pipeline: rebuild + test + lint"
