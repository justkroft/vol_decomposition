install
	@echo "Creating virtual environment and installing dependencies using uv..."
	uv lock
	uv sync --all-groups
	uv run pre-commit install

.PHONY: test
	@echo "Running tests with pytest..."
	uv run pytest tests/ -v

.PHONY: lint
	@echo "Running linter with ruff..."
	uv run ruff format . --config pyproject.toml
	@echo "Running checks with ruff..."
	uv run ruff check . --config pyproject.toml

.PHONY: ci
	@echo "<ind that this target attempts to simulate running tests and linting"
	$(MAKE) test
	$(MAKE) lint
