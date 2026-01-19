.PHONY: install lint format typecheck test check clean pre-commit

# Install dependencies
install:
	poetry install

# Run linter
lint:
	poetry run ruff check src tests

# Run formatter
format:
	poetry run ruff format src tests
	poetry run ruff check --fix src tests

# Run type checker
typecheck:
	poetry run mypy src/noa_swarm

# Run tests
test:
	poetry run pytest tests/ -v

# Run tests with coverage
test-cov:
	poetry run pytest tests/ --cov=src/noa_swarm --cov-report=term-missing --cov-report=html

# Run all checks (lint, typecheck, test)
check: lint typecheck test

# Install pre-commit hooks
pre-commit:
	poetry run pre-commit install

# Run pre-commit on all files
pre-commit-all:
	poetry run pre-commit run --all-files

# Clean up cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type f -name "coverage.xml" -delete 2>/dev/null || true
