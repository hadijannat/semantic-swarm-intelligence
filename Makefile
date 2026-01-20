.PHONY: install lint format typecheck test test-cov test-unit test-int check clean pre-commit pre-commit-all
.PHONY: docker-build docker-up docker-down docker-logs train benchmark reproduce help

# Install dependencies
install:
	poetry install --with ml

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

# Run unit tests only
test-unit:
	poetry run pytest tests/unit/ -v

# Run integration tests only
test-int:
	poetry run pytest tests/integration/ -v

# Docker commands
docker-build:
	docker compose -f docker/docker-compose.dev.yml build

docker-up:
	docker compose -f docker/docker-compose.dev.yml up -d
	@echo "Services started:"
	@echo "  API:     http://localhost:8000"
	@echo "  Gradio:  http://localhost:7860"
	@echo "  Flower:  http://localhost:8080"
	@echo "  MQTT:    localhost:1883"
	@echo "  Postgres: localhost:5432"

docker-down:
	docker compose -f docker/docker-compose.dev.yml down

docker-logs:
	docker compose -f docker/docker-compose.dev.yml logs -f

# ML training and benchmarks
train:
	poetry run python scripts/train_baseline.py --seed 42 --epochs 10

benchmark:
	./scripts/run_benchmarks.sh --quick

reproduce:
	./scripts/run_benchmarks.sh

# Help
help:
	@echo "NOA Semantic Swarm Mapper - Available Commands"
	@echo "=============================================="
	@echo ""
	@echo "Development:"
	@echo "  make install      - Install dependencies"
	@echo "  make lint         - Run linting"
	@echo "  make format       - Format code"
	@echo "  make typecheck    - Type checking"
	@echo ""
	@echo "Testing:"
	@echo "  make test         - Run all tests"
	@echo "  make test-unit    - Unit tests only"
	@echo "  make test-int     - Integration tests"
	@echo "  make test-cov     - Tests with coverage"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build - Build images"
	@echo "  make docker-up    - Start stack"
	@echo "  make docker-down  - Stop stack"
	@echo ""
	@echo "ML:"
	@echo "  make train        - Train baseline model"
	@echo "  make benchmark    - Quick benchmark"
	@echo "  make reproduce    - Full reproducibility test"
