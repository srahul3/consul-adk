# =============================================================================
# Makefile for consul-adk
# =============================================================================
# Purpose: Automate common development tasks including building, testing,
# linting, and deployment for the consul-adk project.
# =============================================================================

.PHONY: help install install-dev test test-verbose test-client test-server \
        lint format clean build dist check coverage docs run-server \
        run-example setup-dev all

# Default target
.DEFAULT_GOAL := help

# Project configuration
PYTHON := python3
PIP := pip3
PACKAGE_NAME := consul-adk
TEST_DIR := tests
SOURCE_DIRS := client models server utilities

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# =============================================================================
# Help Target
# =============================================================================
help: ## Show this help message
	@echo "$(BLUE)consul-adk Makefile$(NC)"
	@echo "=================================="
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""

# =============================================================================
# Installation Targets
# =============================================================================
install: ## Install the package and its dependencies
	@echo "$(GREEN)Installing package dependencies...$(NC)"
	$(PIP) install -e .

install-dev: ## Install development dependencies
	@echo "$(GREEN)Setting up development environment...$(NC)"
	@echo "$(YELLOW)Creating virtual environment...$(NC)"
	$(PYTHON) -m venv .venv
	@echo "$(YELLOW)Activating virtual environment and installing dependencies...$(NC)"
	. .venv/bin/activate && \
	$(PIP) install . && \
	$(PIP) install pytest pytest-asyncio pytest-cov black flake8 mypy isort
	@echo "$(GREEN)Development environment ready!$(NC)"
	@echo "$(BLUE)To activate the virtual environment manually, run:$(NC)"
	@echo "  $(YELLOW)source .venv/bin/activate$(NC)"

setup-dev: install-dev ## Complete development setup (alias for install-dev)

# =============================================================================
# Testing Targets
# =============================================================================
test: ## Run all tests
	@echo "$(GREEN)Running all tests...$(NC)"
	pytest $(TEST_DIR) -v

test-verbose: ## Run all tests with verbose output
	@echo "$(GREEN)Running all tests with verbose output...$(NC)"
	pytest $(TEST_DIR) -vvv --tb=long

test-client: ## Run only client tests
	@echo "$(GREEN)Running client tests...$(NC)"
	pytest $(TEST_DIR)/test_client.py -v

test-server: ## Run only server tests
	@echo "$(GREEN)Running server tests...$(NC)"
	pytest $(TEST_DIR)/test_server.py -v

test-watch: ## Run tests in watch mode (requires pytest-watch)
	@echo "$(GREEN)Running tests in watch mode...$(NC)"
	@which ptw > /dev/null || (echo "$(RED)pytest-watch not installed. Run: pip install pytest-watch$(NC)" && exit 1)
	ptw $(TEST_DIR)

coverage: ## Run tests with coverage report
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	pytest $(TEST_DIR) --cov=$(SOURCE_DIRS) --cov-report=html --cov-report=term
	@echo "$(GREEN)Coverage report generated in htmlcov/$(NC)"

coverage-xml: ## Generate XML coverage report for CI
	@echo "$(GREEN)Generating XML coverage report...$(NC)"
	pytest $(TEST_DIR) --cov=$(SOURCE_DIRS) --cov-report=xml

# =============================================================================
# Code Quality Targets
# =============================================================================
lint: ## Run linting checks
	@echo "$(GREEN)Running linting checks...$(NC)"
	@echo "$(YELLOW)Running flake8...$(NC)"
	flake8 $(SOURCE_DIRS) $(TEST_DIR) --max-line-length=100 --ignore=E203,W503
	@echo "$(YELLOW)Running mypy...$(NC)"
	mypy $(SOURCE_DIRS) --ignore-missing-imports
	@echo "$(GREEN)Linting completed!$(NC)"

format: ## Format code with black and isort
	@echo "$(GREEN)Formatting code...$(NC)"
	@echo "$(YELLOW)Running isort...$(NC)"
	isort $(SOURCE_DIRS) $(TEST_DIR)
	@echo "$(YELLOW)Running black...$(NC)"
	black $(SOURCE_DIRS) $(TEST_DIR)
	@echo "$(GREEN)Code formatting completed!$(NC)"

format-check: ## Check if code formatting is correct
	@echo "$(GREEN)Checking code formatting...$(NC)"
	black --check $(SOURCE_DIRS) $(TEST_DIR)
	isort --check-only $(SOURCE_DIRS) $(TEST_DIR)

check: lint format-check test ## Run all quality checks (lint, format-check, test)

# =============================================================================
# Build Targets
# =============================================================================
clean: ## Clean build artifacts and cache files
	@echo "$(GREEN)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)Clean completed!$(NC)"

build: clean ## Build the package
	@echo "$(GREEN)Building package...$(NC)"
	$(PYTHON) -m build

dist: build ## Create distribution packages
	@echo "$(GREEN)Creating distribution packages...$(NC)"
	@echo "$(GREEN)Distribution packages created in dist/$(NC)"
	ls -la dist/

# =============================================================================
# Development Server Targets
# =============================================================================
run-server: ## Run the development server (requires implementation)
	@echo "$(GREEN)Starting development server...$(NC)"
	@echo "$(YELLOW)Note: Implement server startup script as needed$(NC)"
	# $(PYTHON) -m server.server

run-example: ## Run example implementation
	@echo "$(GREEN)Running example...$(NC)"
	@echo "$(YELLOW)Note: Add example script path as needed$(NC)"
	# $(PYTHON) examples/example_agent.py

# =============================================================================
# Documentation Targets
# =============================================================================
docs: ## Generate documentation (placeholder)
	@echo "$(GREEN)Generating documentation...$(NC)"
	@echo "$(YELLOW)Note: Add documentation generation as needed$(NC)"
	# sphinx-build -b html docs docs/_build

# =============================================================================
# Release Targets
# =============================================================================
version: ## Show current version
	@echo "$(GREEN)Current version:$(NC)"
	@$(PYTHON) -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"

upload-test: dist ## Upload to test PyPI
	@echo "$(GREEN)Uploading to test PyPI...$(NC)"
	twine upload --repository testpypi dist/*

upload: dist ## Upload to PyPI
	@echo "$(RED)Uploading to PyPI...$(NC)"
	@read -p "Are you sure you want to upload to PyPI? (y/N): " confirm && [ "$$confirm" = "y" ]
	twine upload dist/*

# =============================================================================
# Development Workflow Targets
# =============================================================================
dev-check: ## Quick development check (format + lint + test)
	@echo "$(GREEN)Running development checks...$(NC)"
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) test

dev-setup: install-dev ## Setup development environment
	@echo "$(GREEN)Development environment setup complete!$(NC)"
	@echo "$(BLUE)Next steps:$(NC)"
	@echo "  - Run '$(YELLOW)make test$(NC)' to run tests"
	@echo "  - Run '$(YELLOW)make dev-check$(NC)' for quick quality checks"
	@echo "  - Run '$(YELLOW)make help$(NC)' to see all available commands"

all: clean install-dev check build ## Run complete pipeline (clean, install, check, build)

# =============================================================================
# CI/CD Targets
# =============================================================================
ci-test: ## Run tests for CI (with XML coverage)
	@echo "$(GREEN)Running CI tests...$(NC)"
	pytest $(TEST_DIR) -v --junitxml=test-results.xml
	$(MAKE) coverage-xml

ci-check: ## Run all CI checks
	@echo "$(GREEN)Running CI checks...$(NC)"
	$(MAKE) format-check
	$(MAKE) lint
	$(MAKE) ci-test

# =============================================================================
# Docker Targets (if needed)
# =============================================================================
docker-build: ## Build Docker image (placeholder)
	@echo "$(GREEN)Building Docker image...$(NC)"
	@echo "$(YELLOW)Note: Add Dockerfile and docker build command$(NC)"
	# docker build -t $(PACKAGE_NAME):latest .

docker-run: ## Run Docker container (placeholder)
	@echo "$(GREEN)Running Docker container...$(NC)"
	@echo "$(YELLOW)Note: Add docker run command$(NC)"
	# docker run -p 5000:5000 $(PACKAGE_NAME):latest

# =============================================================================
# Utility Targets
# =============================================================================
deps-update: ## Update dependencies (placeholder)
	@echo "$(GREEN)Updating dependencies...$(NC)"
	@echo "$(YELLOW)Note: Review and update pyproject.toml dependencies$(NC)"
	$(PIP) list --outdated

deps-tree: ## Show dependency tree
	@echo "$(GREEN)Dependency tree:$(NC)"
	@which pipdeptree > /dev/null || (echo "$(YELLOW)Installing pipdeptree...$(NC)" && $(PIP) install pipdeptree)
	pipdeptree

info: ## Show project information
	@echo "$(BLUE)Project Information$(NC)"
	@echo "==================="
	@echo "$(YELLOW)Name:$(NC) $(PACKAGE_NAME)"
	@echo "$(YELLOW)Python:$(NC) $(shell $(PYTHON) --version)"
	@echo "$(YELLOW)Pip:$(NC) $(shell $(PIP) --version)"
	@echo "$(YELLOW)Test Directory:$(NC) $(TEST_DIR)"
	@echo "$(YELLOW)Source Directories:$(NC) $(SOURCE_DIRS)"
	@echo ""
	@$(MAKE) version
