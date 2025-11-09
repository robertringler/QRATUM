.PHONY: test validate fmt lint build bench pack deploy video

ROOT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Format code (Python, Terraform)
fmt:
	@echo "Running Terraform fmt (if terraform is available)..."
	@if command -v terraform >/dev/null 2>&1; then \
	terraform fmt -recursive ; \
	else \
	echo "terraform CLI not installed; skipping fmt"; \
	fi
	@echo "Running Python formatting (if ruff/black is available)..."
	@if command -v ruff >/dev/null 2>&1; then \
	ruff format . ; \
	else \
	echo "ruff CLI not installed; skipping Python fmt"; \
	fi

# Lint code (Python, C++, YAML)
lint:
	@echo "Running linters..."
	@if command -v ruff >/dev/null 2>&1; then \
	ruff check . ; \
	else \
	echo "ruff CLI not installed; skipping Python linting"; \
	fi

# Validate infrastructure and run tests
validate test:
	@python3 scripts/test_full_stack.py

# Build QuASIM components
build:
	@echo "Building QuASIM components..."
	@echo "Note: Full build requires CUDA toolkit and dependencies"
	@echo "Validating Python modules..."
	@python3 -m py_compile quasim/__init__.py
	@python3 -m py_compile integrations/adapters/fluent/*.py 2>/dev/null || echo "Fluent adapter not yet implemented"

# Run benchmarks
bench:
	@echo "Running QuASIM benchmarks..."
	@if [ -f benchmarks/quasim_bench.py ]; then \
	python3 benchmarks/quasim_bench.py ; \
	fi
	@if [ -f integrations/benchmarks/aero/run_benchmarks.py ]; then \
	python3 integrations/benchmarks/aero/run_benchmarks.py ; \
	else \
	echo "Aerospace benchmarks not yet implemented"; \
	fi

# Generate video artifacts
video:
	@echo "Generating QuASIM video artifacts..."
	@python3 -m quasim.cli.run_flow --steps=150 --N=300 --T=3.0 --seed=42 --emit-json

# Package artifacts (containers, helm charts)
pack:
	@echo "Packaging QuASIM artifacts..."
	@echo "Note: Requires Docker for container builds"
	@if command -v docker >/dev/null 2>&1; then \
	echo "Docker available - ready to build containers"; \
	else \
	echo "Docker not installed; skipping container build"; \
	fi

# Deploy to Kubernetes (via Helm)
deploy:
	@echo "Deploying QuASIM to Kubernetes..."
	@echo "Note: Requires kubectl and helm, and active Kubernetes context"
	@if command -v kubectl >/dev/null 2>&1 && command -v helm >/dev/null 2>&1; then \
	echo "kubectl and helm available - ready to deploy"; \
	else \
	echo "kubectl or helm not installed; skipping deployment"; \
	fi

# Run full stack locally with Docker Compose
run-full-stack:
	@echo "Starting full stack with Docker Compose..."
	@if command -v docker >/dev/null 2>&1; then \
		docker compose up --build; \
	else \
		echo "Docker not installed. Please install Docker to run the full stack."; \
		exit 1; \
	fi

# Stop full stack
stop-full-stack:
	@echo "Stopping full stack..."
	@if command -v docker >/dev/null 2>&1; then \
		docker compose down; \
	else \
		echo "Docker not installed."; \
		exit 1; \
	fi
