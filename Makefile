.PHONY: test validate fmt lint build bench pack deploy video spacex-demo starship-demo demo-all
.PHONY: demo-aerospace demo-telecom demo-finance demo-healthcare demo-energy
.PHONY: demo-transportation demo-manufacturing demo-agritech demo-all-verticals
.PHONY: test validate fmt lint build bench pack deploy video spacex-demo starship-demo demo-all demos
.PHONY: audit autonomy-test
.PHONY: sbom sbom-generate sbom-validate sbom-diff sbom-clean

ROOT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# SpaceX/NASA Pilot Track Demo Targets
spacex-demo:
	@echo "Running SpaceX Falcon 9 Stage 1 demo..."
	python3 quasim_spacex_demo.py --profile configs/meco_profiles/spacex_f9_stage1.json

starship-demo:
	@echo "Running Starship hot-staging demo..."
	python3 quasim_spacex_demo.py --profile configs/meco_profiles/starship_hotstaging.json

demo-all: spacex-demo starship-demo
	@echo "All pilot track demos complete!"

# Vertical Market Demo Targets
demo-aerospace:
	@echo "Running Aerospace & Defense demo..."
	python3 demos/quasim_aerospace_demo.py --profile configs/vertical_profiles/aerospace_f9.json

demo-telecom:
	@echo "Running Telecom & Satellite Constellations demo..."
	python3 demos/quasim_telecom_demo.py --profile configs/vertical_profiles/telecom_constellation.json

demo-finance:
	@echo "Running Finance - Portfolio Risk demo..."
	python3 demos/quasim_finance_demo.py --profile configs/vertical_profiles/finance_var.json

demo-healthcare:
	@echo "Running Healthcare Genomics demo..."
	python3 demos/quasim_healthcare_demo.py --profile configs/vertical_profiles/healthcare_genomics.json

demo-energy:
	@echo "Running Energy Grid Optimization demo..."
	python3 demos/quasim_energy_demo.py --profile configs/vertical_profiles/energy_grid.json

demo-transportation:
	@echo "Running Transportation & Logistics demo..."
	python3 demos/quasim_transportation_demo.py --profile configs/vertical_profiles/transportation_logistics.json

demo-manufacturing:
	@echo "Running Manufacturing IIoT demo..."
	python3 demos/quasim_manufacturing_demo.py --profile configs/vertical_profiles/manufacturing_iiot.json

demo-agritech:
	@echo "Running Agritech Precision Agriculture demo..."
	python3 demos/quasim_agritech_demo.py --profile configs/vertical_profiles/agritech_optimization.json

demo-all-verticals: demo-aerospace demo-telecom demo-finance demo-healthcare demo-energy demo-transportation demo-manufacturing demo-agritech
	@echo "All vertical market demos complete!"
# Run all vertical demo smoke tests
demos:
	@echo "Running smoke tests for all 8 vertical demos..."
	@python3 -m pytest quasim/demos/*/tests/test_*_smoke.py -q --tb=short
	@echo "✅ All demo smoke tests passed!"

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

# Run full repository audit
audit:
	@echo "Running QuASIM full repository audit..."
	@python3 -m quasim.audit.run --full --export-json audit/audit_summary.json

# Run Phase VIII autonomy tests
autonomy-test:
	@echo "Running Phase VIII autonomy tests..."
	@pytest tests/phaseVIII/ -v --tb=short

# =============================================================================
# SBOM (Software Bill of Materials) Targets
# =============================================================================

# Generate all SBOMs (CycloneDX and SPDX)
sbom-generate:
	@echo "Generating SBOMs for QRATUM..."
	@chmod +x scripts/sbom/generate_sbom.sh
	@bash scripts/sbom/generate_sbom.sh

# Validate generated SBOMs
sbom-validate:
	@echo "Validating SBOMs..."
	@chmod +x scripts/sbom/validate_sbom.sh
	@bash scripts/sbom/validate_sbom.sh

# Diff SBOMs against baseline
sbom-diff:
	@echo "Comparing SBOMs with baseline..."
	@chmod +x scripts/sbom/diff_sbom.sh
	@bash scripts/sbom/diff_sbom.sh

# Full SBOM workflow: generate, validate
sbom: sbom-generate sbom-validate
	@echo "SBOM generation and validation complete!"
	@echo "Artifacts in .sbom/"

# Clean generated SBOM artifacts
sbom-clean:
	@echo "Cleaning SBOM artifacts..."
	@rm -rf .sbom/cyclonedx/*.json .sbom/spdx/*.json
	@rm -rf .sbom/hashes/*.sha256 .sbom/hashes/*.sha512
	@rm -rf .sbom/provenance/*.json .sbom/diffs/*.json
	@echo "SBOM artifacts cleaned"
