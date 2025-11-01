.PHONY: test validate fmt

ROOT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

fmt:
@echo "Running Terraform fmt (if terraform is available)..."
@if command -v terraform >/dev/null 2>&1; then \
terraform fmt -recursive ; \
else \
echo "terraform CLI not installed; skipping fmt"; \
fi

validate test:
@python3 scripts/test_full_stack.py
