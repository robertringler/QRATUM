# Contributing to Sybernix

Thank you for your interest in contributing to Sybernix! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Review Process](#review-process)

## Code of Conduct

This project adheres to professional standards of conduct. Please be respectful and constructive in all interactions.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/sybernix.git`
3. Add upstream remote: `git remote add upstream https://github.com/robertringler/sybernix.git`
4. Create a feature branch: `git checkout -b feature/your-feature-name`

## Development Setup

### Prerequisites

- Python 3.8+
- Terraform ≥ 1.7
- kubectl ≥ 1.29
- helm ≥ 3.14
- Docker (for container builds)
- CUDA toolkit (for GPU development, optional)

### Local Environment

```bash
# Install Python dependencies
pip install -r docker/requirements.txt

# Run validation suite
make test

# Run code formatting
make fmt
```

## Coding Standards

### Python

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use ruff for linting: `ruff check .`
- Use black for formatting (line-length=100)

### C++

- Follow DO-178C coding standards
- Use MISRA-like linting with clang-tidy
- Avoid undefined behavior
- Document all public APIs with Doxygen-style comments

### YAML/Terraform

- Use 2-space indentation
- Validate before committing: `make validate`
- Run `terraform fmt` on all Terraform files

### Documentation

- Use Markdown for documentation
- Keep README files up-to-date
- Document all public APIs
- Include examples for complex features

## Testing

### Unit Tests

- Write unit tests for all new functionality
- Aim for >90% code coverage on adapters and SDKs
- Use pytest for Python tests
- Use synthetic/minimal test data

```bash
# Run Python tests
pytest tests/

# Run with coverage
pytest --cov=. tests/
```

### Integration Tests

- Include integration tests for adapters
- Use minimal meshes and test data
- Ensure tests run in CI environment

### Benchmarks

- Add benchmarks for performance-critical code
- Document expected performance characteristics
- Use deterministic seeds for reproducibility

## Submitting Changes

### Commit Messages

Use conventional commit format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Test additions or modifications
- `chore`: Build process or auxiliary tool changes

Example:

```
feat(adapters): add Fluent CFD adapter with HDF5 support

Implements read-only file shim for Ansys Fluent mesh and boundary
conditions. Converts to QuASIM tensor format and writes back results
as CSV/HDF5/VTK for re-import.

Closes #123
```

### Pull Request Process

1. Update documentation for any changed functionality
2. Add or update tests as needed
3. Ensure all tests pass: `make test`
4. Run linting: `make lint` (if available)
5. Update CHANGELOG.md if applicable
6. Create pull request with descriptive title and summary
7. Reference related issues in PR description

### PR Title Format

Use conventional commit format for PR titles:

```
<type>(<scope>): <description>
```

Example:

```
feat(integrations): initial QuASIM adapters, API, and bench harness
```

## Review Process

1. Automated checks must pass (CI/CD)
2. At least one approval required from code owners
3. Address all review comments
4. Squash commits if requested
5. Maintainer will merge when ready

## Security and Compliance

### Security Issues

- Do NOT commit secrets, credentials, or sensitive data
- Report security vulnerabilities privately to maintainers
- Follow ITAR/export control guidelines for aerospace code

### Compliance Requirements

- Ensure DO-178C compliance for safety-critical code
- Generate SBOM for dependency changes
- Run SAST (CodeQL) before submitting
- Verify license compatibility for new dependencies

### ITAR Considerations

- No export-controlled payloads in public artifacts
- Document ITAR-clean build procedures
- Follow compliance/EXPORT.md guidelines

## Questions?

If you have questions or need help, please:

1. Check existing documentation
2. Search closed issues
3. Open a new issue with the `question` label

Thank you for contributing to Sybernix!
