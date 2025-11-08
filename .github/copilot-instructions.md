# GitHub Copilot Instructions for QuASIM

## Project Overview

QuASIM (Quantum-Inspired Autonomous Simulation) is a production-grade quantum simulation platform engineered for regulated industries requiring aerospace certification (DO-178C Level A), defense compliance (NIST 800-53/171, CMMC 2.0 L2, DFARS), and deterministic reproducibility. Built on a hybrid quantum-classical runtime with NVIDIA cuQuantum acceleration, QuASIM delivers GPU-accelerated tensor network simulation and multi-cloud Kubernetes orchestration with 99.95% SLA.

## Key Technologies

- **Primary Language**: Python 3.10+ (required minimum)
- **Secondary Languages**: C++, CUDA, Terraform, YAML
- **Quantum Framework**: NVIDIA cuQuantum, custom tensor network simulation
- **Infrastructure**: Kubernetes (EKS/GKE/AKS), Docker, Helm, ArgoCD
- **Observability**: Prometheus, Grafana, Loki, Tempo
- **Security**: HashiCorp Vault, OPA Gatekeeper, Cilium CNI
- **Testing**: pytest, pytest-cov, pytest-mock
- **CI/CD**: GitHub Actions with comprehensive validation

## Coding Standards

### Python

- **Style Guide**: Follow PEP 8 strictly
- **Line Length**: Maximum 100 characters
- **Formatter**: Use `ruff format` (configured in `pyproject.toml`)
- **Linter**: Use `ruff check` for all code validation
- **Type Hints**: Required for all public function signatures
- **Import Sorting**: Use isort profile "black" (configured via ruff)
- **Target Version**: Python 3.10+ features allowed

### C++ and CUDA

- **Standard**: Follow DO-178C coding standards for safety-critical code
- **Linting**: Use MISRA-like checks with clang-tidy
- **Documentation**: Doxygen-style comments for all public APIs
- **Safety**: Avoid undefined behavior; document all assumptions

### YAML and Terraform

- **Indentation**: 2 spaces
- **Validation**: Run `terraform fmt` on all Terraform files
- **Kubernetes**: Follow best practices for resource definitions

### Documentation

- **Format**: Markdown for all documentation
- **APIs**: Document all public APIs thoroughly
- **Examples**: Include usage examples for complex features
- **Updates**: Keep README files synchronized with code changes

## Development Workflow

### Before Making Changes

1. **Summarize the goal**: State your objective before starting
2. **Plan your approach**: Output a brief plan before making edits
3. **Minimal changes**: Make the smallest possible changes to achieve the goal
4. **Check existing code**: Review similar patterns in the codebase

### Building and Testing

```bash
# Format code
make fmt               # Runs ruff format and terraform fmt

# Lint code
make lint              # Runs ruff check

# Run tests
make test              # Runs validation suite via scripts/test_full_stack.py
pytest tests/          # Run specific test suites

# Run with coverage
pytest --cov=. tests/

# Build components
make build             # Validates Python modules

# Run benchmarks
make bench             # Executes performance benchmarks
```

### Testing Requirements

- **Coverage Target**: Maintain >90% code coverage for adapters and SDKs
- **Unit Tests**: Required for all new functionality
- **Test Framework**: Use pytest for Python tests
- **Test Data**: Use synthetic/minimal test data
- **Integration Tests**: Include for adapters with minimal test cases
- **Performance Tests**: Document expected characteristics for performance-critical code

### Commit Conventions

Use conventional commit format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Test additions or modifications
- `chore`: Build process or auxiliary tool changes

**Example**:
```
feat(kernels): add adaptive tensor contraction heuristics

Implements dynamic error budget allocation for tensor network
contraction with 10x compression improvement.

Closes #123
```

### Pull Request Process

1. Update documentation for any changed functionality
2. Add or update tests to maintain >90% coverage
3. Ensure all tests pass: `make test`
4. Run linting: `make lint`
5. Update CHANGELOG.md if applicable
6. Create PR with descriptive title using conventional commit format
7. Reference related issues in PR description
8. Address all automated checks and review comments

## Security and Compliance

### Security Requirements

- **NO SECRETS**: Never commit secrets, credentials, or sensitive data
- **SAST**: Run CodeQL before submitting changes
- **Dependencies**: Verify license compatibility for new dependencies
- **SBOM**: Generate Software Bill of Materials for dependency changes
- **Vulnerabilities**: Report security issues privately to maintainers

### Compliance Standards

This project maintains compliance with:

- **DO-178C Level A**: Aerospace software certification
- **NIST 800-53 Rev 5**: Federal security controls (HIGH baseline)
- **CMMC 2.0 Level 2**: Defense contractor cybersecurity
- **DFARS**: Defense acquisition regulations
- **ITAR**: Export control compliance (no export-controlled data in public artifacts)

**Compliance Status**: 98.75% across all frameworks

### Code Quality Requirements

- **Safety-Critical Code**: Must have 100% MC/DC coverage
- **Determinism**: All simulations must be reproducible with seed replay
- **Error Handling**: Comprehensive error handling required
- **Logging**: Use structured logging for observability
- **Validation**: Automated validation must pass before merge

## Architecture Patterns

### Module Organization

```
quasim/                 # Core quantum simulation runtime
  api/                  # Public API interfaces
  distributed/          # Multi-node orchestration
  dtwin/                # Digital twin integration
  opt/                  # Optimization algorithms
  qc/                   # Quantum circuit simulation
  hcal/                 # Hardware calibration

integrations/           # External system adapters
autonomous_systems_platform/  # Phase III RL optimization
compliance/             # Compliance validation tools
infra/                  # Kubernetes and infrastructure
```

### Naming Conventions

- **Python Modules**: lowercase_with_underscores
- **Classes**: PascalCase
- **Functions/Methods**: snake_case
- **Constants**: UPPER_CASE_WITH_UNDERSCORES
- **Private**: Prefix with single underscore (_private)

### Error Handling

- Use exceptions for error conditions
- Provide informative error messages
- Log errors with context
- Handle cleanup in finally blocks
- Document expected exceptions in docstrings

### Performance Considerations

- Profile before optimizing
- Document performance assumptions
- Use GPU acceleration where appropriate (cuQuantum)
- Maintain deterministic reproducibility (<1μs drift tolerance)
- Support FP8/FP16/FP32/FP64 precision modes

## Common Tasks

### Adding a New Feature

1. Check for existing similar functionality
2. Write tests first (TDD approach preferred)
3. Implement minimal code to pass tests
4. Document the feature
5. Update relevant README files
6. Add entry to CHANGELOG.md

### Adding Dependencies

1. Add to appropriate section in `pyproject.toml`
2. Verify license compatibility (Apache 2.0 compatible)
3. Check for security vulnerabilities
4. Update documentation if user-facing
5. Generate updated SBOM

### Fixing Bugs

1. Write a failing test that reproduces the bug
2. Fix the bug with minimal changes
3. Verify the test now passes
4. Check for similar bugs elsewhere
5. Update documentation if behavior changes

### Refactoring

1. Ensure comprehensive test coverage first
2. Make small, incremental changes
3. Run tests after each change
4. Keep commits small and focused
5. Document architectural decisions

## Automated Systems

The repository includes automated code quality systems:

- **Automated Code Review**: Scans and fixes issues on PRs
- **Auto-Fix**: Applies ruff, black, and isort fixes automatically
- **Auto-Merge**: Merges PRs that meet all criteria
- **Security Scanning**: Detects secrets and vulnerabilities
- **Compliance Validation**: Enforces compliance requirements

See [AUTO_MERGE_SYSTEM.md](../docs/AUTO_MERGE_SYSTEM.md) for details.

## Resources

- **Main Documentation**: [README.md](../README.md)
- **Quick Start**: [QUICKSTART.md](../QUICKSTART.md)
- **Contributing Guide**: [CONTRIBUTING.md](../CONTRIBUTING.md)
- **Compliance**: [COMPLIANCE_ASSESSMENT_INDEX.md](../COMPLIANCE_ASSESSMENT_INDEX.md)
- **Security**: [SECURITY.md](../SECURITY.md)
- **Code Quality**: [docs/CODE_QUALITY_SUMMARY.md](../docs/CODE_QUALITY_SUMMARY.md)

## Important Notes

- **Deterministic Reproducibility**: Critical for certification - maintain <1μs seed replay drift
- **GPU Compatibility**: Support NVIDIA (CUDA) and AMD (ROCm) backends
- **Multi-Cloud**: Design for EKS, GKE, and AKS compatibility
- **Observability First**: Include metrics, logging, and tracing in all new features
- **Zero Regression**: Maintain 100% passing tests; DO NOT break existing functionality
- **Certification Moat**: All changes must maintain DO-178C Level A compliance posture
