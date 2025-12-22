# Contributing to QRATUM

Thank you for your interest in contributing to QRATUM-ASI! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Report Issues](#how-to-report-issues)
- [How to Submit Code](#how-to-submit-code)
- [Code Style Requirements](#code-style-requirements)
- [Testing Requirements](#testing-requirements)
- [Priority Contribution Areas](#priority-contribution-areas)
- [Review Process](#review-process)
- [Contact for Questions](#contact-for-questions)

---

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to <conduct@qratum.io>.

**In Summary:**

- Be respectful and constructive in all interactions
- Welcome diverse perspectives and experiences
- Focus on what is best for the community and project
- Show empathy towards other community members

---

## How to Report Issues

### Bug Reports

When reporting bugs, please include:

1. **Environment**: OS, Python version, QRATUM version
2. **Steps to Reproduce**: Minimal code example that demonstrates the issue
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Logs/Screenshots**: Any relevant error messages or outputs
6. **Impact**: How critical is this issue to your use case?

**Template:**

```markdown
## Environment
- OS: Ubuntu 22.04
- Python: 3.10.5
- QRATUM Version: 0.1.0-alpha

## Steps to Reproduce
1. Execute contract with payload: `{"action": "test"}`
2. Observe Merkle chain validation
3. Error occurs at rollback attempt

## Expected Behavior
Contract should rollback to previous state

## Actual Behavior
Error: "Merkle chain integrity violation"

## Logs
[Paste relevant logs here]

## Impact
Blocking production deployment
```

### Feature Requests

When requesting features, please include:

1. **Use Case**: What problem are you trying to solve?
2. **Proposed Solution**: How would you implement this?
3. **Alternatives Considered**: What other approaches did you evaluate?
4. **Additional Context**: Any relevant background or examples

---

## How to Submit Code

### Fork, Branch, Test, PR Workflow

1. **Fork the Repository**

   ```bash
   # Fork via GitHub UI, then clone your fork
   git clone https://github.com/YOUR_USERNAME/QRATUM.git
   cd QRATUM
   ```

2. **Add Upstream Remote**

   ```bash
   git remote add upstream https://github.com/robertringler/QRATUM.git
   ```

3. **Create a Feature Branch**

   ```bash
   git checkout -b feature/your-feature-name
   
   # Branch naming conventions:
   # feature/add-juris-contract-analyzer
   # bugfix/fix-merkle-chain-validation
   # docs/update-architecture-guide
   # refactor/simplify-qradle-core
   ```

4. **Make Your Changes**
   - Follow code style requirements (see below)
   - Write tests for new functionality
   - Update documentation as needed
   - Ensure all tests pass locally

5. **Commit Your Changes**

   ```bash
   git add .
   git commit -m "feat: add contract analysis for JURIS vertical"
   
   # Commit message format:
   # <type>: <subject>
   #
   # Types: feat, fix, docs, refactor, test, chore
   # Subject: imperative mood, lowercase, no period, <72 chars
   ```

6. **Push to Your Fork**

   ```bash
   git push origin feature/your-feature-name
   ```

7. **Submit a Pull Request**
   - Go to GitHub and create a PR from your fork
   - Fill out the PR template completely
   - Link any related issues
   - Request review from maintainers

### Development Setup

**Prerequisites:**

- Python 3.10+
- Git 2.30+
- Docker (optional, for containerized testing)

**Local Environment:**

```bash
# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Or install all at once
pip install -r requirements.txt -r requirements-dev.txt

# Verify installation
python -c "import qradle; print(qradle.__version__)"
```

**Running Tests Locally:**

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=qradle --cov=qratum --cov=qratum_asi tests/

# Run specific test file
pytest tests/test_qradle_contracts.py

# Run specific test
pytest tests/test_qradle_contracts.py::test_contract_execution

# Run with verbose output
pytest -v tests/
```

---

## Code Style Requirements

### Python Standards

QRATUM follows strict Python standards to ensure code quality, maintainability, and certification readiness.

#### PEP 8 Compliance

All code must follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines:

- **Indentation**: 4 spaces (no tabs)
- **Line Length**: 88 characters (Black default)
- **Imports**: Sorted with isort, grouped (stdlib, third-party, local)
- **Naming**:
  - `snake_case` for functions, variables, modules
  - `PascalCase` for classes
  - `UPPER_CASE` for constants
  - `_leading_underscore` for internal/private

#### Type Hints (Required)

All functions must have type hints for parameters and return values:

```python
# ✅ Good
def execute_contract(
    contract_id: str,
    payload: Dict[str, Any],
    safety_level: SafetyLevel = SafetyLevel.ROUTINE
) -> ContractResult:
    """Execute a QRADLE contract with the given payload.
    
    Args:
        contract_id: Unique identifier for the contract
        payload: Input data for contract execution
        safety_level: Safety classification for authorization
        
    Returns:
        ContractResult containing output and merkle proof
        
    Raises:
        ContractExecutionError: If execution fails
        AuthorizationError: If safety level requires approval
    """
    pass

# ❌ Bad (no type hints)
def execute_contract(contract_id, payload, safety_level=None):
    pass
```

#### Code Formatting: Black

All code must be formatted with [Black](https://black.readthedocs.io/):

```bash
# Format all Python files
black .

# Check formatting without changes
black --check .

# Format specific file
black qradle/contracts.py
```

**Configuration** (in `pyproject.toml`):

```toml
[tool.black]
line-length = 88
target-version = ['py310']
```

#### Import Sorting: isort

All imports must be sorted with [isort](https://pycqa.github.io/isort/):

```bash
# Sort imports
isort .

# Check without changes
isort --check .
```

**Configuration** (in `pyproject.toml`):

```toml
[tool.isort]
profile = "black"
line_length = 88
```

#### Linting: Ruff

Code must pass [Ruff](https://beta.ruff.rs/docs/) linting:

```bash
# Run linter
ruff check .

# Auto-fix issues
ruff check --fix .
```

#### Type Checking: mypy

Code must pass [mypy](http://mypy-lang.org/) type checking:

```bash
# Run type checker
mypy qradle/ qratum/ qratum_asi/

# Strict mode (recommended)
mypy --strict qradle/
```

### Pre-Commit Checks

Install pre-commit hooks to automatically check code before committing:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

**Configuration** (`.pre-commit-config.yaml`):

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.0.270
    hooks:
      - id: ruff
```

---

## Testing Requirements

### Coverage Requirements

All contributions must maintain **>80% test coverage**:

```bash
# Run tests with coverage
pytest --cov=qradle --cov=qratum --cov=qratum_asi --cov-report=term-missing tests/

# Generate HTML coverage report
pytest --cov=qradle --cov=qratum --cov=qratum_asi --cov-report=html tests/
open htmlcov/index.html
```

### Deterministic Tests

All tests must be **deterministic** (same inputs = same outputs):

```python
# ✅ Good: Deterministic test with fixed seed
def test_contract_execution_deterministic():
    seed = 12345
    random.seed(seed)
    np.random.seed(seed)
    
    result1 = execute_contract("test-contract", {"seed": seed})
    
    random.seed(seed)
    np.random.seed(seed)
    
    result2 = execute_contract("test-contract", {"seed": seed})
    
    assert result1 == result2  # Must be identical

# ❌ Bad: Non-deterministic test
def test_contract_execution():
    result = execute_contract("test-contract", {})  # Random seed
    assert result.status == "success"  # Flaky!
```

### Test Organization

```
tests/
├── unit/                      # Fast, isolated unit tests
│   ├── test_qradle_contracts.py
│   ├── test_qradle_merkle.py
│   └── test_qratum_verticals.py
├── integration/               # Multi-component integration tests
│   ├── test_contract_lifecycle.py
│   └── test_multi_vertical_reasoning.py
├── system/                    # Full system tests
│   └── test_end_to_end.py
└── fixtures/                  # Shared test fixtures
    ├── contracts.py
    └── test_data.json
```

### Test Naming Conventions

```python
# Pattern: test_<function_name>_<scenario>_<expected_outcome>

def test_execute_contract_valid_payload_returns_success():
    """Test that valid payload execution returns success status."""
    pass

def test_execute_contract_invalid_payload_raises_error():
    """Test that invalid payload raises ContractExecutionError."""
    pass

def test_rollback_contract_to_checkpoint_restores_state():
    """Test that rollback restores system to checkpoint state."""
    pass
```

---

## Priority Contribution Areas

We welcome contributions in all areas, but these are particularly valuable:

| Area | Priority | Description | Examples |
|------|----------|-------------|----------|
| **Adapters** | 🔴 HIGH | Integrate QRATUM with enterprise systems | SAP, Salesforce, Epic EMR, Bloomberg Terminal |
| **Verticals** | 🔴 HIGH | Expand capabilities in 14 domains | JURIS legal analysis, VITRA diagnosis support |
| **Verification** | 🟠 MEDIUM | Formal methods, proofs, certification | DO-178C artifacts, formal verification tools |
| **Safety** | 🔴 HIGH | ASI safety analysis, red teaming | Threat modeling, attack simulations |
| **Documentation** | 🟠 MEDIUM | Examples, tutorials, use cases | Industry-specific guides, deployment tutorials |
| **Testing** | 🟢 LOW | Test coverage, edge cases | Unit tests, integration tests, property tests |
| **Performance** | 🟢 LOW | Optimization, profiling | Query optimization, caching strategies |

### Adapter Development

**Needed Enterprise System Adapters:**

1. **Healthcare**: Epic EMR, Cerner, Meditech, Allscripts
2. **Finance**: Bloomberg Terminal, Refinitiv Eikon, FIS, Finastra
3. **Legal**: LexisNexis, Westlaw, Casetext, PACER
4. **ERP**: SAP S/4HANA, Oracle ERP Cloud, Microsoft Dynamics 365
5. **CRM**: Salesforce, HubSpot, Microsoft Dynamics CRM
6. **Government**: DISA systems, GSA platforms, FedRAMP-certified services

**Adapter Template:**

```python
from qratum.adapters.base import BaseAdapter

class EpicEMRAdapter(BaseAdapter):
    """Adapter for Epic Electronic Medical Record system."""
    
    def __init__(self, fhir_endpoint: str, credentials: Dict[str, str]):
        super().__init__(name="Epic EMR", version="1.0.0")
        self.endpoint = fhir_endpoint
        self.credentials = credentials
        
    def connect(self) -> bool:
        """Establish connection to Epic FHIR API."""
        pass
        
    def fetch_patient_data(self, patient_id: str) -> Dict[str, Any]:
        """Fetch patient data via FHIR API."""
        pass
        
    def to_qratum_format(self, epic_data: Dict) -> Dict:
        """Convert Epic data to QRATUM internal format."""
        pass
```

### Vertical Development

**Priority Verticals for Expansion:**

1. **JURIS (Legal)**: Contract clause extraction, risk scoring, regulatory compliance
2. **VITRA (Healthcare)**: Differential diagnosis, drug interaction checking, clinical pathways
3. **ECORA (Climate)**: Carbon footprint calculation, sustainability scoring, climate risk assessment
4. **CAPRA (Finance)**: Credit risk modeling, fraud detection, portfolio optimization
5. **SENTRA (Security)**: Threat intelligence, vulnerability scanning, incident response

**Vertical Contribution Checklist:**

- [ ] Domain-specific knowledge graphs
- [ ] Reasoning algorithms (deductive, inductive, abductive)
- [ ] Integration with existing tools (APIs, databases, file formats)
- [ ] Validation against domain expert benchmarks
- [ ] Documentation with real-world examples
- [ ] Unit tests with >80% coverage
- [ ] Type hints for all functions

---

## Review Process

### Maintainer Review

All pull requests are reviewed by project maintainers:

1. **Automated Checks**: CI/CD runs tests, linting, type checking
2. **Code Review**: Maintainer reviews code quality, architecture, style
3. **Testing**: Maintainer verifies tests are comprehensive and deterministic
4. **Documentation**: Maintainer checks that docs are updated as needed
5. **Approval**: At least one maintainer must approve

**Timeline:**

- Initial review: Within **3 business days**
- Revisions: Within **2 business days** of updates
- Merge: Within **1 business day** of final approval

### Safety-Critical Code (Requires Two Approvals)

Code affecting safety-critical components requires **two maintainer approvals**:

**Safety-Critical Components:**

- QRADLE contract execution engine
- Merkle chain integrity checks
- Authorization and permission systems
- Rollback and recovery mechanisms
- ASI safety constraints (8 Fatal Invariants)
- Q-EVOLVE self-improvement boundaries

**Additional Requirements:**

- Formal verification (where applicable)
- Red team security review
- Determinism validation
- Regression test suite execution
- Certification artifact updates (DO-178C, CMMC)

### Review Criteria

**Code Quality:**

- [ ] Follows PEP 8 and project style guidelines
- [ ] All functions have type hints
- [ ] Code is formatted with Black and isort
- [ ] Passes Ruff linting and mypy type checking
- [ ] No commented-out code or debug statements
- [ ] Clear, descriptive variable and function names

**Testing:**

- [ ] New functionality has unit tests
- [ ] Tests are deterministic (fixed seeds, no randomness)
- [ ] Test coverage >80% for modified code
- [ ] Integration tests for multi-component features
- [ ] Edge cases and error conditions tested

**Documentation:**

- [ ] Docstrings for all public functions/classes
- [ ] README updated if public API changes
- [ ] Architecture docs updated if design changes
- [ ] CHANGELOG updated with description of changes
- [ ] Examples provided for new features

**Safety & Security:**

- [ ] No secrets or credentials in code
- [ ] Input validation for all external data
- [ ] Error handling doesn't leak sensitive information
- [ ] Authorization checks for privileged operations
- [ ] Determinism maintained (if applicable)
- [ ] Auditability preserved (events emitted)

---

## Contact for Questions

**General Questions:**

- GitHub Discussions: [QRATUM Discussions](https://github.com/robertringler/QRATUM/discussions)
- Email: <contribute@qratum.io>

**Technical Questions:**

- Architecture: <architecture@qratum.io>
- Safety & Security: <security@qratum.io>
- Vertical Domains: <verticals@qratum.io>

**Maintainers:**

- Robert Ringler (Project Lead): <robert@qratum.io>

---

## Recognition

We value all contributions! Contributors will be:

- Listed in `CONTRIBUTORS.md`
- Mentioned in release notes for significant contributions
- Invited to join the QRATUM Contributors program (for regular contributors)
- Eligible for speaking opportunities at QRATUM conferences/events

---

## License

By contributing to QRATUM, you agree that your contributions will be licensed under the Apache License 2.0, consistent with the project's overall license.

See [LICENSE](LICENSE) for full text.

---

Thank you for contributing to QRATUM-ASI! Together, we're building the infrastructure for safe, sovereign, and auditable superintelligence.

*Every contribution brings us closer to AI systems we can trust.*

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
