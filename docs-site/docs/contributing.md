# Contributing to QRATUM

Thank you for your interest in contributing to QRATUM! This guide helps you get started.

## Ways to Contribute

- **Bug reports** - Found a bug? Open an issue
- **Feature requests** - Have an idea? Start a discussion
- **Code contributions** - Fix bugs, add features
- **Documentation** - Improve docs, fix typos
- **Testing** - Add tests, improve coverage

## Getting Started

### 1. Fork and Clone

```bash
# Fork via GitHub UI, then:
git clone https://github.com/YOUR_USERNAME/QRATUM.git
cd QRATUM
git remote add upstream https://github.com/robertringler/QRATUM.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev,quantum]"
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

## Development Workflow

### Code Style

We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Format code
make fmt

# Check linting
make lint

# Or manually
ruff format .
ruff check .
```

### Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=quasim tests/

# Run specific tests
pytest tests/quantum/test_vqe.py
```

### Type Checking

```bash
mypy quasim/
```

## Commit Guidelines

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Code style (formatting) |
| `refactor` | Refactoring |
| `test` | Add/modify tests |
| `chore` | Build/tooling changes |

### Examples

```bash
# Feature
git commit -m "feat(quantum): add QAOA warm-start initialization"

# Bug fix
git commit -m "fix(vqe): correct energy calculation for bond length > 2.0"

# Documentation
git commit -m "docs: update VQE tutorial with convergence tips"
```

## Pull Request Process

### 1. Update Your Branch

```bash
git fetch upstream
git rebase upstream/main
```

### 2. Run Checks

```bash
# Format and lint
make fmt
make lint

# Run tests
make test

# Check coverage
pytest --cov=quasim --cov-report=term-missing tests/
```

### 3. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a PR via GitHub.

### 4. PR Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] New code has tests
- [ ] Documentation updated
- [ ] Commit messages follow convention
- [ ] PR description explains changes

## Code Review

All PRs require review. Reviewers check:

- **Correctness** - Does it work?
- **Tests** - Are changes tested?
- **Style** - Does it follow guidelines?
- **Documentation** - Is it documented?
- **Security** - Are there security concerns?

## Security Contributions

For security vulnerabilities:

**DO NOT** open public issues.

Instead, email maintainers privately. See [SECURITY.md](https://github.com/robertringler/QRATUM/blob/main/SECURITY.md).

## Documentation Contributions

### Building Docs Locally

```bash
cd docs-site
pip install mkdocs-material
mkdocs serve
```

Then open http://localhost:8000

### Documentation Style

- Use clear, concise language
- Include code examples
- Add links to related topics
- Use admonitions for warnings/tips

```markdown
!!! tip "Pro Tip"
    This is a helpful tip.

!!! warning "Caution"
    Be careful with this.
```

## What We Accept

✅ **We welcome:**

- Bug fixes with tests
- Features aligned with roadmap
- Documentation improvements
- Performance optimizations (with benchmarks)
- Test coverage improvements

❌ **We don't accept:**

- Unsubstantiated quantum claims
- Code without tests
- Features not in roadmap (discuss first)
- Major refactoring without prior discussion

## Recognition

Contributors are recognized in:

- GitHub contributor list
- Release notes (for significant contributions)
- CONTRIBUTORS.md (for major contributors)

## Questions?

- **General questions** - Open a Discussion
- **Bug reports** - Open an Issue
- **Feature ideas** - Open a Discussion first

Thank you for contributing to QRATUM! 🎉
