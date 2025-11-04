# PR Compliance Workflows

This directory contains two compliance workflows that automatically run on pull requests to ensure code quality and security standards.

## Workflows

### 1. PR Compliance (`pr-compliance.yml`)

**Purpose**: Ensures general code quality and compliance standards.

**Checks performed**:
- **Code Quality**: Ruff linting, Black formatting, isort import ordering, mypy type checking
- **Documentation**: YAML and Markdown file validation
- **Workflow Validation**: GitHub Actions workflow syntax and permissions compliance

**Trigger**: Runs automatically on pull requests (opened, synchronize, reopened) targeting `main` or `master` branches.

**Watch progress**:
```bash
gh run watch --exit-status --workflow "PR Compliance"
```

### 2. PR Defense Compliance (`pr-defense-compliance.yml`)

**Purpose**: Ensures security and defense standards are met.

**Checks performed**:
- **Security Scanning**: Bandit and pip-audit for vulnerabilities
- **Secret Detection**: Scans for exposed secrets, credentials, and API keys
- **Dependency Review**: Reviews dependencies for known vulnerabilities
- **Permissions Check**: Verifies no elevated permissions or unsafe patterns
- **Dockerfile Security**: Scans Dockerfiles for security best practices

**Trigger**: Runs automatically on pull requests (opened, synchronize, reopened) targeting `main` or `master` branches.

**Watch progress**:
```bash
gh run watch --exit-status --workflow "PR Defense Compliance"
```

## Usage Example

### Creating a PR with Compliance Checks

```bash
# Ensure you're on latest default branch
git fetch origin
git checkout main
git pull --ff-only

# Create a short-lived branch for triggering workflows
BR="ci/trigger-$(date -u +%Y%m%d-%H%M%S)"
git checkout -b "$BR"

# Add a tiny change to ensure workflows run
date -u +"Triggered: %Y-%m-%d %H:%M:%S UTC" > .ci-trigger
git add .ci-trigger
git commit -m "chore(ci): trigger compliance workflows"

# Push branch
git push -u origin "$BR"

# Create PR
gh pr create --fill --base main --head "$BR"

# Watch PR Compliance workflow
gh run watch --exit-status --workflow "PR Compliance"

# Watch PR Defense Compliance workflow (in a separate terminal if desired)
gh run watch --exit-status --workflow "PR Defense Compliance"
```

## Workflow Features

### Permissions
Both workflows use least-privilege permissions:
- `contents: read` - Read repository content
- `pull-requests: read` - Read PR information
- `security-events: write` - Write security events (Defense Compliance only)
- `checks: write` - Write check results (PR Compliance only)

### Error Handling
Most checks use `continue-on-error: true` to ensure all checks run even if one fails, providing complete feedback to developers.

### Reporting
Both workflows generate summaries that appear in:
- GitHub Actions workflow summary
- PR checks status
- Job artifacts (for security reports)

## Customization

To modify the compliance requirements:
1. Edit the respective workflow file in `.github/workflows/`
2. Adjust checks, add new jobs, or modify existing ones
3. Test changes by creating a PR
4. Verify workflows run as expected

## Troubleshooting

### Workflow not triggering
- Ensure the PR targets `main` or `master` branch
- Check that workflow files are in `.github/workflows/` directory
- Verify YAML syntax is valid

### Check failures
- Review the specific job that failed in the Actions tab
- Check the job logs for detailed error messages
- Address the issues and push new commits to re-trigger checks

### Permission issues
- Verify repository settings allow Actions to run
- Ensure required permissions are granted in workflow files
- Check organization/repository security policies

## Support

For issues or questions about these workflows:
1. Check workflow run logs in the Actions tab
2. Review this documentation
3. Consult GitHub Actions documentation
4. Contact the repository maintainers
