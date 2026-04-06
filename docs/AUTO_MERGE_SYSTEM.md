# Automated Code Review, Auto-Fix, and Auto-Merge System

## Overview

This repository implements a comprehensive automated code quality and pull request management system with three main components:

1. **Code Review & Auto-Fix**: Automatically reviews code and fixes quality issues
2. **PR Auto-Resolution**: Resolves issues in pull requests and prepares them for merge
3. **Auto-Merge**: Automatically merges pull requests that meet all criteria

## Components

### 1. Code Review & Auto-Fix Workflow

**File**: `.github/workflows/code-review-autofix.yml`

This workflow automatically:

- Scans code for security issues using Bandit
- Detects potential secrets in code
- Fixes linting errors with ruff
- Formats code with black
- Sorts imports with isort
- Commits and pushes fixes to the PR
- Comments on the PR with a summary of changes

**Triggers**:

- When a PR is opened, synchronized, or reopened
- When code is pushed to main or develop
- Manual trigger via workflow_dispatch

**Permissions Required**:

- `contents: write` - To commit fixes
- `pull-requests: write` - To comment on PRs
- `checks: write` - To update check status

### 2. PR Auto-Resolution Script

**File**: `scripts/pr_auto_resolver.py`

This Python script:

- Fetches open pull requests
- Audits PRs for issues (conflicts, CI failures, lint errors)
- Automatically fixes detected issues
- Attempts to merge PRs that meet criteria
- Labels PRs that need manual review

**Features**:

- Merge conflict resolution
- Lint error auto-fixing
- CI status checking
- Test execution
- Automatic branch deletion after merge

**Usage**:

```bash
# Set environment variables
export GITHUB_TOKEN="your_github_token"

# Run on all open PRs
python scripts/pr_auto_resolver.py

# Run on a specific PR
export PR_NUMBER=123
python scripts/pr_auto_resolver.py
```

### 3. Auto-Merge Workflow

**File**: `.github/workflows/auto-merge.yml`

This workflow automatically merges PRs that meet all criteria:

**Merge Criteria**:

- ✅ All CI checks must pass
- ✅ No merge conflicts
- ✅ Has the `auto-merge` or `automerge` label
- ✅ No blocking labels (`do-not-merge`, `wip`, `needs-work`, `needs-manual-review`)
- ✅ Not a draft PR
- ✅ No changes requested in reviews

**Triggers**:

- When a PR is opened, synchronized, reopened, or labeled
- When check suites complete
- When CI/Code Review workflows complete
- Manual trigger for specific PR

**Merge Strategies**:

- `hotfix/*` branches: rebase
- `copilot/*` branches: squash
- All others: squash

## Setup Instructions

### 1. Enable Workflows

All workflows are already configured and will run automatically. No additional setup required.

### 2. Configure Labels

Create these labels in your repository:

- `auto-merge` or `automerge` - Enables automatic merging
- `do-not-merge` - Prevents automatic merging
- `wip` - Work in progress, blocks merging
- `needs-work` - Requires changes before merge
- `needs-manual-review` - Requires human review

### 3. Set Branch Protection Rules (Optional)

For enhanced security, configure branch protection:

1. Go to Settings → Branches → Add rule
2. Branch name pattern: `main`
3. Enable:
   - Require pull request reviews before merging
   - Require status checks to pass before merging
   - Require conversation resolution before merging

### 4. Configure Secrets (if needed)

The workflows use `GITHUB_TOKEN` which is automatically provided by GitHub Actions.

## Usage Guide

### For Contributors

#### Enable Auto-Merge on Your PR

1. Create a pull request
2. Wait for CI checks to complete
3. Add the `auto-merge` label to your PR
4. The system will automatically merge once all criteria are met

#### Review Auto-Fix Changes

When you open a PR, the code review workflow will:

1. Automatically fix linting and formatting issues
2. Commit the fixes to your branch
3. Comment on the PR with details

Review these automated commits and ensure they don't change logic.

#### Handle Merge Conflicts

The PR auto-resolver will attempt to automatically resolve merge conflicts in:

- Documentation files (`.md`, `.txt`, `.rst`) - prefers incoming changes
- Configuration files - attempts smart merging

For complex conflicts, manual resolution is required.

### For Maintainers

#### Manually Trigger PR Resolution

1. Go to Actions → "Automated PR Resolution and Merge"
2. Click "Run workflow"
3. Enter PR number (optional, leave empty for all open PRs)
4. Click "Run workflow"

#### Manually Trigger Auto-Merge Evaluation

1. Go to Actions → "Auto-Merge PRs"
2. Click "Run workflow"
3. Enter PR number
4. Click "Run workflow"

#### Monitor Workflow Results

- Check workflow run logs for detailed information
- Review comments posted by the bot on PRs
- Check uploaded artifacts for detailed reports

## Safety Features

### 1. Multiple Validation Checks

- PRs are evaluated multiple times before merge
- CI must pass at evaluation time
- Mergeable status is checked immediately before merge

### 2. Blocking Mechanisms

- Draft PRs are never auto-merged
- PRs with blocking labels are skipped
- PRs with changes requested are not merged

### 3. Rollback Support

- All merges use standard Git merge commits (or squash)
- Easy to revert if issues are discovered
- Branch names are preserved in merge commits

### 4. Audit Trail

- All actions are logged in workflow runs
- Comments are posted on PRs for transparency
- JSON summaries are generated for monitoring

## Troubleshooting

### PR Not Auto-Merging

Check that:

1. PR has `auto-merge` label
2. All CI checks have passed
3. No merge conflicts exist
4. No blocking labels present
5. PR is not a draft

Check the workflow logs for specific reasons.

### Auto-Fix Not Working

Ensure:

1. Workflow has write permissions
2. No branch protection preventing commits
3. Check workflow logs for errors

### PR Marked "Needs Manual Review"

This happens when:

- Merge conflicts couldn't be auto-resolved
- Critical issues remain after max attempts (3)
- CI checks repeatedly fail

Manual intervention is required.

## Best Practices

1. **Use Auto-Merge Judiciously**: Only use for low-risk PRs (docs, small fixes)
2. **Review Auto-Fixes**: Always review automated commits before merging
3. **Monitor Workflows**: Regularly check workflow logs for issues
4. **Update Labels**: Keep blocking labels up to date
5. **Test Locally**: Run linters locally before pushing to reduce CI time

## Maintenance

### Updating Linting Rules

Edit `.ruff.toml` to configure ruff linting rules.

### Updating Auto-Fix Tools

Modify `.github/workflows/code-review-autofix.yml` to:

- Add new linting tools
- Change formatter settings
- Add security scanners

### Adjusting Merge Criteria

Edit `scripts/pr_auto_resolver.py` to change:

- Merge thresholds
- Merge strategies
- Required checks

## Security Considerations

1. **Limited Permissions**: Workflows use least-privilege permissions
2. **Token Security**: Uses built-in `GITHUB_TOKEN`, no external secrets
3. **Secret Scanning**: Automated checks for hardcoded secrets
4. **Security Scanning**: Bandit scans Python code for vulnerabilities
5. **Audit Logging**: All actions are logged and traceable

## Performance

- **Code Review**: ~2-3 minutes per run
- **PR Auto-Resolution**: ~5-10 minutes per PR (includes tests)
- **Auto-Merge Evaluation**: ~30 seconds per PR

## Monitoring and Metrics

Track these metrics:

- Number of PRs auto-merged vs. manually merged
- Number of issues auto-fixed
- Average time to merge
- Number of failed auto-merge attempts

Access via GitHub Actions insights and workflow artifacts.

## Support

For issues or questions:

1. Check workflow logs for detailed error messages
2. Review this documentation
3. Open an issue with the `question` label
4. Contact repository maintainers

## Future Enhancements

Potential improvements:

- Machine learning-based conflict resolution
- Intelligent test selection based on changed files
- Advanced security scanning integration
- Automated dependency updates
- Performance regression detection
- Automated changelog generation
