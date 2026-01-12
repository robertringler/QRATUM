# Automated Pull Request Resolution and Merge System

## Overview

The automated PR resolution system audits, repairs, and merges pull requests automatically, ensuring code quality and reducing manual review overhead.

## Features

### 1. Fetch & Review

- Lists all open pull requests
- For each PR, checks for:
  - Merge conflicts
  - Failed CI/CD pipelines
  - Lint/test violations
  - Dependency inconsistencies
  - Deprecated APIs
  - Type mismatches
  - Style violations (PEP 8, etc.)

### 2. Automatic Fixing

- **Merge Conflicts**: Intelligently resolves conflicts preferring main-branch stability
- **Lint Errors**: Applies automatic code fixes using ruff, black, and isort
- **Test Errors**: Identifies and attempts to fix test failures
- **Documentation**: Updates changelogs and version numbers as needed
- **Build Consistency**: Ensures Dockerfiles and manifests remain consistent

### 3. Verification

- Re-runs full test suite after each fix
- Confirms CI passes at 100%
- Performs security linting
- Validates code coverage meets threshold

### 4. Merge Policy

Merges only when:

- All tests pass
- No conflicts remain
- Code coverage ≥ threshold (configurable)
- Versioning and changelog entries updated

Merge strategies:

- **Squash and merge** for feature branches (default)
- **Rebase and merge** for hotfix branches

### 5. Post-Merge Actions

- Deletes merged branches automatically
- Triggers deployment workflows if configured
- Posts summary comment: "✅ All issues resolved, tests passed, and PR merged automatically by Copilot Workflow."

## Safety Features

### Manual Review Trigger

If a PR cannot be merged cleanly after 3 automated attempts, the system:

- Labels it `needs-manual-review`
- Stops processing further automatic merges
- Comments with details of unresolved issues

## Usage

### Automatic Trigger

The system runs automatically:

- On every PR open/update event
- Every 6 hours via cron schedule
- When manually triggered via workflow dispatch

### Manual Trigger

To process a specific PR:

```bash
# Via GitHub CLI
gh workflow run pr-auto-resolution.yml -f pr_number=123

# Via GitHub UI
Go to Actions → Automated PR Resolution and Merge → Run workflow
```

### Local Testing

To test the resolution system locally:

```bash
export GITHUB_TOKEN="your_token_here"
python scripts/pr_auto_resolver.py
```

To process a specific PR:

```bash
export GITHUB_TOKEN="your_token_here"
export PR_NUMBER=123
python scripts/pr_auto_resolver.py
```

## Configuration

### Environment Variables

- `GITHUB_TOKEN`: GitHub API token (required)
- `PR_NUMBER`: Specific PR to process (optional)
- `MERGE_THRESHOLD_COVERAGE`: Minimum code coverage required (default: 0.0)

### Merge Strategies

The system determines merge strategy based on branch naming:

- Branches starting with `hotfix/` → Rebase and merge
- All other branches → Squash and merge

## Architecture

### Components

#### 1. GitHub Actions Workflow (`.github/workflows/pr-auto-resolution.yml`)

- Triggers on PR events and schedule
- Sets up Python environment
- Installs dependencies
- Runs the PR auto-resolver script
- Posts summary comments

#### 2. PR Auto Resolver Script (`scripts/pr_auto_resolver.py`)

Main resolution logic:

- `PRAutoResolver`: Main class coordinating the resolution
- `audit_pr()`: Identifies issues in a PR
- `fix_merge_conflicts()`: Resolves merge conflicts automatically
- `fix_lint_errors()`: Applies automatic code formatting and fixes
- `run_tests()`: Executes test suite
- `check_merge_criteria()`: Validates merge readiness
- `merge_pr()`: Performs the merge with appropriate strategy
- `resolve_pr()`: Orchestrates the full resolution process

### Issue Types

The system handles:

- `conflict`: Merge conflicts with base branch
- `ci_failure`: Failed CI/CD checks
- `lint_error`: Code style and quality issues
- `test_failure`: Failing unit or integration tests
- `dependency`: Dependency version conflicts
- `outdated`: Branch behind base branch

### Issue Severity Levels

- `critical`: Must be fixed before merge (conflicts, security)
- `high`: Should be fixed (failing tests, broken builds)
- `medium`: Good to fix (outdated branch, minor lint issues)
- `low`: Nice to have (documentation, style preferences)

## Workflow Diagram

```
┌─────────────────────────────────────────┐
│  PR Opened/Updated or Scheduled Run     │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Fetch Open PRs (or specific PR)        │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  For Each PR: Audit for Issues          │
│  - Merge conflicts?                     │
│  - Failed CI checks?                    │
│  - Lint/test errors?                    │
│  - Outdated branch?                     │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Issues Found?                          │
└──────┬────────────────────────┬─────────┘
       │ No                     │ Yes
       │                        │
       │                        ▼
       │         ┌─────────────────────────┐
       │         │  Attempt Auto-Fix        │
       │         │  (max 3 attempts)        │
       │         └─────────┬───────────────┘
       │                   │
       │                   ▼
       │         ┌─────────────────────────┐
       │         │  Run Tests & CI          │
       │         └─────────┬───────────────┘
       │                   │
       │                   ▼
       │         ┌─────────────────────────┐
       │         │  All Critical Issues     │
       │         │  Fixed?                  │
       │         └──┬──────────────────┬───┘
       │            │ Yes              │ No
       │            │                  │
       │            │                  ▼
       │            │      ┌──────────────────────┐
       │            │      │  Label as             │
       │            │      │  needs-manual-review  │
       │            │      └──────────────────────┘
       │            │
       ▼            ▼
┌─────────────────────────────────────────┐
│  Check Merge Criteria                   │
│  - All tests pass?                      │
│  - CI checks green?                     │
│  - Coverage threshold met?              │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Criteria Met?                          │
└──────┬────────────────────────┬─────────┘
       │ Yes                    │ No
       │                        │
       ▼                        ▼
┌──────────────────┐  ┌──────────────────┐
│  Merge PR        │  │  Skip & Continue  │
│  - Squash/Rebase │  │                   │
│  - Delete Branch │  │                   │
│  - Post Comment  │  │                   │
└──────────────────┘  └──────────────────┘
```

## Examples

### Successful Automatic Merge

```
Processing PR #123: Add new feature X
Found 2 issues
- Merge conflict in config.yaml (fixable)
- Lint error in module.py (fixable)

Attempting to fix: Merge conflict in config.yaml
✓ Fixed

Attempting to fix: Lint error in module.py
✓ Fixed

Running tests... ✓ All tests passed

Merging PR #123 using squash strategy
✓ PR merged successfully
✓ Branch deleted
✓ Comment posted
```

### Manual Review Required

```
Processing PR #456: Refactor core module
Found 3 issues
- Merge conflict in core.py (fixable)
- Test failure in test_core.py (fixable)
- Breaking API change (not auto-fixable)

Attempt 1/3
✗ Could not resolve test failure

Attempt 2/3
✗ Could not resolve test failure

Attempt 3/3
✗ Could not resolve test failure

✗ Max attempts reached
✓ Labeled as 'needs-manual-review'
```

## Best Practices

### For Contributors

1. Keep PRs focused and small
2. Ensure your branch is up to date before creating PR
3. Run local tests before pushing
4. Follow code style guidelines (automatic formatting available)

### For Maintainers

1. Review `needs-manual-review` PRs promptly
2. Monitor the automated merge logs
3. Adjust merge criteria as needed
4. Update the auto-fix logic for common patterns

## Troubleshooting

### PR Not Being Processed

- Check if PR is marked as draft
- Verify CI checks are completing
- Look for the `needs-manual-review` label

### Auto-Fix Not Working

- Review the workflow logs in GitHub Actions
- Check if the issue type is supported for auto-fix
- Verify GitHub token has sufficient permissions

### Merge Failing

- Confirm all merge criteria are met
- Check branch protection rules
- Verify no required reviews are missing

## Security Considerations

- GitHub token requires `contents: write` and `pull-requests: write` permissions
- Auto-fixes are committed using the GitHub Actions bot identity
- All changes are logged and auditable
- Manual review required for security-sensitive changes

## Future Enhancements

Potential improvements:

- Integration with additional CI/CD platforms
- More sophisticated conflict resolution strategies
- Machine learning for predicting PR quality
- Automatic dependency updates
- Integration with code review AI tools
- Custom merge criteria per repository
- Automatic rollback on post-merge failures

## Support

For issues or questions:

1. Check the [GitHub Actions logs](../../actions)
2. Review the [PR auto-resolver script](../../scripts/pr_auto_resolver.py)
3. Create an issue with the `automation` label
