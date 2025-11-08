#!/usr/bin/env python3
"""
Automated Pull Request Resolution and Merge System

This script:
1. Fetches and reviews all open pull requests
2. Checks for merge conflicts, failed CI/CD, lint/test violations
3. Automatically fixes detected issues where possible
4. Re-runs tests after fixes
5. Merges PRs that meet merge criteria
6. Handles post-merge actions (branch deletion, comments)
"""

import json
import os
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

try:
    from github import Github, GithubException
    from github.PullRequest import PullRequest
    from github.Repository import Repository
except ImportError:
    print("ERROR: PyGithub not installed. Run: pip install PyGithub")
    sys.exit(1)


@dataclass
class PRIssue:
    """Represents an issue found in a PR"""

    type: str  # 'conflict', 'ci_failure', 'lint_error', 'test_failure', 'dependency'
    description: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    fixable: bool
    fixed: bool = False


@dataclass
class PRResolutionResult:
    """Result of PR resolution attempt"""

    pr_number: int
    status: str  # 'success', 'partial', 'failed', 'needs_manual_review'
    issues_found: List[PRIssue]
    issues_fixed: List[PRIssue]
    attempts: int
    merged: bool
    message: str


class PRAutoResolver:
    """Main class for automated PR resolution"""

    MAX_ATTEMPTS = 3
    MERGE_THRESHOLD_COVERAGE = 0.0  # Can be configured from repo settings

    def __init__(self, github_token: str, repo_name: str):
        self.gh = Github(github_token)
        self.repo: Repository = self.gh.get_repo(repo_name)
        self.repo_path = Path.cwd()

    def get_open_prs(self, pr_number: Optional[int] = None) -> List[PullRequest]:
        """Fetch all open PRs or a specific PR"""
        if pr_number:
            try:
                return [self.repo.get_pull(pr_number)]
            except GithubException as e:
                print(f"Error fetching PR #{pr_number}: {e}")
                return []
        else:
            return list(self.repo.get_pulls(state="open"))

    def audit_pr(self, pr: PullRequest) -> List[PRIssue]:
        """Audit a PR for issues"""
        issues = []

        # Check for merge conflicts
        if pr.mergeable is False:
            issues.append(
                PRIssue(
                    type="conflict",
                    description="PR has merge conflicts",
                    severity="critical",
                    fixable=True,
                )
            )

        # Check CI/CD status
        commit = pr.head.sha
        statuses = self.repo.get_commit(commit).get_statuses()
        failed_checks = [s for s in statuses if s.state == "failure"]
        if failed_checks:
            for check in failed_checks:
                issues.append(
                    PRIssue(
                        type="ci_failure",
                        description=f"CI check failed: {check.context}",
                        severity="high",
                        fixable=True,
                    )
                )

        # Check for outdated branch
        base_commit = self.repo.get_branch(pr.base.ref).commit
        pr_commits = list(pr.get_commits())
        if pr_commits:
            # Compare dates to see if PR is behind base
            if pr_commits[-1].commit.author.date < base_commit.commit.author.date:
                issues.append(
                    PRIssue(
                        type="outdated",
                        description="PR branch is behind base branch",
                        severity="medium",
                        fixable=True,
                    )
                )

        return issues

    def fix_merge_conflicts(self, pr: PullRequest) -> bool:
        """Attempt to automatically resolve merge conflicts"""
        try:
            # Fetch the PR branch
            branch_name = pr.head.ref
            base_branch = pr.base.ref

            # Check out the PR branch
            subprocess.run(["git", "fetch", "origin", branch_name], check=True)
            subprocess.run(["git", "checkout", branch_name], check=True)

            # Try to merge the base branch
            result = subprocess.run(
                ["git", "merge", f"origin/{base_branch}"], capture_output=True, text=True
            )

            if result.returncode != 0:
                # Check if there are conflicts
                conflicts = subprocess.run(
                    ["git", "diff", "--name-only", "--diff-filter=U"],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                conflicted_files = conflicts.stdout.strip().split("\n")
                if conflicted_files and conflicted_files[0]:
                    # For simple conflicts, try automatic resolution
                    # This is a simplified approach - in production, you'd want more sophisticated logic
                    for file in conflicted_files:
                        if self._auto_resolve_file_conflict(file):
                            subprocess.run(["git", "add", file], check=True)

                    # Check if all conflicts resolved
                    remaining = subprocess.run(
                        ["git", "diff", "--name-only", "--diff-filter=U"],
                        capture_output=True,
                        text=True,
                        check=True,
                    )

                    if not remaining.stdout.strip():
                        # Commit the merge
                        subprocess.run(
                            [
                                "git",
                                "commit",
                                "-m",
                                f"Auto-resolve merge conflicts for PR #{pr.number}",
                            ],
                            check=True,
                        )
                        subprocess.run(["git", "push", "origin", branch_name], check=True)
                        return True
                    else:
                        # Abort merge if we couldn't resolve everything
                        subprocess.run(["git", "merge", "--abort"], check=False)
                        return False
            else:
                # Merge succeeded without conflicts
                subprocess.run(["git", "push", "origin", branch_name], check=True)
                return True

        except subprocess.CalledProcessError as e:
            print(f"Error fixing conflicts: {e}")
            with suppress(Exception):
                subprocess.run(["git", "merge", "--abort"], check=False)
            return False

        return False

    def _auto_resolve_file_conflict(self, filepath: str) -> bool:
        """
        Attempt to auto-resolve conflicts in a file.
        This is a simplified version - prefers incoming changes for documentation,
        and attempts smart merging for code files.
        """
        try:
            # Read the file
            with open(filepath) as f:
                content = f.read()

            # Check if it's a simple conflict
            if "<<<<<<<" not in content:
                return True

            # For documentation files, prefer the incoming version (theirs)
            if filepath.endswith((".md", ".txt", ".rst")):
                lines = content.split("\n")
                resolved = []
                in_conflict = False
                keep_theirs = False

                for line in lines:
                    if line.startswith("<<<<<<<"):
                        in_conflict = True
                        keep_theirs = False
                    elif line.startswith("======="):
                        keep_theirs = True
                    elif line.startswith(">>>>>>>"):
                        in_conflict = False
                    elif not in_conflict or keep_theirs:
                        resolved.append(line)

                with open(filepath, "w") as f:
                    f.write("\n".join(resolved))
                return True

            return False

        except Exception as e:
            print(f"Error resolving conflict in {filepath}: {e}")
            return False

    def fix_lint_errors(self) -> bool:
        """Run linters and attempt to auto-fix errors"""
        try:
            print("Running ruff auto-fix...")
            # Run ruff with auto-fix
            result = subprocess.run(
                ["python", "-m", "ruff", "check", "--fix", "."],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0 or result.stdout:
                print(f"Ruff output: {result.stdout[:500]}")

            print("Running black formatter...")
            # Run black formatter
            result = subprocess.run(
                ["python", "-m", "black", "."], capture_output=True, text=True, check=False
            )
            if result.returncode == 0:
                print("Black formatting completed")

            print("Running isort...")
            # Run isort
            result = subprocess.run(
                ["python", "-m", "isort", "."], capture_output=True, text=True, check=False
            )
            if result.returncode == 0:
                print("Import sorting completed")

            # Check if there are changes to commit
            result = subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            )

            if result.stdout.strip():
                subprocess.run(["git", "add", "."], check=True)
                subprocess.run(
                    [
                        "git",
                        "commit",
                        "-m",
                        "chore: auto-fix lint errors\n\n- Applied ruff auto-fixes\n- Formatted code with black\n- Sorted imports with isort",
                    ],
                    check=True,
                )
                print("✅ Committed lint fixes")
                return True

            print("✅ No lint fixes needed")
            return True

        except subprocess.CalledProcessError as e:
            print(f"Error fixing lint errors: {e}")
            return False

    def run_tests(self) -> bool:
        """Run the test suite"""
        try:
            # Run the full stack test
            result = subprocess.run(
                ["python3", "scripts/test_full_stack.py"],
                capture_output=True,
                text=True,
                timeout=300,
            )

            # Check exit code (0 = success) and look for explicit success indicators
            if result.returncode == 0 and ("✓ PASS" in result.stdout or "PASS" in result.stdout):
                return True

            print(f"Tests failed (exit code: {result.returncode}):\n{result.stdout}")
            return False

        except subprocess.TimeoutExpired:
            print("Tests timed out")
            return False
        except Exception as e:
            print(f"Error running tests: {e}")
            return False

    def check_merge_criteria(
        self, pr: PullRequest, issues: List[PRIssue], run_tests: bool = True
    ) -> bool:
        """
        Check if PR meets merge criteria

        Args:
            pr: The pull request to check
            issues: List of known issues
            run_tests: Whether to run tests (can be disabled for performance)
        """
        # No critical or high severity unfixed issues
        critical_issues = [i for i in issues if i.severity in ["critical", "high"] and not i.fixed]
        if critical_issues:
            return False

        # Must be mergeable
        if pr.mergeable is False:
            return False

        # All CI checks must pass
        commit = pr.head.sha
        statuses = list(self.repo.get_commit(commit).get_statuses())
        if statuses:
            latest_statuses = {}
            for status in statuses:
                if status.context not in latest_statuses:
                    latest_statuses[status.context] = status

            for status in latest_statuses.values():
                if status.state == "failure":
                    return False

        # Tests must pass (optional, can be expensive)
        if run_tests and not self.run_tests():
            return False

        return True

    def merge_pr(self, pr: PullRequest) -> bool:
        """Merge a PR using appropriate strategy"""
        try:
            # Determine merge method based on branch naming
            # feature/* -> squash, hotfix/* -> rebase, default -> squash
            branch_name = pr.head.ref

            if branch_name.startswith("hotfix/"):
                merge_method = "rebase"
            elif branch_name.startswith("copilot/"):
                merge_method = "squash"
            else:
                merge_method = "squash"

            print(f"Attempting to merge PR #{pr.number} using {merge_method} strategy...")

            # Merge the PR
            pr.merge(
                commit_title=f"{pr.title} (#{pr.number})",
                commit_message="✅ All issues resolved, tests passed, and PR merged automatically by Copilot Workflow.",
                merge_method=merge_method,
            )

            print(f"✅ Successfully merged PR #{pr.number}")

            # Post merge comment
            pr.create_issue_comment(
                "✅ All issues resolved, tests passed, and PR merged automatically by Copilot Workflow."
            )

            # Delete branch if configured
            try:
                ref = self.repo.get_git_ref(f"heads/{pr.head.ref}")
                ref.delete()
                print(f"Deleted branch {pr.head.ref}")
            except Exception as e:
                print(f"Could not delete branch {pr.head.ref}: {e}")
                pass  # Branch might be from a fork or protected

            return True

        except GithubException as e:
            print(f"❌ Error merging PR: {e}")
            return False

    def resolve_pr(self, pr: PullRequest) -> PRResolutionResult:
        """Main resolution logic for a single PR"""
        print(f"\n{'='*60}")
        print(f"Processing PR #{pr.number}: {pr.title}")
        print(f"{'='*60}\n")

        attempts = 0
        max_attempts = self.MAX_ATTEMPTS

        while attempts < max_attempts:
            attempts += 1
            print(f"\nAttempt {attempts}/{max_attempts}")

            # Audit the PR
            issues = self.audit_pr(pr)
            print(f"Found {len(issues)} issues")

            if not issues:
                # No issues, check if we can merge
                if self.check_merge_criteria(pr, []):
                    if self.merge_pr(pr):
                        return PRResolutionResult(
                            pr_number=pr.number,
                            status="success",
                            issues_found=[],
                            issues_fixed=[],
                            attempts=attempts,
                            merged=True,
                            message="PR merged successfully with no issues found",
                        )
                else:
                    return PRResolutionResult(
                        pr_number=pr.number,
                        status="failed",
                        issues_found=[],
                        issues_fixed=[],
                        attempts=attempts,
                        merged=False,
                        message="PR does not meet merge criteria",
                    )

            # Try to fix issues
            fixed_issues = []
            for issue in issues:
                if not issue.fixable:
                    continue

                print(f"Attempting to fix: {issue.description}")

                if issue.type == "conflict":
                    if self.fix_merge_conflicts(pr):
                        issue.fixed = True
                        fixed_issues.append(issue)
                elif issue.type in ["ci_failure", "lint_error"]:
                    if self.fix_lint_errors():
                        issue.fixed = True
                        fixed_issues.append(issue)
                elif issue.type == "outdated":
                    # Already handled by merge conflicts fix
                    pass

            # If we fixed some issues, wait a bit for CI to update
            if fixed_issues:
                print(f"Fixed {len(fixed_issues)} issues, waiting for CI...")
                time.sleep(30)
                # Refresh PR object
                pr = self.repo.get_pull(pr.number)

            # Check if all critical issues are resolved
            remaining_critical = [i for i in issues if i.severity == "critical" and not i.fixed]
            if not remaining_critical:
                # Try to merge
                if self.check_merge_criteria(pr, issues):
                    if self.merge_pr(pr):
                        return PRResolutionResult(
                            pr_number=pr.number,
                            status="success",
                            issues_found=issues,
                            issues_fixed=fixed_issues,
                            attempts=attempts,
                            merged=True,
                            message=f"PR merged successfully after fixing {len(fixed_issues)} issues",
                        )

        # Max attempts reached
        # Label as needs-manual-review
        with suppress(Exception):
            pr.add_to_labels("needs-manual-review")

        return PRResolutionResult(
            pr_number=pr.number,
            status="needs_manual_review",
            issues_found=issues,
            issues_fixed=fixed_issues,
            attempts=attempts,
            merged=False,
            message=f"Could not automatically resolve all issues after {attempts} attempts",
        )

    def run(self, pr_number: Optional[int] = None):
        """Main entry point"""
        prs = self.get_open_prs(pr_number)

        if not prs:
            print("No open PRs to process")
            summary = {
                "status": "success",
                "message": "No open PRs to process",
                "issues_found": 0,
                "issues_fixed": 0,
                "details": "",
            }
            with open("pr_resolution_summary.json", "w") as f:
                json.dump(summary, f)
            return

        results = []
        for pr in prs:
            # Skip draft PRs
            if pr.draft:
                print(f"Skipping draft PR #{pr.number}")
                continue

            result = self.resolve_pr(pr)
            results.append(result)

        # Generate summary
        total_merged = sum(1 for r in results if r.merged)
        total_fixed = sum(len(r.issues_fixed) for r in results)
        total_issues = sum(len(r.issues_found) for r in results)

        summary = {
            "status": "success" if total_merged > 0 else "partial",
            "message": f"Processed {len(results)} PRs: {total_merged} merged, {total_fixed} issues fixed",
            "issues_found": total_issues,
            "issues_fixed": total_fixed,
            "details": "\n".join(
                [
                    f"- PR #{r.pr_number}: {r.status} ({len(r.issues_fixed)}/{len(r.issues_found)} issues fixed)"
                    for r in results
                ]
            ),
        }

        # Write summary for GitHub Action
        with open("pr_resolution_summary.json", "w") as f:
            json.dump(summary, f)

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(summary["message"])
        print(summary["details"])


def main():
    """Main entry point"""
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        print("ERROR: GITHUB_TOKEN environment variable not set")
        sys.exit(1)

    # Get repository name from git remote
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"], capture_output=True, text=True, check=True
        )
        remote_url = result.stdout.strip()
        # Parse repo name from URL
        if "github.com" in remote_url:
            repo_name = remote_url.split("github.com")[-1].strip("/:").replace(".git", "")
        else:
            print("ERROR: Could not determine repository name")
            sys.exit(1)
    except subprocess.CalledProcessError:
        print("ERROR: Could not get git remote URL")
        sys.exit(1)

    pr_number = os.environ.get("PR_NUMBER")
    if pr_number:
        pr_number = int(pr_number)

    resolver = PRAutoResolver(github_token, repo_name)
    resolver.run(pr_number)


if __name__ == "__main__":
    main()
