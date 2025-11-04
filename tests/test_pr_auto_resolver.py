#!/usr/bin/env python3
"""
Unit tests for PR auto-resolver
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from pr_auto_resolver import PRAutoResolver, PRIssue, PRResolutionResult


class TestPRIssue(unittest.TestCase):
    """Test PRIssue dataclass"""

    def test_create_issue(self):
        """Test creating a PR issue"""
        issue = PRIssue(
            type='conflict',
            description='Merge conflict in file.py',
            severity='critical',
            fixable=True
        )

        self.assertEqual(issue.type, 'conflict')
        self.assertEqual(issue.severity, 'critical')
        self.assertTrue(issue.fixable)
        self.assertFalse(issue.fixed)


class TestPRAutoResolver(unittest.TestCase):
    """Test PRAutoResolver class"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_github = Mock()
        self.mock_repo = Mock()

        with patch('pr_auto_resolver.Github') as mock_gh_class:
            mock_gh_class.return_value = self.mock_github
            self.mock_github.get_repo.return_value = self.mock_repo
            self.resolver = PRAutoResolver('fake_token', 'owner/repo')

    def test_initialization(self):
        """Test PRAutoResolver initialization"""
        self.assertIsNotNone(self.resolver)
        self.assertEqual(self.resolver.MAX_ATTEMPTS, 3)
        self.assertEqual(self.resolver.MERGE_THRESHOLD_COVERAGE, 0.0)

    def test_get_open_prs_all(self):
        """Test getting all open PRs"""
        mock_pr1 = Mock()
        mock_pr2 = Mock()
        self.mock_repo.get_pulls.return_value = [mock_pr1, mock_pr2]

        prs = self.resolver.get_open_prs()

        self.assertEqual(len(prs), 2)
        self.mock_repo.get_pulls.assert_called_once_with(state='open')

    def test_get_open_prs_specific(self):
        """Test getting a specific PR"""
        mock_pr = Mock()
        self.mock_repo.get_pull.return_value = mock_pr

        prs = self.resolver.get_open_prs(pr_number=123)

        self.assertEqual(len(prs), 1)
        self.assertEqual(prs[0], mock_pr)
        self.mock_repo.get_pull.assert_called_once_with(123)

    def test_audit_pr_no_issues(self):
        """Test auditing a PR with no issues"""
        mock_pr = Mock()
        mock_pr.mergeable = True
        mock_pr.head.sha = 'abc123'
        mock_pr.base.ref = 'main'
        mock_pr.get_commits.return_value = []

        mock_commit = Mock()
        mock_commit.get_statuses.return_value = []
        self.mock_repo.get_commit.return_value = mock_commit

        mock_branch = Mock()
        mock_branch.commit = Mock()
        self.mock_repo.get_branch.return_value = mock_branch

        issues = self.resolver.audit_pr(mock_pr)

        self.assertEqual(len(issues), 0)

    def test_audit_pr_with_conflict(self):
        """Test auditing a PR with merge conflict"""
        mock_pr = Mock()
        mock_pr.mergeable = False
        mock_pr.head.sha = 'abc123'
        mock_pr.base.ref = 'main'
        mock_pr.get_commits.return_value = []

        mock_commit = Mock()
        mock_commit.get_statuses.return_value = []
        self.mock_repo.get_commit.return_value = mock_commit

        mock_branch = Mock()
        mock_branch.commit = Mock()
        self.mock_repo.get_branch.return_value = mock_branch

        issues = self.resolver.audit_pr(mock_pr)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].type, 'conflict')
        self.assertEqual(issues[0].severity, 'critical')
        self.assertTrue(issues[0].fixable)

    def test_audit_pr_with_ci_failure(self):
        """Test auditing a PR with CI failure"""
        mock_pr = Mock()
        mock_pr.mergeable = True
        mock_pr.head.sha = 'abc123'
        mock_pr.base.ref = 'main'
        mock_pr.get_commits.return_value = []

        mock_status = Mock()
        mock_status.state = 'failure'
        mock_status.context = 'ci/test'

        mock_commit = Mock()
        mock_commit.get_statuses.return_value = [mock_status]
        self.mock_repo.get_commit.return_value = mock_commit

        mock_branch = Mock()
        mock_branch.commit = Mock()
        self.mock_repo.get_branch.return_value = mock_branch

        issues = self.resolver.audit_pr(mock_pr)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].type, 'ci_failure')
        self.assertTrue('ci/test' in issues[0].description)

    def test_check_merge_criteria_with_critical_issue(self):
        """Test merge criteria with critical unfixed issue"""
        mock_pr = Mock()
        mock_pr.mergeable = True
        mock_pr.head.sha = 'abc123'

        issues = [
            PRIssue(
                type='conflict',
                description='Test conflict',
                severity='critical',
                fixable=True,
                fixed=False
            )
        ]

        result = self.resolver.check_merge_criteria(mock_pr, issues)

        self.assertFalse(result)

    def test_check_merge_criteria_not_mergeable(self):
        """Test merge criteria when PR not mergeable"""
        mock_pr = Mock()
        mock_pr.mergeable = False
        mock_pr.head.sha = 'abc123'

        result = self.resolver.check_merge_criteria(mock_pr, [])

        self.assertFalse(result)

    @patch('pr_auto_resolver.subprocess.run')
    def test_auto_resolve_file_conflict_documentation(self, mock_run):
        """Test auto-resolving conflict in documentation file"""
        import tempfile

        # Create a test file with conflict markers
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            test_file = Path(f.name)
            f.write("""
Some content
<<<<<<< HEAD
Original content
=======
New content
>>>>>>> branch
More content
""")

        try:
            result = self.resolver._auto_resolve_file_conflict(str(test_file))

            self.assertTrue(result)

            # Check that conflict markers are removed
            content = test_file.read_text()
            self.assertNotIn('<<<<<<<', content)
            self.assertNotIn('=======', content)
            self.assertNotIn('>>>>>>>', content)
            self.assertIn('New content', content)  # Should prefer incoming
        finally:
            # Cleanup
            if test_file.exists():
                test_file.unlink()


class TestPRResolutionResult(unittest.TestCase):
    """Test PRResolutionResult dataclass"""

    def test_create_result(self):
        """Test creating a resolution result"""
        result = PRResolutionResult(
            pr_number=123,
            status='success',
            issues_found=[],
            issues_fixed=[],
            attempts=1,
            merged=True,
            message='All good'
        )

        self.assertEqual(result.pr_number, 123)
        self.assertEqual(result.status, 'success')
        self.assertTrue(result.merged)
        self.assertEqual(result.attempts, 1)


if __name__ == '__main__':
    unittest.main()
