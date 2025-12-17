"""Dockerfile compliance testing for QuASIM CUDA image.

This module tests compliance with:
- NIST 800-53 AC-6 (Least Privilege)
- NIST 800-53 SC-28 (Integrity/Reproducibility)
- Docker best practices
"""

from __future__ import annotations

from pathlib import Path

import pytest


class TestDockerfileCompliance:
    """Test Dockerfile.cuda compliance with security standards."""

    @pytest.fixture
    def dockerfile_content(self):
        """Load Dockerfile.cuda content."""

        dockerfile_path = Path(__file__).parent.parent / "QuASIM" / "Dockerfile.cuda"
        assert dockerfile_path.exists(), "Dockerfile.cuda must exist"
        return dockerfile_path.read_text()

    def test_nist_ac6_non_root_user(self, dockerfile_content):
        """Test AC-6 Least Privilege: non-root user present (NIST 800-53)."""

        # Verify user creation with specific UID
        assert "useradd" in dockerfile_content, "Non-root user must be created"
        assert "-u 1000" in dockerfile_content, "User must have UID 1000"
        assert "appuser" in dockerfile_content, "User must be named 'appuser'"

        # Verify USER directive switches to non-root
        assert "USER appuser" in dockerfile_content, "Must switch to non-root user"

    def test_nist_ac6_workspace_ownership(self, dockerfile_content):
        """Test AC-6: workspace ownership transfer to non-root user."""

        # Verify ownership is transferred to appuser
        assert "chown" in dockerfile_content, "Ownership must be transferred"
        assert "appuser:appuser" in dockerfile_content, "Ownership must be given to appuser"
        assert "/workspace" in dockerfile_content, "Workspace directory must exist"

    def test_nist_sc28_pinned_dependencies(self, dockerfile_content):
        """Test SC-28 Integrity: dependency versions pinned (NIST 800-53)."""

        # Verify pybind11 is pinned
        assert "pybind11==" in dockerfile_content, "pybind11 must be pinned"
        assert "2.11.1" in dockerfile_content, "pybind11 must be pinned to 2.11.1"

        # Verify numpy is pinned (aligned with requirements-prod.txt)
        assert "numpy==" in dockerfile_content, "numpy must be pinned"
        assert "1.24.3" in dockerfile_content, "numpy must be pinned to 1.24.3"

    def test_docker_best_practice_apt_cleanup(self, dockerfile_content):
        """Test Docker best practice: apt cache cleanup."""

        # Verify apt lists are cleaned up
        assert "rm -rf /var/lib/apt/lists/*" in dockerfile_content, "apt cache must be cleaned up"

        # Verify cleanup is in same RUN command as apt-get
        # Split on RUN commands to check each separately
        run_commands = []
        in_run = False
        current_run = []

        for line in dockerfile_content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("RUN"):
                if current_run and in_run:
                    run_commands.append(" ".join(current_run))
                in_run = True
                current_run = [stripped]
            elif in_run:
                if stripped.endswith("\\"):
                    current_run.append(stripped)
                elif stripped:
                    current_run.append(stripped)
                    run_commands.append(" ".join(current_run))
                    in_run = False
                    current_run = []
                elif not stripped and current_run:
                    run_commands.append(" ".join(current_run))
                    in_run = False
                    current_run = []

        if current_run:
            run_commands.append(" ".join(current_run))

        # Find the RUN command with apt-get and verify it has cleanup
        apt_run_with_cleanup = False
        for run_cmd in run_commands:
            if (
                "apt-get update" in run_cmd
                and "apt-get install" in run_cmd
                and "rm -rf /var/lib/apt/lists/*" in run_cmd
            ):
                apt_run_with_cleanup = True
                break

        assert apt_run_with_cleanup, "apt cleanup must be in same RUN command as apt-get"

    def test_compliance_comments_present(self, dockerfile_content):
        """Test that compliance references are documented."""

        # Verify NIST 800-53 references
        assert "NIST 800-53" in dockerfile_content, "NIST 800-53 must be referenced"
        assert "CMMC 2.0" in dockerfile_content, "CMMC 2.0 must be referenced"
        assert "AC-6" in dockerfile_content, "AC-6 control must be referenced"
        assert "SC-28" in dockerfile_content, "SC-28 control must be referenced"

    def test_build_order_preserved(self, dockerfile_content):
        """Test that build order is logical and correct."""

        lines = [line.strip() for line in dockerfile_content.split("\n") if line.strip()]

        # Extract key commands in order (only actual Dockerfile commands, not comments)
        def find_command_index(predicate, error_msg):
            """Find index of command matching predicate."""

            try:
                return next(i for i, line in enumerate(lines) if predicate(line))
            except StopIteration:
                pytest.fail(error_msg)

        from_idx = find_command_index(
            lambda line: line.startswith("FROM"), "FROM command not found"
        )
        user_create_idx = find_command_index(
            lambda line: line.startswith("RUN") and "useradd" in line,
            "RUN useradd command not found",
        )
        workdir_idx = find_command_index(
            lambda line: line.startswith("WORKDIR"), "WORKDIR command not found"
        )
        copy_idx = find_command_index(
            lambda line: line.startswith("COPY"), "COPY command not found"
        )
        pip_idx = find_command_index(
            lambda line: line.startswith("RUN") and "pip3 install" in line and "pybind11" in line,
            "RUN pip3 install command not found",
        )
        cmake_idx = find_command_index(
            lambda line: line.startswith("RUN") and "cmake -S . -B build" in line,
            "RUN cmake command not found",
        )
        chown_idx = find_command_index(
            lambda line: line.startswith("RUN") and "chown" in line, "RUN chown command not found"
        )
        user_switch_idx = find_command_index(
            lambda line: line.startswith("USER"), "USER command not found"
        )
        cmd_idx = find_command_index(lambda line: line.startswith("CMD"), "CMD command not found")

        # Verify order
        assert from_idx < user_create_idx, "User must be created after FROM"
        assert user_create_idx < workdir_idx, "User must be created before WORKDIR"
        assert workdir_idx < copy_idx, "WORKDIR must be set before COPY"
        assert copy_idx < pip_idx, "Files must be copied before pip install"
        assert pip_idx < cmake_idx, "Dependencies must be installed before build"
        assert cmake_idx < chown_idx, "Build must complete before ownership transfer"
        assert chown_idx < user_switch_idx, "Ownership must be transferred before USER switch"
        assert user_switch_idx < cmd_idx, "USER must be switched before CMD"

    def test_no_security_regressions(self, dockerfile_content):
        """Test that no security anti-patterns are present."""

        # Ensure no hardcoded secrets
        assert "password" not in dockerfile_content.lower(), "No hardcoded passwords"
        assert "secret" not in dockerfile_content.lower(), "No hardcoded secrets"
        assert "token" not in dockerfile_content.lower(), "No hardcoded tokens"

        # Ensure no dangerous commands
        assert "chmod 777" not in dockerfile_content, "No world-writable permissions"
        assert "chown -R root" not in dockerfile_content, "No root ownership after user switch"
