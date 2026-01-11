"""Unit tests for Kaggle Chess integration modules."""

from __future__ import annotations

import json
import os
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock

from qratum_chess.benchmarks.kaggle_config import (
    KaggleCredentials,
    KaggleConfig,
    KaggleCompetitionConfig,
    SubmissionFormat,
)
from qratum_chess.benchmarks.kaggle_integration import (
    KaggleIntegration,
    KaggleBenchmarkPosition,
    KaggleLeaderboardData,
)
from qratum_chess.benchmarks.kaggle_submission import (
    KaggleSubmission,
    SubmissionResult,
)


class TestKaggleCredentials:
    """Tests for KaggleCredentials."""
    
    def test_from_file_success(self):
        """Test loading credentials from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"username": "test_user", "key": "test_key"}, f)
            temp_path = f.name
        
        try:
            creds = KaggleCredentials.from_file(temp_path)
            assert creds.username == "test_user"
            assert creds.key == "test_key"
        finally:
            os.unlink(temp_path)
    
    def test_from_file_not_found(self):
        """Test error when credentials file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            KaggleCredentials.from_file("/nonexistent/path/kaggle.json")
    
    def test_from_file_invalid_format(self):
        """Test error when credentials file is invalid."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"invalid": "data"}, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError):
                KaggleCredentials.from_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_from_env_success(self):
        """Test loading credentials from environment."""
        with patch.dict(os.environ, {"KAGGLE_USERNAME": "env_user", "KAGGLE_KEY": "env_key"}):
            creds = KaggleCredentials.from_env()
            assert creds.username == "env_user"
            assert creds.key == "env_key"
    
    def test_from_env_missing(self):
        """Test error when environment variables are missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError):
                KaggleCredentials.from_env()


class TestKaggleCompetitionConfig:
    """Tests for KaggleCompetitionConfig."""
    
    def test_default_config(self):
        """Test default competition configuration."""
        config = KaggleCompetitionConfig()
        assert config.competition_id == "chess-engine-leaderboard"
        assert config.submission_format == "csv"
        assert "position_id" in config.required_fields
    
    def test_custom_config(self):
        """Test custom competition configuration."""
        config = KaggleCompetitionConfig(
            competition_id="test-competition",
            submission_format="json"
        )
        assert config.competition_id == "test-competition"
        assert config.submission_format == "json"
    
    def test_get_submission_endpoint(self):
        """Test submission endpoint generation."""
        config = KaggleCompetitionConfig(competition_id="test-comp")
        endpoint = config.get_submission_endpoint()
        assert "test-comp" in endpoint
        assert "submissions" in endpoint


class TestSubmissionFormat:
    """Tests for SubmissionFormat."""
    
    def test_default_format(self):
        """Test default submission format."""
        fmt = SubmissionFormat()
        assert fmt.format_type == "csv"
        assert "position_id" in fmt.headers
        assert "best_move" in fmt.headers
    
    def test_validate_valid_data(self):
        """Test validation of valid submission data."""
        fmt = SubmissionFormat()
        data = [
            {"position_id": "1", "best_move": "e2e4", "evaluation": 0.5},
            {"position_id": "2", "best_move": "d2d4", "evaluation": 0.3},
        ]
        
        is_valid, error = fmt.validate_submission_data(data)
        assert is_valid
        assert error == ""
    
    def test_validate_empty_data(self):
        """Test validation of empty data."""
        fmt = SubmissionFormat()
        is_valid, error = fmt.validate_submission_data([])
        assert not is_valid
        assert "empty" in error.lower()
    
    def test_validate_missing_field(self):
        """Test validation with missing required field."""
        fmt = SubmissionFormat()
        data = [
            {"position_id": "1", "evaluation": 0.5},  # Missing best_move
        ]
        
        is_valid, error = fmt.validate_submission_data(data)
        assert not is_valid
        assert "missing" in error.lower()


class TestKaggleBenchmarkPosition:
    """Tests for KaggleBenchmarkPosition."""
    
    def test_position_creation(self):
        """Test creating a benchmark position."""
        pos = KaggleBenchmarkPosition(
            position_id="test_1",
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            description="Starting position"
        )
        
        assert pos.position_id == "test_1"
        assert pos.position is not None
        assert pos.description == "Starting position"
    
    def test_position_with_expected_move(self):
        """Test position with expected move."""
        pos = KaggleBenchmarkPosition(
            position_id="test_2",
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            expected_move="e2e4"
        )
        
        assert pos.expected_move == "e2e4"
    
    def test_invalid_fen(self):
        """Test handling of invalid FEN."""
        pos = KaggleBenchmarkPosition(
            position_id="invalid",
            fen="invalid_fen_string"
        )
        
        # Should not raise error, but position should be None
        assert pos.position is None


class TestKaggleIntegration:
    """Tests for KaggleIntegration."""
    
    def test_create_sample_positions(self):
        """Test creating sample positions."""
        integration = KaggleIntegration()
        positions = integration.create_sample_positions()
        
        assert len(positions) > 0
        assert all(isinstance(p, KaggleBenchmarkPosition) for p in positions)
        assert all(p.position is not None for p in positions)
    
    def test_load_from_file(self):
        """Test loading leaderboard data from file."""
        test_data = {
            "positions": [
                {
                    "id": "pos_1",
                    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                    "description": "Test position"
                }
            ],
            "leaderboard": [
                {"team": "TestTeam", "score": 0.95}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            integration = KaggleIntegration()
            leaderboard = integration.load_leaderboard_from_file(temp_path)
            
            assert isinstance(leaderboard, KaggleLeaderboardData)
            assert len(leaderboard.positions) == 1
            assert leaderboard.positions[0].position_id == "pos_1"
        finally:
            os.unlink(temp_path)
    
    def test_looks_like_fen(self):
        """Test FEN string detection."""
        integration = KaggleIntegration()
        
        # Valid FEN
        assert integration._looks_like_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        )
        
        # Invalid FEN strings
        assert not integration._looks_like_fen("not a fen string")
        assert not integration._looks_like_fen("rnbqkbnr/pppppppp w KQkq - 0 1")  # Missing slashes
        assert not integration._looks_like_fen("")


class TestKaggleSubmission:
    """Tests for KaggleSubmission."""
    
    def test_format_as_csv(self):
        """Test CSV formatting."""
        config = KaggleConfig(use_env_credentials=False)
        # Mock the credentials to avoid file access
        config.credentials = KaggleCredentials(username="test", key="test")
        
        submission = KaggleSubmission(config)
        
        results = [
            {"position_id": "1", "best_move": "e2e4", "evaluation": 0.5, 
             "nodes_searched": 1000, "time_ms": 100.0},
            {"position_id": "2", "best_move": "d2d4", "evaluation": 0.3,
             "nodes_searched": 2000, "time_ms": 150.0},
        ]
        
        csv_output = submission._format_as_csv(results)
        
        assert "position_id" in csv_output
        assert "best_move" in csv_output
        assert "e2e4" in csv_output
        assert "d2d4" in csv_output
    
    def test_format_as_json(self):
        """Test JSON formatting."""
        config = KaggleConfig(use_env_credentials=False)
        config.credentials = KaggleCredentials(username="test", key="test")
        
        submission = KaggleSubmission(config)
        
        results = [
            {"position_id": "1", "best_move": "e2e4"},
        ]
        
        json_output = submission._format_as_json(results)
        parsed = json.loads(json_output)
        
        assert len(parsed) == 1
        assert parsed[0]["position_id"] == "1"
    
    def test_validate_submission_valid(self):
        """Test validation of valid submission."""
        config = KaggleConfig(use_env_credentials=False)
        config.credentials = KaggleCredentials(username="test", key="test")
        
        submission = KaggleSubmission(config)
        
        results = [
            {"position_id": "1", "best_move": "e2e4", "evaluation": 0.5},
            {"position_id": "2", "best_move": "d2d4", "evaluation": 0.3},
        ]
        
        is_valid, error = submission.validate_submission(results)
        assert is_valid
    
    def test_validate_submission_duplicates(self):
        """Test validation detects duplicate position IDs."""
        config = KaggleConfig(use_env_credentials=False)
        config.credentials = KaggleCredentials(username="test", key="test")
        
        submission = KaggleSubmission(config)
        
        results = [
            {"position_id": "1", "best_move": "e2e4", "evaluation": 0.5},
            {"position_id": "1", "best_move": "d2d4", "evaluation": 0.3},  # Duplicate
        ]
        
        is_valid, error = submission.validate_submission(results)
        assert not is_valid
        assert "duplicate" in error.lower()
    
    def test_submit_dry_run(self):
        """Test dry run submission."""
        config = KaggleConfig(use_env_credentials=False)
        config.credentials = KaggleCredentials(username="test", key="test")
        
        submission = KaggleSubmission(config)
        
        results = [
            {"position_id": "1", "best_move": "e2e4", "evaluation": 0.5,
             "nodes_searched": 1000, "time_ms": 100.0},
        ]
        
        result = submission.submit_to_kaggle(results, dry_run=True)
        
        assert result.success
        assert "dry run" in result.message.lower()


class TestSubmissionResult:
    """Tests for SubmissionResult."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        from datetime import datetime
        
        result = SubmissionResult(
            success=True,
            submission_id="12345",
            message="Success",
            score=0.95,
            leaderboard_position=10,
            timestamp=datetime(2024, 1, 1, 12, 0, 0)
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["success"] is True
        assert result_dict["submission_id"] == "12345"
        assert result_dict["score"] == 0.95
        assert result_dict["leaderboard_position"] == 10
    
    def test_failed_result(self):
        """Test failed submission result."""
        result = SubmissionResult(
            success=False,
            error="Test error"
        )
        
        assert not result.success
        assert result.error == "Test error"
        assert result.submission_id is None


class TestKaggleLeaderboardData:
    """Tests for KaggleLeaderboardData."""
    
    def test_get_position_by_id(self):
        """Test getting position by ID."""
        pos1 = KaggleBenchmarkPosition(
            position_id="pos_1",
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        )
        pos2 = KaggleBenchmarkPosition(
            position_id="pos_2",
            fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        )
        
        leaderboard = KaggleLeaderboardData(
            positions=[pos1, pos2],
            leaderboard_entries=[],
            metadata={}
        )
        
        found = leaderboard.get_position_by_id("pos_1")
        assert found is not None
        assert found.position_id == "pos_1"
        
        not_found = leaderboard.get_position_by_id("pos_3")
        assert not_found is None
    
    def test_get_top_engines(self):
        """Test getting top engines from leaderboard."""
        entries = [
            {"name": "Engine1", "score": 0.95},
            {"name": "Engine2", "score": 0.90},
            {"name": "Engine3", "score": 0.85},
        ]
        
        leaderboard = KaggleLeaderboardData(
            positions=[],
            leaderboard_entries=entries,
            metadata={}
        )
        
        top = leaderboard.get_top_engines(n=2)
        assert len(top) == 2
        assert top[0]["name"] == "Engine1"
        assert top[1]["name"] == "Engine2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
