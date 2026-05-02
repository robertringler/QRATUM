"""Kaggle Chess Leaderboard Integration.

Loads and parses Kaggle leaderboard data, extracts benchmark positions,
and converts them to QRATUM Position objects.
"""Kaggle Chess Leaderboard API Integration for QRATUM-Chess.

This module provides functionality to:
- Load and parse Kaggle chess leaderboard JSON data
- Extract benchmark positions and test cases from the leaderboard
- Support FEN position extraction from Kaggle API responses
- Convert Kaggle test data to QRATUM Position objects

Usage:
    loader = KaggleLeaderboardLoader()
    leaderboard = loader.load_from_file("kaggle_chess_leaderboard.json")
    positions = loader.extract_positions(leaderboard)
    
    for position in positions:
        # Run QRATUM engine analysis
        pass
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json
import logging
import requests

from qratum_chess.core.position import Position
from qratum_chess.benchmarks.kaggle_config import KaggleConfig


# Set up logging
logger = logging.getLogger(__name__)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json

from qratum_chess.core.position import Position


@dataclass
class KaggleSubmission:
    """Represents a single submission from the Kaggle leaderboard.
    
    Attributes:
        team_name: Name of the team/engine.
        score: Overall leaderboard score.
        rank: Current rank on the leaderboard.
        submission_date: Date of submission.
        metadata: Additional metadata from the submission.
    """
    team_name: str
    score: float
    rank: int
    submission_date: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KaggleBenchmarkPosition:
    """A single benchmark position from Kaggle leaderboard.
    
    Attributes:
        position_id: Unique identifier for the position.
        fen: FEN string representation.
        description: Description or category of the position.
        expected_move: Expected best move (if available).
        difficulty: Difficulty rating (if available).
        position: QRATUM Position object.
    """
    position_id: str
    fen: str
    description: str = ""
    expected_move: str | None = None
    difficulty: float | None = None
    position: Position | None = None
    
    def __post_init__(self):
        """Initialize Position object from FEN."""
        if self.position is None and self.fen:
            try:
                self.position = Position.from_fen(self.fen)
            except Exception as e:
                logger.warning(f"Failed to parse FEN for position {self.position_id}: {e}")
                self.position = None


@dataclass
class KaggleLeaderboardData:
    """Kaggle leaderboard data.
    
    Attributes:
        positions: List of benchmark positions.
        leaderboard_entries: List of leaderboard entries.
        metadata: Additional metadata from the leaderboard.
    """
    positions: list[KaggleBenchmarkPosition]
    leaderboard_entries: list[dict[str, Any]]
    metadata: dict[str, Any]
    
    def get_position_by_id(self, position_id: str) -> KaggleBenchmarkPosition | None:
        """Get a position by its ID.
        
        Args:
            position_id: Position identifier.
            
        Returns:
            KaggleBenchmarkPosition or None if not found.
        """
        for pos in self.positions:
            if pos.position_id == position_id:
                return pos
        return None
    
    def get_top_engines(self, n: int = 10) -> list[dict[str, Any]]:
        """Get top N engines from leaderboard.
        
        Args:
            n: Number of top engines to return.
            
        Returns:
            List of leaderboard entries.
        """
        return sorted(
            self.leaderboard_entries,
            key=lambda x: x.get("score", 0),
            reverse=True
        )[:n]


class KaggleIntegration:
    """Integration with Kaggle Chess Leaderboard API.
    
    Handles data loading, parsing, and position extraction from Kaggle.
    """
    
    def __init__(self, config: KaggleConfig | None = None):
        """Initialize Kaggle integration.
        
        Args:
            config: Kaggle configuration. If None, uses default config.
        """
        self.config = config
    
    def load_leaderboard_from_file(self, filepath: str) -> KaggleLeaderboardData:
        """Load leaderboard data from a local JSON file.
        
        Args:
            filepath: Path to JSON file.
            
        Returns:
            KaggleLeaderboardData instance.
            
        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If JSON is invalid.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return self._parse_leaderboard_data(data)
    
    def download_leaderboard_data(
        self,
        competition_id: str | None = None,
        save_path: str | None = None
    ) -> KaggleLeaderboardData:
        """Download leaderboard data from Kaggle API.
        
        Args:
            competition_id: Kaggle competition ID. Uses config default if None.
            save_path: Optional path to save the downloaded data.
            
        Returns:
            KaggleLeaderboardData instance.
            
        Raises:
            requests.RequestException: If API request fails.
        """
        if self.config is None:
            raise ValueError("Config required for API access")
        
        if competition_id is None:
            competition_id = self.config.competition.competition_id
        
        # Download leaderboard data
        url = f"https://www.kaggle.com/api/v1/competitions/{competition_id}/leaderboard"
        headers = self.config.get_auth_headers()
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        # Save to file if requested
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        return self._parse_leaderboard_data(data)
    
    def download_benchmark_positions(
        self,
        competition_id: str | None = None,
        save_path: str | None = None
    ) -> list[KaggleBenchmarkPosition]:
        """Download benchmark positions from Kaggle competition.
        
        Args:
            competition_id: Kaggle competition ID.
            save_path: Optional path to save the downloaded data.
            
        Returns:
            List of KaggleBenchmarkPosition instances.
        """
        if self.config is None:
            raise ValueError("Config required for API access")
        
        if competition_id is None:
            competition_id = self.config.competition.competition_id
        
        # Try to download test data
        url = f"https://www.kaggle.com/api/v1/competitions/{competition_id}/data/list"
        headers = self.config.get_auth_headers()
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            
            # Save to file if requested
            if save_path:
                with open(save_path, 'w') as f:
                    json.dump(data, f, indent=2)
            
            # Parse positions from data
            return self._extract_positions_from_data(data)
        else:
            # If API call fails, log warning and return empty list
            logger.warning(f"Could not download benchmark positions: {response.status_code}")
            return []
    
    def _parse_leaderboard_data(self, data: dict[str, Any]) -> KaggleLeaderboardData:
        """Parse leaderboard data from JSON.
        
        Args:
            data: Raw JSON data from Kaggle API.
            
        Returns:
            KaggleLeaderboardData instance.
        """
        # Extract positions from data
        positions = self._extract_positions_from_data(data)
        
        # Extract leaderboard entries
        leaderboard_entries = []
        if "submissions" in data:
            leaderboard_entries = data["submissions"]
        elif "leaderboard" in data:
            leaderboard_entries = data["leaderboard"]
        
        # Extract metadata
        metadata = {
            "competition_id": data.get("competition_id", "unknown"),
            "total_positions": len(positions),
            "total_entries": len(leaderboard_entries),
            "last_updated": data.get("last_updated", None),
        }
        
        return KaggleLeaderboardData(
            positions=positions,
            leaderboard_entries=leaderboard_entries,
            metadata=metadata
        )
    
    def _extract_positions_from_data(
        self,
        data: dict[str, Any]
    ) -> list[KaggleBenchmarkPosition]:
        """Extract benchmark positions from Kaggle data.
    """Represents a benchmark position from Kaggle data.
    
    Attributes:
        fen: FEN string of the position.
        position: QRATUM Position object.
        test_id: Identifier for this test position.
        expected_move: Expected best move (if available).
        expected_eval: Expected evaluation (if available).
        difficulty: Difficulty rating of the position.
        category: Category of the position (tactics, endgame, etc.).
    """
    fen: str
    position: Position
    test_id: str
    expected_move: str | None = None
    expected_eval: float | None = None
    difficulty: str = "medium"
    category: str = "general"


@dataclass
class KaggleLeaderboard:
    """Complete Kaggle leaderboard data.
    
    Attributes:
        benchmark_name: Name of the benchmark.
        version: Version of the benchmark.
        submissions: List of submissions.
        test_positions: List of benchmark positions.
        metadata: Additional leaderboard metadata.
    """
    benchmark_name: str
    version: str
    submissions: list[KaggleSubmission] = field(default_factory=list)
    test_positions: list[KaggleBenchmarkPosition] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class KaggleLeaderboardLoader:
    """Loads and parses Kaggle chess leaderboard data.
    
    Handles various Kaggle API response formats and extracts
    chess positions for benchmarking.
    """
    
    def __init__(self):
        """Initialize the Kaggle leaderboard loader."""
        self.leaderboard: KaggleLeaderboard | None = None
    
    def load_from_file(self, filepath: str | Path) -> KaggleLeaderboard:
        """Load leaderboard data from a JSON file.
        
        Args:
            filepath: Path to the JSON file.
            
        Returns:
            Parsed leaderboard data.
            
        Raises:
            FileNotFoundError: If the file doesn't exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Kaggle leaderboard file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return self.parse_leaderboard(data)
    
    def load_from_dict(self, data: dict[str, Any]) -> KaggleLeaderboard:
        """Load leaderboard data from a dictionary.
        
        Args:
            data: Leaderboard data as a dictionary.
            
        Returns:
            Parsed leaderboard data.
        """
        return self.parse_leaderboard(data)
    
    def parse_leaderboard(self, data: dict[str, Any]) -> KaggleLeaderboard:
        """Parse leaderboard data from Kaggle API response.
        
        Args:
            data: Raw data from Kaggle API.
            
        Returns:
            Parsed leaderboard object.
        """
        # Extract basic metadata
        benchmark_name = data.get("benchmarkName", "chess")
        version = str(data.get("version", "1"))
        
        # Parse submissions
        submissions = []
        submissions_data = data.get("submissions", [])
        
        for idx, sub_data in enumerate(submissions_data):
            submission = KaggleSubmission(
                team_name=sub_data.get("teamName", f"Team_{idx}"),
                score=float(sub_data.get("score", 0.0)),
                rank=int(sub_data.get("rank", idx + 1)),
                submission_date=sub_data.get("submissionDate", ""),
                metadata=sub_data.get("metadata", {})
            )
            submissions.append(submission)
        
        # Parse test positions
        test_positions = self._extract_test_positions(data)
        
        leaderboard = KaggleLeaderboard(
            benchmark_name=benchmark_name,
            version=version,
            submissions=submissions,
            test_positions=test_positions,
            metadata={k: v for k, v in data.items() if k not in ["submissions", "testData"]}
        )
        
        self.leaderboard = leaderboard
        return leaderboard
    
    def _extract_test_positions(self, data: dict[str, Any]) -> list[KaggleBenchmarkPosition]:
        """Extract test positions from Kaggle data.
        
        Args:
            data: Raw data from Kaggle API.
            
        Returns:
            List of KaggleBenchmarkPosition instances.
        """
        positions = []
        
        # Try different data structures
        position_data = None
        if "positions" in data:
            position_data = data["positions"]
        elif "test_positions" in data:
            position_data = data["test_positions"]
        elif "benchmark_positions" in data:
            position_data = data["benchmark_positions"]
        elif "data" in data and isinstance(data["data"], list):
            position_data = data["data"]
        
        if position_data is None:
            # Try to find FEN strings in any field
            position_data = self._find_fen_strings(data)
        
        if position_data:
            for i, pos_entry in enumerate(position_data):
                if isinstance(pos_entry, dict):
                    position = self._parse_position_entry(pos_entry, i)
                elif isinstance(pos_entry, str):
                    # Assume it's a FEN string
                    position = KaggleBenchmarkPosition(
                        position_id=f"pos_{i}",
                        fen=pos_entry
                    )
                else:
                    continue
                
                if position:
                    positions.append(position)
        
        return positions
    
    def _parse_position_entry(
        self,
        entry: dict[str, Any],
        index: int
    ) -> KaggleBenchmarkPosition | None:
        """Parse a single position entry.
        
        Args:
            entry: Position entry dictionary.
            index: Position index.
            
        Returns:
            KaggleBenchmarkPosition or None if parsing fails.
        """
        # Try to find FEN field
        fen = None
        for field in ["fen", "FEN", "position", "Position", "board"]:
            if field in entry:
                fen = entry[field]
                break
        
        if not fen:
            return None
        
        # Extract other fields
        position_id = entry.get("id", entry.get("position_id", f"pos_{index}"))
        description = entry.get("description", entry.get("category", ""))
        expected_move = entry.get("expected_move", entry.get("best_move", None))
        difficulty = entry.get("difficulty", entry.get("rating", None))
        
        return KaggleBenchmarkPosition(
            position_id=str(position_id),
            fen=fen,
            description=description,
            expected_move=expected_move,
            difficulty=difficulty
        )
    
    def _find_fen_strings(self, data: dict[str, Any]) -> list[str]:
        """Recursively find FEN strings in data.
        
        Args:
            data: Data dictionary to search.
            
        Returns:
            List of FEN strings found.
        """
        fen_strings = []
        
        def search_dict(d):
            if isinstance(d, dict):
                for key, value in d.items():
                    if isinstance(value, str) and self._looks_like_fen(value):
                        fen_strings.append(value)
                    elif isinstance(value, (dict, list)):
                        search_dict(value)
            elif isinstance(d, list):
                for item in d:
                    search_dict(item)
        
        search_dict(data)
        return fen_strings
    
    def _looks_like_fen(self, s: str) -> bool:
        """Check if a string looks like a FEN string.
        
        Args:
            s: String to check.
            
        Returns:
            True if string looks like FEN.
        """
        if not isinstance(s, str):
            return False
        
        parts = s.split()
        
        # FEN should have 6 parts
        if len(parts) != 6:
            return False
        
        # First part should be board position with 7 slashes
        if parts[0].count('/') != 7:
            return False
        
        # Second part should be w or b
        if parts[1] not in ['w', 'b']:
            return False
        
        return True
    
    def create_sample_positions(self) -> list[KaggleBenchmarkPosition]:
        """Create sample benchmark positions for testing.
        
        Returns:
            List of sample KaggleBenchmarkPosition instances.
        """
        sample_fens = [
            ("start", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting position"),
            ("sicilian", "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2", "Sicilian Defense"),
            ("endgame", "8/8/8/8/8/6k1/4K3/7R w - - 0 1", "Rook endgame"),
            ("tactics", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "Tactical position"),
            ("complex", "r1bq1rk1/pp2ppbp/2np1np1/8/2BNP3/2N1BP2/PPPQ2PP/R3K2R w KQ - 2 10", "Complex middlegame"),
        ]
        
        positions = []
        for pos_id, fen, desc in sample_fens:
            positions.append(KaggleBenchmarkPosition(
                position_id=pos_id,
                fen=fen,
                description=desc
            ))
        
        return positions
            List of benchmark positions.
        """
        positions = []
        
        # Check various possible locations for test data
        test_data = data.get("testData", [])
        if not test_data:
            test_data = data.get("test_data", [])
        if not test_data:
            test_data = data.get("positions", [])
        if not test_data:
            test_data = data.get("benchmarkData", [])
        
        for idx, test_item in enumerate(test_data):
            # Extract FEN string
            fen = self._extract_fen(test_item, idx)
            
            if fen:
                try:
                    position_obj = Position.from_fen(fen)
                    
                    benchmark_pos = KaggleBenchmarkPosition(
                        fen=fen,
                        position=position_obj,
                        test_id=test_item.get("id", f"test_{idx}"),
                        expected_move=test_item.get("bestMove", test_item.get("expected_move")),
                        expected_eval=test_item.get("evaluation", test_item.get("expected_eval")),
                        difficulty=test_item.get("difficulty", "medium"),
                        category=test_item.get("category", "general")
                    )
                    positions.append(benchmark_pos)
                except (ValueError, KeyError) as e:
                    # Skip invalid positions
                    print(f"Warning: Failed to parse position {idx}: {e}")
                    continue
        
        # If no positions found, generate standard test positions
        if not positions:
            positions = self._generate_standard_positions()
        
        return positions
    
    def _extract_fen(self, test_item: dict[str, Any], idx: int) -> str | None:
        """Extract FEN string from a test item.
        
        Args:
            test_item: Test data item.
            idx: Index of the test item.
            
        Returns:
            FEN string or None if not found.
        """
        # Try various field names
        fen = test_item.get("fen")
        if not fen:
            fen = test_item.get("FEN")
        if not fen:
            fen = test_item.get("position")
        if not fen:
            fen = test_item.get("board")
        
        return fen
    
    def _generate_standard_positions(self) -> list[KaggleBenchmarkPosition]:
        """Generate standard chess benchmark positions.
        
        Returns:
            List of standard benchmark positions.
        """
        # Standard benchmark positions used in chess engines
        standard_positions = [
            {
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "test_id": "starting",
                "category": "opening",
                "difficulty": "easy"
            },
            {
                "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
                "test_id": "italian",
                "category": "opening",
                "difficulty": "easy"
            },
            {
                "fen": "rnbqkb1r/pp1ppppp/5n2/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
                "test_id": "sicilian",
                "category": "opening",
                "difficulty": "medium"
            },
            {
                "fen": "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 4 7",
                "test_id": "middlegame_1",
                "category": "middlegame",
                "difficulty": "medium"
            },
            {
                "fen": "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 7 9",
                "test_id": "middlegame_2",
                "category": "middlegame",
                "difficulty": "hard"
            },
            {
                "fen": "8/8/8/8/8/6k1/4K3/7R w - - 0 1",
                "test_id": "endgame_rook",
                "category": "endgame",
                "difficulty": "easy"
            },
            {
                "fen": "8/8/p1p5/1p5p/1P5p/8/PPP2K1k/8 w - - 0 1",
                "test_id": "endgame_pawn",
                "category": "endgame",
                "difficulty": "hard"
            },
            {
                "fen": "1r3rk1/5ppp/8/8/8/8/5PPP/1R3RK1 w - - 0 1",
                "test_id": "endgame_rook_pair",
                "category": "endgame",
                "difficulty": "medium"
            }
        ]
        
        positions = []
        for pos_data in standard_positions:
            try:
                position_obj = Position.from_fen(pos_data["fen"])
                benchmark_pos = KaggleBenchmarkPosition(
                    fen=pos_data["fen"],
                    position=position_obj,
                    test_id=pos_data["test_id"],
                    category=pos_data["category"],
                    difficulty=pos_data["difficulty"]
                )
                positions.append(benchmark_pos)
            except (ValueError, KeyError):
                continue
        
        return positions
    
    def extract_positions(
        self,
        leaderboard: KaggleLeaderboard | None = None
    ) -> list[KaggleBenchmarkPosition]:
        """Extract all benchmark positions from the leaderboard.
        
        Args:
            leaderboard: Leaderboard to extract from (uses self.leaderboard if None).
            
        Returns:
            List of benchmark positions.
        """
        if leaderboard is None:
            leaderboard = self.leaderboard
        
        if leaderboard is None:
            raise ValueError("No leaderboard data loaded")
        
        return leaderboard.test_positions
    
    def get_top_submissions(
        self,
        n: int = 10,
        leaderboard: KaggleLeaderboard | None = None
    ) -> list[KaggleSubmission]:
        """Get top N submissions from the leaderboard.
        
        Args:
            n: Number of top submissions to return.
            leaderboard: Leaderboard to extract from (uses self.leaderboard if None).
            
        Returns:
            List of top submissions.
        """
        if leaderboard is None:
            leaderboard = self.leaderboard
        
        if leaderboard is None:
            raise ValueError("No leaderboard data loaded")
        
        # Sort by rank (lower is better) or score (higher is better)
        sorted_submissions = sorted(
            leaderboard.submissions,
            key=lambda s: (s.rank, -s.score)
        )
        
        return sorted_submissions[:n]
    
    def export_positions_to_fen_file(
        self,
        filepath: str | Path,
        leaderboard: KaggleLeaderboard | None = None
    ) -> None:
        """Export positions to a FEN file (one per line).
        
        Args:
            filepath: Output file path.
            leaderboard: Leaderboard to export from (uses self.leaderboard if None).
        """
        positions = self.extract_positions(leaderboard)
        
        filepath = Path(filepath)
        with open(filepath, 'w') as f:
            for pos in positions:
                f.write(f"{pos.fen}\n")


def download_kaggle_leaderboard(output_path: str | Path) -> bool:
    """Download Kaggle chess leaderboard using curl.
    
    Args:
        output_path: Path to save the leaderboard JSON.
        
    Returns:
        True if download was successful, False otherwise.
    """
    import subprocess
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    url = "https://www.kaggle.com/api/v1/benchmarks/kaggle/chess/versions/1/leaderboard"
    
    try:
        # Use curl to download
        result = subprocess.run(
            ["curl", "-L", "-o", str(output_path), url],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0 and output_path.exists():
            return True
        else:
            print(f"Download failed: {result.stderr}")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"Error downloading leaderboard: {e}")
        return False
