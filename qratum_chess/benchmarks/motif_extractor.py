"""Motif extraction module for QRATUM-Chess benchmarking.

Extracts and classifies novel chess patterns discovered during benchmarking:
- Tactical motifs (combinations, traps, deflections)
- Strategic motifs (pawn structures, piece coordination)
- Opening motifs (novel opening lines)
- Endgame motifs (conversion techniques, zugzwang)

Each motif is analyzed for:
- Cortex activation patterns
- Novelty score vs engine databases
- FEN position and evaluation
- Move sequence leading to discovery
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
import csv
import json
import time


class MotifType(Enum):
    """Type of chess motif."""
    TACTICAL = "tactical"
    STRATEGIC = "strategic"
    OPENING = "opening"
    ENDGAME = "endgame"
    CONCEPTUAL = "conceptual"


class GamePhase(Enum):
    """Phase of the game."""
    OPENING = "opening"
    MIDDLEGAME = "middlegame"
    ENDGAME = "endgame"


@dataclass
class Motif:
    """Represents a discovered chess motif.
    
    Attributes:
        motif_id: Unique identifier for the motif.
        motif_type: Classification of the motif.
        game_phase: Phase of game where motif was discovered.
        position_fen: Position in FEN notation.
        move_sequence: Sequence of moves (UCI notation).
        evaluation: Position evaluation score (centipawns).
        novelty_score: Novelty score 0.0-1.0 (1.0 = completely novel).
        cortex_weights: Activation weights for each cortex.
        novelty_pressure: Novelty pressure contribution Ω(a).
        engine_comparison: Comparison to standard engines.
        discovery_timestamp: When motif was discovered.
        description: Human-readable description (optional).
    """
    motif_id: str
    motif_type: MotifType
    game_phase: GamePhase
    position_fen: str
    move_sequence: list[str]
    evaluation: float
    novelty_score: float
    cortex_weights: dict[str, float] = field(default_factory=dict)
    novelty_pressure: float = 0.0
    engine_comparison: dict[str, Any] = field(default_factory=dict)
    discovery_timestamp: float = field(default_factory=time.time)
    description: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert motif to dictionary format."""
        return {
            "motif_id": self.motif_id,
            "motif_type": self.motif_type.value,
            "game_phase": self.game_phase.value,
            "position_fen": self.position_fen,
            "move_sequence": self.move_sequence,
            "evaluation": self.evaluation,
            "novelty_score": self.novelty_score,
            "cortex_weights": self.cortex_weights,
            "novelty_pressure": self.novelty_pressure,
            "engine_comparison": self.engine_comparison,
            "discovery_timestamp": self.discovery_timestamp,
            "description": self.description,
        }


class MotifExtractor:
    """Extracts and classifies novel chess motifs from telemetry data.
    
    Analyzes telemetry output to identify:
    - Positions with high novelty pressure
    - Moves diverging from engine databases
    - Patterns with conceptual cortex activation
    - Tactical/strategic breakthroughs
    """
    
    def __init__(
        self,
        novelty_threshold: float = 0.6,
        divergence_threshold: float = 0.5,
        min_cortex_activation: float = 0.3,
    ):
        """Initialize motif extractor.
        
        Args:
            novelty_threshold: Minimum novelty score to classify as motif.
            divergence_threshold: Minimum divergence from engines to classify as novel.
            min_cortex_activation: Minimum cortex activation to consider.
        """
        self.novelty_threshold = novelty_threshold
        self.divergence_threshold = divergence_threshold
        self.min_cortex_activation = min_cortex_activation
        self.motifs: list[Motif] = []
        self.motif_counter = 0
    
    def extract_from_telemetry(self, telemetry_data: dict[str, Any]) -> list[Motif]:
        """Extract motifs from telemetry data.
        
        Args:
            telemetry_data: Telemetry data dictionary from TelemetryOutput.
            
        Returns:
            List of discovered motifs.
        """
        current = telemetry_data.get("current", {})
        
        # Extract from pattern inventions (explicit novel patterns)
        pattern_inventions = current.get("pattern_inventions", [])
        for invention in pattern_inventions:
            if invention.get("novelty_score", 0) >= self.novelty_threshold:
                motif = self._create_motif_from_invention(invention)
                self.motifs.append(motif)
        
        # Extract from move divergences (implicit novel moves)
        move_divergences = current.get("move_divergence", [])
        novelty_pressure = current.get("novelty_pressure", [])
        cortex_activations = current.get("cortex_activations", [])
        
        for i, divergence in enumerate(move_divergences):
            if divergence.get("divergence", 0) >= self.divergence_threshold:
                # Get corresponding data
                pressure = novelty_pressure[i] if i < len(novelty_pressure) else 0.0
                cortex = cortex_activations[i] if i < len(cortex_activations) else {}
                
                motif = self._create_motif_from_divergence(divergence, pressure, cortex)
                self.motifs.append(motif)
        
        return self.motifs
    
    def _create_motif_from_invention(self, invention: dict[str, Any]) -> Motif:
        """Create motif from pattern invention record.
        
        Args:
            invention: Pattern invention data.
            
        Returns:
            Motif object.
        """
        self.motif_counter += 1
        motif_id = f"MOTIF_{self.motif_counter:04d}"
        
        # Determine motif type from pattern type
        pattern_type = invention.get("pattern_type", "tactical")
        motif_type = self._classify_motif_type(pattern_type)
        
        # Determine game phase from position
        position_fen = invention.get("position", "")
        game_phase = self._determine_game_phase(position_fen)
        
        return Motif(
            motif_id=motif_id,
            motif_type=motif_type,
            game_phase=game_phase,
            position_fen=position_fen,
            move_sequence=invention.get("moves", []),
            evaluation=self._estimate_evaluation(position_fen),
            novelty_score=invention.get("novelty_score", 0.0),
            cortex_weights={},
            novelty_pressure=0.0,
            engine_comparison={},
            discovery_timestamp=invention.get("timestamp", time.time()),
            description=f"Novel {pattern_type} pattern discovered",
        )
    
    def _create_motif_from_divergence(
        self,
        divergence: dict[str, Any],
        pressure: float,
        cortex: dict[str, float]
    ) -> Motif:
        """Create motif from move divergence record.
        
        Args:
            divergence: Move divergence data.
            pressure: Novelty pressure value.
            cortex: Cortex activation weights.
            
        Returns:
            Motif object.
        """
        self.motif_counter += 1
        motif_id = f"MOTIF_{self.motif_counter:04d}"
        
        position_fen = divergence.get("position", "")
        selected_move = divergence.get("selected_move", "")
        engine_move = divergence.get("engine_move", "")
        divergence_score = divergence.get("divergence", 0.0)
        
        # Classify motif type based on cortex activations
        motif_type = self._classify_from_cortex(cortex)
        game_phase = self._determine_game_phase(position_fen)
        
        return Motif(
            motif_id=motif_id,
            motif_type=motif_type,
            game_phase=game_phase,
            position_fen=position_fen,
            move_sequence=[selected_move],
            evaluation=self._estimate_evaluation(position_fen),
            novelty_score=divergence_score,
            cortex_weights=cortex,
            novelty_pressure=pressure,
            engine_comparison={
                "engine_move": engine_move,
                "selected_move": selected_move,
                "divergence": divergence_score,
            },
            discovery_timestamp=divergence.get("timestamp", time.time()),
            description=f"Novel move diverging from engine baseline",
        )
    
    def _classify_motif_type(self, pattern_type: str) -> MotifType:
        """Classify motif type from pattern type string.
        
        Args:
            pattern_type: Pattern type string.
            
        Returns:
            MotifType enum value.
        """
        pattern_lower = pattern_type.lower()
        
        if "tactical" in pattern_lower:
            return MotifType.TACTICAL
        elif "strategic" in pattern_lower:
            return MotifType.STRATEGIC
        elif "opening" in pattern_lower:
            return MotifType.OPENING
        elif "endgame" in pattern_lower:
            return MotifType.ENDGAME
        elif "conceptual" in pattern_lower:
            return MotifType.CONCEPTUAL
        else:
            return MotifType.TACTICAL
    
    def _classify_from_cortex(self, cortex: dict[str, float]) -> MotifType:
        """Classify motif type from cortex activation weights.
        
        Args:
            cortex: Cortex activation weights.
            
        Returns:
            MotifType enum value.
        """
        if not cortex:
            return MotifType.TACTICAL
        
        tactical = cortex.get("tactical", 0.0)
        strategic = cortex.get("strategic", 0.0)
        conceptual = cortex.get("conceptual", 0.0)
        
        # Find dominant cortex
        max_activation = max(tactical, strategic, conceptual)
        
        if max_activation < self.min_cortex_activation:
            return MotifType.TACTICAL
        
        if tactical == max_activation:
            return MotifType.TACTICAL
        elif strategic == max_activation:
            return MotifType.STRATEGIC
        else:
            return MotifType.CONCEPTUAL
    
    def _determine_game_phase(self, position_fen: str) -> GamePhase:
        """Determine game phase from FEN position.
        
        Args:
            position_fen: Position in FEN notation.
            
        Returns:
            GamePhase enum value.
        """
        if not position_fen:
            return GamePhase.MIDDLEGAME
        
        # Simple heuristic based on piece count
        piece_placement = position_fen.split()[0]
        
        # Count pieces (excluding kings and pawns)
        piece_count = sum(1 for c in piece_placement 
                         if c.upper() in "QRBN")
        
        # Count total pieces
        total_pieces = sum(1 for c in piece_placement 
                          if c.upper() in "KQRBNP")
        
        # Opening: many pieces, likely early game
        if total_pieces >= 28:
            return GamePhase.OPENING
        # Endgame: few pieces
        elif piece_count <= 6:
            return GamePhase.ENDGAME
        # Middlegame: default
        else:
            return GamePhase.MIDDLEGAME
    
    def _estimate_evaluation(self, position_fen: str) -> float:
        """Estimate position evaluation (placeholder).
        
        Args:
            position_fen: Position in FEN notation.
            
        Returns:
            Evaluation in centipawns (0.0 if cannot estimate).
        """
        # Placeholder - in real implementation, this would call an evaluator
        return 0.0
    
    def filter_by_type(self, motif_type: MotifType) -> list[Motif]:
        """Filter motifs by type.
        
        Args:
            motif_type: Type to filter by.
            
        Returns:
            List of motifs of the specified type.
        """
        return [m for m in self.motifs if m.motif_type == motif_type]
    
    def filter_by_phase(self, game_phase: GamePhase) -> list[Motif]:
        """Filter motifs by game phase.
        
        Args:
            game_phase: Phase to filter by.
            
        Returns:
            List of motifs from the specified phase.
        """
        return [m for m in self.motifs if m.game_phase == game_phase]
    
    def get_top_motifs(self, n: int = 10, sort_by: str = "novelty") -> list[Motif]:
        """Get top N motifs sorted by a criterion.
        
        Args:
            n: Number of top motifs to return.
            sort_by: Sorting criterion ("novelty", "pressure", "evaluation").
            
        Returns:
            List of top motifs.
        """
        if sort_by == "novelty":
            sorted_motifs = sorted(self.motifs, key=lambda m: m.novelty_score, reverse=True)
        elif sort_by == "pressure":
            sorted_motifs = sorted(self.motifs, key=lambda m: m.novelty_pressure, reverse=True)
        elif sort_by == "evaluation":
            sorted_motifs = sorted(self.motifs, key=lambda m: abs(m.evaluation), reverse=True)
        else:
            sorted_motifs = self.motifs
        
        return sorted_motifs[:n]
    
    def export_catalog_json(self, filepath: Path) -> None:
        """Export motif catalog to JSON file.
        
        Args:
            filepath: Output file path.
        """
        catalog = {
            "total_motifs": len(self.motifs),
            "extraction_timestamp": time.time(),
            "thresholds": {
                "novelty_threshold": self.novelty_threshold,
                "divergence_threshold": self.divergence_threshold,
                "min_cortex_activation": self.min_cortex_activation,
            },
            "motifs_by_type": {
                motif_type.value: len(self.filter_by_type(motif_type))
                for motif_type in MotifType
            },
            "motifs_by_phase": {
                phase.value: len(self.filter_by_phase(phase))
                for phase in GamePhase
            },
            "motifs": [m.to_dict() for m in self.motifs],
        }
        
        with open(filepath, 'w') as f:
            json.dump(catalog, f, indent=2)
    
    def export_summary_csv(self, filepath: Path) -> None:
        """Export motif summary to CSV file.
        
        Args:
            filepath: Output file path.
        """
        rows = [["motif_id", "type", "phase", "novelty_score", "evaluation", 
                "novelty_pressure", "move_count", "position_fen"]]
        
        for motif in self.motifs:
            rows.append([
                motif.motif_id,
                motif.motif_type.value,
                motif.game_phase.value,
                f"{motif.novelty_score:.4f}",
                f"{motif.evaluation:.2f}",
                f"{motif.novelty_pressure:.4f}",
                len(motif.move_sequence),
                motif.position_fen,
            ])
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
    
    def export_pgn(self, filepath: Path, motif_type: MotifType | None = None) -> None:
        """Export motifs as PGN file.
        
        Args:
            filepath: Output file path.
            motif_type: Optional filter by motif type.
        """
        motifs = self.filter_by_type(motif_type) if motif_type else self.motifs
        
        with open(filepath, 'w') as f:
            for motif in motifs:
                # PGN header
                f.write(f'[Event "QRATUM-Chess Motif Discovery"]\n')
                f.write(f'[Site "Benchmark Suite"]\n')
                f.write(f'[Date "{time.strftime("%Y.%m.%d", time.localtime(motif.discovery_timestamp))}"]\n')
                f.write(f'[Round "{motif.motif_id}"]\n')
                f.write(f'[White "Bob (QRATUM)"]\n')
                f.write(f'[Black "Analysis"]\n')
                f.write(f'[Result "*"]\n')
                f.write(f'[FEN "{motif.position_fen}"]\n')
                f.write(f'[MotifType "{motif.motif_type.value}"]\n')
                f.write(f'[GamePhase "{motif.game_phase.value}"]\n')
                f.write(f'[NoveltyScore "{motif.novelty_score:.4f}"]\n')
                f.write(f'[Evaluation "{motif.evaluation:.2f}"]\n')
                f.write('\n')
                
                # Moves
                moves_str = ' '.join(motif.move_sequence)
                f.write(f'{moves_str} *\n\n')
    
    def generate_html_report(self, filepath: Path) -> None:
        """Generate HTML report of discovered motifs.
        
        Args:
            filepath: Output file path.
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>QRATUM-Chess Motif Catalog</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; 
               background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 100%); color: #fff; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #00f5ff; text-align: center; margin-bottom: 5px; }}
        .subtitle {{ text-align: center; color: #888; margin-bottom: 30px; }}
        .summary {{ background: rgba(15,15,25,0.9); border: 1px solid #333; 
                   border-radius: 8px; padding: 20px; margin: 20px 0; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); 
                      gap: 15px; margin: 20px 0; }}
        .stat-box {{ background: #1a1a2e; padding: 15px; border-radius: 4px; text-align: center; }}
        .stat-label {{ color: #888; font-size: 0.9em; }}
        .stat-value {{ color: #00f5ff; font-size: 1.5em; font-weight: bold; }}
        .motif-card {{ background: rgba(15,15,25,0.9); border: 1px solid #333; 
                      border-radius: 8px; padding: 20px; margin: 15px 0; }}
        .motif-header {{ display: flex; justify-content: space-between; align-items: center; 
                        border-bottom: 1px solid #333; padding-bottom: 10px; margin-bottom: 15px; }}
        .motif-id {{ color: #00f5ff; font-size: 1.2em; font-weight: bold; }}
        .motif-type {{ background: #7b2cbf; color: #fff; padding: 5px 10px; 
                      border-radius: 4px; font-size: 0.9em; }}
        .motif-details {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }}
        .detail-item {{ margin: 5px 0; }}
        .detail-label {{ color: #888; display: inline-block; width: 150px; }}
        .detail-value {{ color: #fff; }}
        .fen-box {{ background: #0a0a0f; padding: 10px; border-radius: 4px; 
                   font-family: monospace; font-size: 0.9em; margin: 10px 0; }}
        .novelty-high {{ color: #00ff88; }}
        .novelty-medium {{ color: #ffaa00; }}
        .novelty-low {{ color: #ff6b6b; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧩 QRATUM-Chess Novel Motif Catalog</h1>
        <p class="subtitle">Generated: {timestamp}</p>
        
        <div class="summary">
            <h2 style="color: #7b2cbf; margin-top: 0;">Summary Statistics</h2>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-label">Total Motifs</div>
                    <div class="stat-value">{len(self.motifs)}</div>
                </div>
"""
        
        # Statistics by type
        for motif_type in MotifType:
            count = len(self.filter_by_type(motif_type))
            html_content += f"""
                <div class="stat-box">
                    <div class="stat-label">{motif_type.value.title()}</div>
                    <div class="stat-value">{count}</div>
                </div>
"""
        
        html_content += """
            </div>
        </div>
"""
        
        # Top motifs
        top_motifs = self.get_top_motifs(n=20, sort_by="novelty")
        
        html_content += """
        <h2 style="color: #7b2cbf; margin-top: 30px;">Top Discovered Motifs</h2>
"""
        
        for motif in top_motifs:
            novelty_class = ("novelty-high" if motif.novelty_score >= 0.8 
                           else "novelty-medium" if motif.novelty_score >= 0.6 
                           else "novelty-low")
            
            html_content += f"""
        <div class="motif-card">
            <div class="motif-header">
                <span class="motif-id">{motif.motif_id}</span>
                <span class="motif-type">{motif.motif_type.value.upper()}</span>
            </div>
            <div class="motif-details">
                <div>
                    <div class="detail-item">
                        <span class="detail-label">Game Phase:</span>
                        <span class="detail-value">{motif.game_phase.value.title()}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Novelty Score:</span>
                        <span class="detail-value {novelty_class}">{motif.novelty_score:.4f}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Evaluation:</span>
                        <span class="detail-value">{motif.evaluation:.2f} cp</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Novelty Pressure Ω:</span>
                        <span class="detail-value">{motif.novelty_pressure:.4f}</span>
                    </div>
                </div>
                <div>
                    <div class="detail-item">
                        <span class="detail-label">Move Sequence:</span>
                        <span class="detail-value">{' '.join(motif.move_sequence)}</span>
                    </div>
"""
            
            if motif.cortex_weights:
                html_content += f"""
                    <div class="detail-item">
                        <span class="detail-label">Cortex Activations:</span>
                        <span class="detail-value">
                            T:{motif.cortex_weights.get('tactical', 0):.2f} 
                            S:{motif.cortex_weights.get('strategic', 0):.2f} 
                            C:{motif.cortex_weights.get('conceptual', 0):.2f}
                        </span>
                    </div>
"""
            
            html_content += f"""
                </div>
            </div>
            <div class="fen-box">{motif.position_fen}</div>
            {f'<p style="color: #888; font-style: italic;">{motif.description}</p>' if motif.description else ''}
        </div>
"""
        
        html_content += """
    </div>
</body>
</html>
"""
        
        with open(filepath, 'w') as f:
            f.write(html_content)
