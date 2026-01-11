"""
BOB Chess Engine - Kaggle Model Inference Endpoint
"""
import json
from typing import Dict, Any
from engine.bob_engine import BOBEngine

# Initialize BOB engine globally (loaded once)
engine = BOBEngine(
    name="BOB",
    elo=1508,
    max_depth=20,
    use_multi_agent=True
)

def predict(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Kaggle Model API: Predict best chess move
    
    Args:
        input_data: {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "time_limit_ms": 1000,  # optional
            "depth": 20              # optional
        }
    
    Returns:
        {
            "move": "e2e4",
            "evaluation": 0.25,
            "depth": 20,
            "nodes": 1234567,
            "time_ms": 850,
            "pv": ["e2e4", "e7e5", "g1f3"],
            "engine": "BOB",
            "elo": 1508
        }
    """
    fen = input_data.get("fen")
    time_limit = input_data.get("time_limit_ms", 1000)
    depth = input_data.get("depth", 20)
    
    # Search for best move
    result = engine.search(
        fen=fen,
        max_depth=depth,
        time_limit_ms=time_limit
    )
    
    return {
        "move": result["best_move"],
        "evaluation": result["score"],
        "depth": result["depth"],
        "nodes": result["nodes_searched"],
        "time_ms": result["time_ms"],
        "pv": result.get("principal_variation", []),
        "engine": "BOB",
        "elo": 1508,
        "version": "1.0.0"
    }

def batch_predict(input_batch: list) -> list:
    """Batch inference for multiple positions"""
    return [predict(item) for item in input_batch]
