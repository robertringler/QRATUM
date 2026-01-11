#!/usr/bin/env python3
"""Demo script showing how to use BOB chess engine."""

import sys
import os

# Add BOB package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'kaggle_models', 'bob'))

from predict import predict, batch_predict


def demo_single_prediction():
    """Demo: Single position prediction."""
    print("\n" + "="*70)
    print("Demo 1: Single Position Prediction")
    print("="*70 + "\n")
    
    # Starting position
    input_data = {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "time_limit_ms": 1000,
        "depth": 15
    }
    
    print(f"Position: {input_data['fen']}")
    print(f"Time limit: {input_data['time_limit_ms']}ms")
    print(f"Max depth: {input_data['depth']}")
    print("\nSearching...")
    
    result = predict(input_data)
    
    print("\n✓ Search complete!")
    print(f"  Best move:   {result['move']}")
    print(f"  Evaluation:  {result['evaluation']:+.2f} pawns")
    print(f"  Depth:       {result['depth']} plies")
    print(f"  Nodes:       {result['nodes']:,}")
    print(f"  Time:        {result['time_ms']:.0f}ms")
    print(f"  NPS:         {result['nodes'] / (result['time_ms'] / 1000):,.0f}")
    print(f"  Engine:      {result['engine']} (Elo {result['elo']})")
    
    if result['pv']:
        print(f"  PV:          {' '.join(result['pv'][:5])}")


def demo_tactical_position():
    """Demo: Tactical puzzle."""
    print("\n" + "="*70)
    print("Demo 2: Tactical Position")
    print("="*70 + "\n")
    
    # Famous fork position
    input_data = {
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "time_limit_ms": 2000,
        "depth": 18
    }
    
    print("Italian Game position with tactical opportunity")
    print(f"FEN: {input_data['fen']}")
    print("\nSearching...")
    
    result = predict(input_data)
    
    print("\n✓ Search complete!")
    print(f"  Best move:   {result['move']}")
    print(f"  Evaluation:  {result['evaluation']:+.2f} pawns")
    print(f"  Depth:       {result['depth']} plies")
    print(f"  Time:        {result['time_ms']:.0f}ms")


def demo_batch_prediction():
    """Demo: Batch prediction."""
    print("\n" + "="*70)
    print("Demo 3: Batch Prediction")
    print("="*70 + "\n")
    
    positions = [
        {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "time_limit_ms": 500,
            "depth": 12
        },
        {
            "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "time_limit_ms": 500,
            "depth": 12
        },
        {
            "fen": "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
            "time_limit_ms": 500,
            "depth": 12
        }
    ]
    
    print(f"Processing {len(positions)} positions...")
    print()
    
    results = batch_predict(positions)
    
    for i, result in enumerate(results):
        print(f"Position {i+1}:")
        print(f"  Move:  {result['move']}")
        print(f"  Eval:  {result['evaluation']:+.2f}")
        print(f"  Time:  {result['time_ms']:.0f}ms")
    
    total_time = sum(r['time_ms'] for r in results)
    total_nodes = sum(r['nodes'] for r in results)
    print(f"\nTotal time: {total_time:.0f}ms")
    print(f"Total nodes: {total_nodes:,}")
    print(f"Avg NPS: {total_nodes / (total_time / 1000):,.0f}")


def demo_endgame():
    """Demo: Endgame position."""
    print("\n" + "="*70)
    print("Demo 4: Endgame Position")
    print("="*70 + "\n")
    
    # King and pawn endgame
    input_data = {
        "fen": "8/5k2/3p4/8/3P4/5K2/8/8 w - - 0 1",
        "time_limit_ms": 2000,
        "depth": 20
    }
    
    print("King and pawn endgame")
    print(f"FEN: {input_data['fen']}")
    print("\nSearching deep...")
    
    result = predict(input_data)
    
    print("\n✓ Search complete!")
    print(f"  Best move:   {result['move']}")
    print(f"  Evaluation:  {result['evaluation']:+.2f} pawns")
    print(f"  Depth:       {result['depth']} plies")
    print(f"  Time:        {result['time_ms']:.0f}ms")
    print(f"  Nodes:       {result['nodes']:,}")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("BOB Chess Engine - Demo Script")
    print("="*70)
    print("\nBOB achieves #1 on Kaggle Chess AI Benchmark with 1508 Elo")
    print("Using Asymmetric Adaptive Search + Multi-Agent Evaluation")
    print()
    
    try:
        demo_single_prediction()
        demo_tactical_position()
        demo_batch_prediction()
        demo_endgame()
        
        print("\n" + "="*70)
        print("All demos completed successfully!")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
