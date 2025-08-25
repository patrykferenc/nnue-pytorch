import torch
import model as M
import halfka_v2_hm
import features
import chess
import chess.engine
import numpy as np
import sys
import ctypes
import subprocess
import re
import warnings
from nnue_dataset import make_sparse_batch_from_fens, destroy_sparse_batch, SparseBatch

# Test positions with expected characteristics
# Extended test positions with expected characteristics
TEST_POSITIONS = [
    # ========== OPENING POSITIONS ==========
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
     "Starting position", (-50, 50), 0, 0, 1),

    ("r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
     "Italian Game", (-100, 100), 50, 8, 1),

    ("rnbqkb1r/pp1ppppp/5n2/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq c6 0 4",
     "Sicilian Defense", (-50, 150), 30, 8, 1),

    ("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
     "Sicilian Defense - Move 2", (-50, 100), 20, 2, 1),

    ("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
     "Italian Game - Black to move", (-100, 100), -50, 7, 1),

    ("rnbqkb1r/ppp1pppp/3p1n2/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 4",
     "Pirc Defense", (0, 200), 100, 8, 1),

    ("rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
     "Italian Game - Two Knights", (-50, 50), 0, 8, 1),

    ("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
     "Ruy Lopez", (0, 100), 50, 6, 1),

    ("rnbqkb1r/pp2pppp/3p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq - 0 6",
     "King's Indian Attack", (-50, 150), 75, 12, 1),

    # ========== MIDDLEGAME POSITIONS ==========
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
     "Complex middlegame - Perft position", (0, 300), 150, 20, 2),

    ("r2q1rk1/ppp2ppp/2n1bn2/2bpp3/3P4/2N1PN2/PPP1BPPP/R1BQK2R w KQ - 0 9",
     "Closed center middlegame", (-100, 100), 0, 18, 1),

    ("r1bq1rk1/pp2ppbp/2np1np1/8/3PP3/2N2N2/PPP1BPPP/R1BQ1RK1 w - - 0 10",
     "King's Indian middlegame", (-50, 150), 50, 20, 1),

    ("r1b1k2r/ppppqppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 7",
     "Open center with development", (-100, 100), 0, 14, 1),

    ("r2qkb1r/1b1n1ppp/p3pn2/1p6/3NP3/2N1B3/PPP1BPPP/R2QK2R w KQkq - 0 11",
     "Sicilian Najdorf structure", (100, 300), 200, 22, 2),

    ("r1bqr1k1/pp1nbppp/2p2n2/3p2B1/3P4/2N1PN2/PPP2PPP/R2QRBK1 w - - 5 11",
     "Symmetrical pawn structure", (-50, 50), 0, 22, 1),

    # ========== TACTICAL POSITIONS ==========
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 0 5",
     "Tactical position - pins", (100, 300), 200, 10, 2),

    ("r3k2r/ppp2pp1/2n1b2p/3p4/3P2Pq/2P1PN2/PP3P1P/R2QKB1R w KQkq - 0 12",
     "Queen on the prowl", (-300, -100), -200, 24, 0),

    ("2kr1b1r/pp1npppp/2p1bn2/q7/3P4/2N1BN2/PPP1BPPP/R2Q1RK1 w - - 3 10",
     "Queen sortie early", (200, 400), 300, 20, 2),

    ("r1b2rk1/2q1bppp/p1n1pn2/1p6/3P4/1BN1PN2/PP2QPPP/R1B2RK1 w - - 0 12",
     "Piece coordination", (0, 200), 100, 24, 1),

    # ========== ENDGAME POSITIONS ==========
    ("8/8/8/4k3/8/8/4K3/R7 w - - 0 1",
     "Rook endgame - White winning", (400, 800), 600, 80, 2),

    ("8/8/8/4k3/8/8/4K3/7r b - - 0 1",
     "Rook endgame - Black winning", (-800, -400), -600, 80, 0),

    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
     "Complex rook endgame", (-200, 200), 0, 60, 1),

    ("8/8/4k3/8/2K5/8/5P2/8 w - - 0 1",
     "King and pawn vs King", (300, 500), 400, 90, 2),

    ("8/5pk1/8/5P1K/8/8/8/8 w - - 0 1",
     "Pawn race", (0, 200), 100, 85, 1),

    ("8/8/1p6/p1p5/P1P5/1P6/8/4K2k w - - 0 1",
     "Pawn endgame - symmetrical", (-50, 50), 0, 70, 1),

    ("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
     "Basic King and pawn", (0, 100), 50, 95, 1),

    ("8/2k5/8/2K5/2P5/8/8/8 w - - 0 1",
     "King and pawn - critical squares", (800, 1200), 1000, 92, 2),

    ("6k1/5p2/6p1/8/7P/6P1/5P2/6K1 w - - 0 1",
     "Symmetrical pawn endgame", (-50, 50), 0, 75, 1),

    ("8/8/3k4/3p4/3P4/3K4/8/8 w - - 0 1",
     "King and pawn opposition", (-50, 50), 0, 88, 1),

    # ========== PIECE ENDGAMES ==========
    ("8/8/4k3/8/4N3/4K3/8/8 w - - 0 1",
     "Knight vs lone king - draw", (-50, 50), 0, 95, 1),

    ("8/8/8/3k4/8/3K4/3B4/8 w - - 0 1",
     "Bishop vs lone king - draw", (-50, 50), 0, 95, 1),

    ("8/8/3kb3/8/3KB3/8/8/8 w - - 0 1",
     "Bishop vs bishop - same color", (-50, 50), 0, 90, 1),

    ("8/8/3k4/8/3KB3/8/8/7b b - - 0 1",
     "Bishop vs bishop - opposite color", (-50, 50), 0, 90, 1),

    ("r7/8/8/8/8/8/8/R3K2k b - - 0 1",
     "Rook endgame - basic", (-50, 50), 0, 82, 1),

    ("8/8/8/4k3/8/8/8/R3K3 w - - 0 1",
     "Rook vs lone king", (900, 1100), 1000, 85, 2),

    # ========== QUEEN ENDGAMES ==========
    ("8/8/4k3/8/8/8/4K3/Q7 w - - 0 1",
     "Queen vs lone king", (900, 1100), 1000, 85, 2),

    ("8/8/4k3/8/8/8/q3K3/8 b - - 0 1",
     "Queen vs lone king - Black", (-1100, -900), -1000, 85, 0),

    ("8/6k1/8/8/8/8/Q7/7K w - - 0 1",
     "Queen endgame - winning", (800, 1200), 1000, 78, 2),

    # ========== IMBALANCED POSITIONS ==========
    ("rnb1kbnr/pppp1ppp/8/4p3/4P2q/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 4",
     "Queen out early", (200, 400), 300, 8, 2),

    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
     "Rooks and kings only", (-50, 50), 0, 60, 1),

    ("rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
     "Symmetrical development", (-50, 50), 0, 8, 1),

    ("8/2p5/3p4/KP6/2b3r1/8/4P3/5k2 b - - 0 1",
     "Minor pieces vs pawns", (-300, -100), -200, 65, 0),

    ("r1bqkbnr/ppp2ppp/2n5/3pp3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
     "Center tension", (0, 100), 50, 8, 1),

    # ========== SPECIAL/FORTRESS POSITIONS ==========
    ("8/8/4kpp1/3p4/3P4/4KPP1/8/8 w - - 0 1",
     "Fortress position", (-50, 50), 0, 85, 1),

    ("8/p7/8/1P6/K7/8/k7/8 w - - 0 1",
     "Tempo critical position", (-200, 200), 0, 90, 1),

    ("k7/8/KP6/8/8/8/8/8 w - - 0 1",
     "King and pawn - winning", (900, 1100), 1000, 92, 2),

    ("8/8/8/8/4k3/8/8/R3K2R w KQ - 0 1",
     "Castling rights matter", (400, 600), 500, 75, 2),

    # ========== COMPLEX POSITIONS ==========
    ("r3r1k1/pp3pbp/1qp2np1/8/2BP4/2N1PN2/PP2QPPP/R4RK1 w - - 0 15",
     "Complex middlegame with pressure", (100, 300), 200, 30, 2),

    ("2rq1rk1/pb1nbppp/1p2pn2/8/2BNP3/2N1BP2/PPP3PP/R2Q1RK1 w - - 0 13",
     "Piece coordination test", (0, 200), 100, 26, 1),

    ("r1b1kb1r/pp1n1ppp/2p1pn2/q7/3P4/2N1BN2/PPP1BPPP/R2Q1RK1 b kq - 5 9",
     "Queen sortie evaluation", (200, 400), 300, 18, 2),

    ("r2q1rk1/1b1nbppp/p3pn2/1p6/3PP3/1BN1BN2/PP3PPP/R2Q1RK1 w - - 0 12",
     "Closed Ruy Lopez structure", (0, 150), 75, 24, 1),

    ("rnbq1rk1/pp2ppbp/3p1np1/8/3PP3/2N2N2/PPP1BPPP/R1BQ1RK1 b - - 0 8",
     "King's Indian Defense", (-50, 100), 25, 16, 1),

    # ========== MATERIAL IMBALANCE ==========
    ("8/8/4k3/8/8/3BK3/8/8 w - - 0 1",
     "Bishop vs nothing", (200, 400), 300, 95, 2),

    ("8/8/4k3/8/8/3NK3/8/8 w - - 0 1",
     "Knight vs nothing", (200, 400), 300, 95, 2),

    ("8/8/3pk3/8/3PK3/3B4/8/8 w - - 0 1",
     "Bishop and pawn vs pawn", (100, 300), 200, 88, 2),

    ("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 4",
     "Four knights game", (-50, 50), 0, 8, 1),
]


def get_features_from_fen(feature_set, fen, device, score=0, ply=0, result=1):
    """Use the C++ data loader to get proper features from FEN"""
    batch_ptr = make_sparse_batch_from_fens(
        feature_set,
        [fen],  # List of FENs
        [score],  # List of scores
        [ply],  # List of plies
        [result]  # List of results
    )

    if not batch_ptr:
        raise ValueError(f"Failed to create batch from FEN: {fen}")

    try:
        # Use the built-in get_tensors method which handles conversion properly
        tensors = batch_ptr.contents.get_tensors(device)
        return tensors

    finally:
        destroy_sparse_batch(batch_ptr)


def evaluate_position_with_model(model, feature_set, fen, device, score=0, ply=0, result=1):
    """Evaluate a position using the KAN model with proper feature extraction and error handling"""
    try:
        # Ensure model is in eval mode and on correct device
        model.eval()
        model = model.to(device)

        # CRITICAL FIX: Set idx_offset correctly for single position evaluation
        with torch.no_grad():
            # Reset idx_offset for single batch evaluation
            model.layer_stacks.idx_offset = torch.zeros((1,), dtype=torch.int64, device=device)

            # Get features using the C++ data loader with proper device handling
            features = get_features_from_fen(feature_set, fen, device, score, ply, result)

            # Unpack the features - they're already on the correct device
            us, them, w_idx, w_val, b_idx, b_val, outcome, score_t, psqt_idx, ls_idx = features

            # Ensure all tensors are on the correct device and properly formatted
            us = us.to(device)
            them = them.to(device)
            w_idx = w_idx.to(device)
            w_val = w_val.to(device)
            b_idx = b_idx.to(device)
            b_val = b_val.to(device)
            psqt_idx = psqt_idx.to(device)
            ls_idx = ls_idx.to(device)

            # Synchronize CUDA before forward pass
            if device.type == 'cuda':
                torch.cuda.synchronize(device)

            # The tensors are already properly formatted, just do the forward pass
            output = model(us, them, w_idx, w_val, b_idx, b_val, psqt_idx, ls_idx)

            # Convert to centipawns
            eval_score = output.item() * model.nnue2score

            # Adjust for side to move
            board = chess.Board(fen)
            if not board.turn:
                eval_score = -eval_score

            return eval_score

    except RuntimeError as e:
        if "CUDA" in str(e) and device.type == 'cuda':
            print(f"CUDA error encountered, falling back to CPU: {e}")
            # Retry on CPU
            cpu_device = torch.device('cpu')
            return evaluate_position_with_model(model.to(cpu_device), feature_set, fen, cpu_device, score, ply, result)
        else:
            raise


def get_stockfish_eval(fen: str, stockfish_path: str = "stockfish") -> float:
    """Get Stockfish NNUE static evaluation for a position"""
    try:
        # Start Stockfish process
        engine = subprocess.Popen(
            stockfish_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )

        # Initialize UCI
        engine.stdin.write('uci\n')
        for line in iter(engine.stdout.readline, ''):
            line = line.strip()
            if 'uciok' in line:
                break

        engine.stdin.write('isready\n')
        for line in iter(engine.stdout.readline, ''):
            line = line.strip()
            if 'readyok' in line:
                break

        # Set position and get evaluation
        engine.stdin.write(f'position fen {fen}\n')
        engine.stdin.write('eval\n')

        nnue_eval = None
        for line in iter(engine.stdout.readline, ''):
            line = line.strip()

            # Look for NNUE evaluation line
            if 'NNUE evaluation' in line and 'info string' not in line:
                # Parse: "NNUE evaluation        +0.45 (white side)"
                value = line.split('NNUE evaluation')[1]
                value = value.split('(white side)')[0].strip()
                nnue_eval = float(value) * 100  # Convert to centipawns
                break

        engine.stdin.write('quit\n')
        engine.terminate()

        # Adjust for side to move (NNUE eval is always from white's perspective)
        board = chess.Board(fen)
        if not board.turn:  # If black to move
            nnue_eval = -nnue_eval

        return nnue_eval

    except Exception as e:
        print(f"Warning: Could not get Stockfish NNUE eval: {e}")
        return None


def run_sanity_check(ckpt_path: str, stockfish_path: str = "stockfish", use_cpu: bool = False):
    """Run comprehensive sanity check with proper feature extraction and error handling"""
    warnings.filterwarnings("ignore", message="std\\(\\): degrees of freedom is <= 0")

    print("=" * 80)
    print("NNUE KAN Model Sanity Check")
    print("=" * 80)

    # Load model
    print(f"\n1. Loading model from {ckpt_path}...")
    feature_set = features.get_feature_set_from_name("HalfKAv2_hm^")
    model = M.NNUE(feature_set)

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Device selection with fallback
    if use_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f" Moving model to {device}...")

    try:
        model = model.to(device)

        # CRITICAL FIX: Proper idx_offset initialization for single position evaluation
        model.layer_stacks.idx_offset = torch.zeros((1,), dtype=torch.int64, device=device)

        # Test CUDA memory allocation
        if device.type == 'cuda':
            torch.cuda.empty_cache()  # Clear cache
            torch.cuda.synchronize(device)  # Ensure synchronization

        print(f" Model loaded successfully on {device}")

    except Exception as e:
        print(f" Error loading model on {device}: {e}")
        if not use_cpu and device.type == 'cuda':
            print("\n Falling back to CPU...")
            return run_sanity_check(ckpt_path, stockfish_path, use_cpu=True)
        else:
            raise

    # Test with starting position first with proper error handling
    print("\n2. Testing with starting position...")
    try:
        start_eval = evaluate_position_with_model(
            model, feature_set,
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            device
        )
        print(f" Starting position evaluation: {start_eval:.0f} cp")
        print(" ✓ Model can process positions with proper features")
    except Exception as e:
        print(f" ✗ Error evaluating starting position: {e}")
        # Try CPU if CUDA failed
        if not use_cpu and 'CUDA' in str(e):
            print("\n Retrying with CPU...")
            return run_sanity_check(ckpt_path, stockfish_path, use_cpu=True)
        else:
            import traceback
            traceback.print_exc()
            return

    # Test all positions
    print("\\n3. Testing various positions...")
    print("-" * 80)

    results = []

    for fen, description, expected_range, score, ply, result in TEST_POSITIONS:
        try:
            print(f"\\n   Testing: {description}")

            # Get model evaluation
            model_eval = evaluate_position_with_model(
                model, feature_set, fen, device, score, ply, result
            )

            # Get Stockfish evaluation if available
            sf_eval = get_stockfish_eval(fen, stockfish_path)

            results.append({
                'description': description,
                'fen': fen,
                'model_eval': model_eval,
                'sf_eval': sf_eval,
                'expected_range': expected_range,
                'in_range': expected_range[0] <= model_eval <= expected_range[1]
            })

            # Print results
            print(f"   Model eval:     {model_eval:7.0f} cp")
            if sf_eval is not None:
                print(f"   Stockfish eval: {sf_eval:7.0f} cp")
                diff = abs(model_eval - sf_eval)
                print(f"   Difference:     {diff:7.0f} cp")

                # Color code the difference
                if diff < 100:
                    diff_status = "✓ Excellent"
                elif diff < 300:
                    diff_status = "⚠ Good"
                elif diff < 500:
                    diff_status = "⚠ Moderate"
                else:
                    diff_status = "✗ Large"
                print(f"   Diff status:    {diff_status}")

            print(f"   Expected range: [{expected_range[0]:4.0f}, {expected_range[1]:4.0f}] cp")
            print(f"   In range:       {'✓' if results[-1]['in_range'] else '✗'}")

        except Exception as e:
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    successful_evals = [r for r in results if r['model_eval'] is not None]

    if not successful_evals:
        print("✗ No positions evaluated successfully")
        return

    in_range_count = sum(1 for r in successful_evals if r['in_range'])
    total_count = len(successful_evals)

    print(f"\\nPositions evaluated: {total_count}/{len(TEST_POSITIONS)}")
    print(f"Positions in expected range: {in_range_count}/{total_count}")

    # Statistics vs Stockfish
    sf_comparisons = [(r['model_eval'], r['sf_eval'])
                      for r in successful_evals if r['sf_eval'] is not None]

    if sf_comparisons:
        diffs = [abs(m - s) for m, s in sf_comparisons]
        avg_diff = sum(diffs) / len(diffs)
        max_diff = max(diffs)
        min_diff = min(diffs)

        print(f"\\nComparison with Stockfish ({len(sf_comparisons)} positions):")
        print(f"  Average difference: {avg_diff:6.0f} cp")
        print(f"  Maximum difference: {max_diff:6.0f} cp")
        print(f"  Minimum difference: {min_diff:6.0f} cp")

        # Correlation
        if len(sf_comparisons) > 1:
            model_evals = [m for m, s in sf_comparisons]
            sf_evals = [s for m, s in sf_comparisons]
            correlation = np.corrcoef(model_evals, sf_evals)[0, 1]
            print(f"  Correlation:        {correlation:6.3f}")

    # Check evaluation variance
    evals = [r['model_eval'] for r in successful_evals]
    if evals:
        eval_std = np.std(evals)
        eval_mean = np.mean(evals)

        print(f"\\nModel evaluation statistics:")
        print(f"  Mean:               {eval_mean:6.0f} cp")
        print(f"  Std deviation:      {eval_std:6.0f} cp")

        if eval_std < 10:
            print("  ⚠️  WARNING: Very low variance - model may not be discriminating positions well")
        elif eval_std < 50:
            print("  ⚠️  Low variance - model discrimination could be better")
        else:
            print("  ✓  Good variance in evaluations")

    # Final verdict
    print("\\n" + "=" * 80)

    issues = []
    if total_count < len(TEST_POSITIONS):
        issues.append("Some positions failed to evaluate")
    if total_count > 0 and in_range_count < total_count * 0.5:
        issues.append("Many evaluations outside expected ranges")
    if evals and eval_std < 50:
        issues.append("Low evaluation variance")
    if sf_comparisons and avg_diff > 500:
        issues.append("Large average difference from Stockfish")

    if not issues:
        print("✓ SANITY CHECK PASSED")
        print("  The model appears to be working correctly")
    else:
        print("⚠️  SANITY CHECK COMPLETED WITH ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
        print("\\n  The model is functional but may need more training or tuning")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Sanity check for KAN NNUE model")
    parser.add_argument("checkpoint", help="Path to .ckpt file")
    parser.add_argument("--stockfish", default="stockfish",
                        help="Path to Stockfish executable")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU usage")

    args = parser.parse_args()

    # Check data loader
    import glob
    if not glob.glob("./*training_data_loader.*"):
        print("ERROR: Cannot find training_data_loader library!")
        print("Please compile it first using:")
        print("  bash compile_data_loader.sh")
        print("or")
        print("  compile_data_loader.bat (on Windows)")
        sys.exit(1)

    run_sanity_check(args.checkpoint, args.stockfish, use_cpu=args.cpu)


if __name__ == "__main__":
    main()