import torch
import model as M
import features
import halfka_v2_hm
import chess
import chess.engine
import subprocess
import numpy as np
import sys
import time
from typing import List, Tuple

# Test positions with expected characteristics
TEST_POSITIONS = [
    # (FEN, Description, Expected eval range)
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
     "Starting position", (-50, 50)),

    ("r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
     "Italian Game", (-100, 100)),

    ("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
     "Italian Game - Black to move", (-100, 100)),

    ("8/8/8/4k3/8/8/4K3/R7 w - - 0 1",
     "Rook endgame - White winning", (400, 800)),

    ("8/8/8/4k3/8/8/4K3/7r b - - 0 1",
     "Rook endgame - Black winning", (-800, -400)),

    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
     "Complex endgame", (-200, 200)),

    ("rnbqkb1r/pp1ppppp/5n2/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq c6 0 4",
     "Sicilian Defense", (-50, 150)),

    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
     "Complex middlegame", (0, 300)),
]


class SimpleFeatureExtractor:
    """Simplified feature extraction for HalfKAv2_hm - not exact but usable for testing"""

    @staticmethod
    def orient(is_white_pov: bool, sq: int, ksq: int) -> int:
        """Orient square based on king position and perspective"""
        kfile = ksq % 8
        flip_h = kfile < 4
        flip_v = not is_white_pov

        sq_rank = sq // 8
        sq_file = sq % 8

        if flip_h:
            sq_file = 7 - sq_file
        if flip_v:
            sq_rank = 7 - sq_rank

        return sq_rank * 8 + sq_file

    @staticmethod
    def get_piece_index(piece: chess.Piece, is_white_pov: bool) -> int:
        """Get piece type index for features"""
        # Piece encoding: pawn=0,1 knight=2,3 bishop=4,5 rook=6,7 queen=8,9 king=10
        p_idx = (piece.piece_type - 1) * 2 + (piece.color != is_white_pov)
        if p_idx == 11:  # Opponent king
            p_idx = 10  # Merge with our king
        return p_idx

    @staticmethod
    def get_king_bucket(ksq: int, is_white_pov: bool) -> int:
        """Get king bucket index"""
        # Simplified: use king square directly as bucket for e-h files
        kfile = ksq % 8
        if kfile < 4:
            ksq = ksq ^ 7  # Mirror horizontally

        if not is_white_pov:
            ksq = ksq ^ 56  # Mirror vertically

        # Map to 32 buckets (half board)
        return halfka_v2_hm.KingBuckets[ksq] if ksq < 64 else 0

    @staticmethod
    def extract_features(board: chess.Board) -> Tuple[torch.Tensor, ...]:
        """Extract features from a chess position"""
        batch_size = 1
        max_features = 32  # Maximum active features per side

        # Initialize tensors
        white_indices = torch.zeros((batch_size, max_features), dtype=torch.int32)
        white_values = torch.zeros((batch_size, max_features), dtype=torch.float32)
        black_indices = torch.zeros((batch_size, max_features), dtype=torch.int32)
        black_values = torch.zeros((batch_size, max_features), dtype=torch.float32)

        # Get king positions
        white_king_sq = board.king(chess.WHITE)
        black_king_sq = board.king(chess.BLACK)

        # Get buckets
        white_bucket = SimpleFeatureExtractor.get_king_bucket(white_king_sq, True)
        black_bucket = SimpleFeatureExtractor.get_king_bucket(black_king_sq, False)

        # Extract features for each perspective
        w_idx = 0
        b_idx = 0

        for sq, piece in board.piece_map().items():
            if w_idx < max_features:
                # White perspective
                oriented_sq = SimpleFeatureExtractor.orient(True, sq, white_king_sq)
                piece_idx = SimpleFeatureExtractor.get_piece_index(piece, True)
                feature_idx = (oriented_sq +
                               piece_idx * 64 +
                               white_bucket * halfka_v2_hm.NUM_PLANES_REAL)

                # Clamp to valid range
                feature_idx = min(feature_idx, halfka_v2_hm.NUM_INPUTS - 1)

                white_indices[0, w_idx] = feature_idx
                white_values[0, w_idx] = 1.0
                w_idx += 1

            if b_idx < max_features:
                # Black perspective
                oriented_sq = SimpleFeatureExtractor.orient(False, sq, black_king_sq)
                piece_idx = SimpleFeatureExtractor.get_piece_index(piece, False)
                feature_idx = (oriented_sq +
                               piece_idx * 64 +
                               black_bucket * halfka_v2_hm.NUM_PLANES_REAL)

                # Clamp to valid range
                feature_idx = min(feature_idx, halfka_v2_hm.NUM_INPUTS - 1)

                black_indices[0, b_idx] = feature_idx
                black_values[0, b_idx] = 1.0
                b_idx += 1

        # Perspective tensors
        us = torch.tensor([[1.0 if board.turn else 0.0]], dtype=torch.float32)
        them = 1.0 - us

        # Bucket indices
        psqt_indices = torch.tensor([white_bucket if board.turn else black_bucket],
                                    dtype=torch.int64)
        layer_stack_indices = torch.tensor([white_bucket if board.turn else black_bucket],
                                           dtype=torch.int64)

        return (us, them, white_indices, white_values,
                black_indices, black_values, psqt_indices, layer_stack_indices)


def get_stockfish_eval(fen: str, stockfish_path: str = "stockfish",
                       depth: int = 15) -> float:
    """Get Stockfish evaluation for a position"""
    try:
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            board = chess.Board(fen)
            info = engine.analyse(board, chess.engine.Limit(depth=depth))

            if "score" in info:
                score = info["score"].relative
                if score.is_mate():
                    # Return large value for mate
                    mate_in = score.mate()
                    return 10000.0 * (1 if mate_in > 0 else -1) / abs(mate_in)
                else:
                    return float(score.score())
            return 0.0
    except Exception as e:
        print(f"Error getting Stockfish eval: {e}")
        return None


def evaluate_with_model(model: M.NNUE, board: chess.Board, device: torch.device) -> float:
    """Evaluate position with KAN model"""
    with torch.no_grad():
        features = SimpleFeatureExtractor.extract_features(board)

        # Move to device
        features = tuple(f.to(device) if isinstance(f, torch.Tensor) else f
                         for f in features)

        us, them, w_idx, w_val, b_idx, b_val, psqt_idx, ls_idx = features
        output = model(us, them, w_idx, w_val, b_idx, b_val, psqt_idx, ls_idx)

        # Convert to centipawns
        eval_score = output.item() * model.nnue2score

        # Flip if black to move
        if not board.turn:
            eval_score = -eval_score

        return eval_score


def run_sanity_check(ckpt_path: str, stockfish_path: str = "stockfish",
                     use_simple_features: bool = False):
    """Run comprehensive sanity check"""

    print("=" * 80)
    print("NNUE KAN Model Sanity Check")
    print("=" * 80)

    # Load model
    print(f"\\n1. Loading model from {ckpt_path}...")
    feature_set = features.get_feature_set_from_name("HalfKAv2_hm^")
    model = M.NNUE(feature_set)

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"   Model loaded successfully on {device}")

    # Set idx_offset
    model.layer_stacks.idx_offset = torch.arange(
        0, 1 * model.num_ls_buckets, model.num_ls_buckets,
        device=device
    )

    # Test with dummy features first
    print("\\n2. Testing with dummy features...")
    batch_size = 1
    max_features = 32

    with torch.no_grad():
        us = torch.ones((batch_size, 1), device=device, dtype=torch.float32)
        them = torch.zeros((batch_size, 1), device=device, dtype=torch.float32)
        white_indices = torch.zeros((batch_size, max_features), dtype=torch.int32, device=device)
        white_values = torch.ones((batch_size, max_features), dtype=torch.float32, device=device)
        black_indices = torch.zeros((batch_size, max_features), dtype=torch.int32, device=device)
        black_values = torch.ones((batch_size, max_features), dtype=torch.float32, device=device)
        psqt_indices = torch.zeros((batch_size,), dtype=torch.int64, device=device)
        layer_stack_indices = torch.zeros((batch_size,), dtype=torch.int64, device=device)

        output = model(us, them, white_indices, white_values,
                       black_indices, black_values, psqt_indices, layer_stack_indices)
        dummy_eval = output.item() * 600
        print(f"   Dummy position evaluation: {dummy_eval:.0f} cp")

    # Test real positions
    print("\\n3. Testing real positions...")
    print("-" * 80)

    if use_simple_features:
        print("   Using simplified feature extraction (less accurate)")
    else:
        print("   Using random features (for basic testing only)")

    results = []

    for fen, description, expected_range in TEST_POSITIONS:
        board = chess.Board(fen)

        # Get model evaluation
        if use_simple_features:
            model_eval = evaluate_with_model(model, board, device)
        else:
            # Use random features for basic testing
            with torch.no_grad():
                batch_size = 1
                max_features = 32
                us = torch.tensor([[1.0 if board.turn else 0.0]], device=device, dtype=torch.float32)
                them = 1.0 - us

                # Random but deterministic features based on FEN
                seed = hash(fen) % 10000
                torch.manual_seed(seed)

                white_indices = torch.randint(0, halfka_v2_hm.NUM_INPUTS,
                                              (batch_size, max_features), dtype=torch.int32, device=device)
                white_values = torch.rand((batch_size, max_features), dtype=torch.float32, device=device)
                black_indices = torch.randint(0, halfka_v2_hm.NUM_INPUTS,
                                              (batch_size, max_features), dtype=torch.int32, device=device)
                black_values = torch.rand((batch_size, max_features), dtype=torch.float32, device=device)
                psqt_indices = torch.zeros((batch_size,), dtype=torch.int64, device=device)
                layer_stack_indices = torch.zeros((batch_size,), dtype=torch.int64, device=device)

                output = model(us, them, white_indices, white_values,
                               black_indices, black_values, psqt_indices, layer_stack_indices)
                model_eval = output.item() * model.nnue2score

        # Get Stockfish evaluation
        sf_eval = get_stockfish_eval(fen, stockfish_path)

        # Check if in expected range
        in_range = expected_range[0] <= model_eval <= expected_range[1]

        results.append({
            'description': description,
            'fen': fen,
            'model_eval': model_eval,
            'sf_eval': sf_eval,
            'expected_range': expected_range,
            'in_range': in_range
        })

        # Print results
        print(f"\\n   {description}:")
        print(f"   FEN: {fen[:50]}...")
        print(f"   Model eval:     {model_eval:7.0f} cp")
        if sf_eval is not None:
            print(f"   Stockfish eval: {sf_eval:7.0f} cp")
            print(f"   Difference:     {abs(model_eval - sf_eval):7.0f} cp")
        print(f"   Expected range: [{expected_range[0]:4.0f}, {expected_range[1]:4.0f}] cp")
        print(f"   In range:       {'✓' if in_range else '✗'}")

    # Summary
    print("\\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    in_range_count = sum(1 for r in results if r['in_range'])
    total_count = len(results)

    print(f"\\nPositions in expected range: {in_range_count}/{total_count}")

    if sf_eval is not None:
        valid_diffs = [abs(r['model_eval'] - r['sf_eval'])
                       for r in results if r['sf_eval'] is not None]
        if valid_diffs:
            avg_diff = sum(valid_diffs) / len(valid_diffs)
            max_diff = max(valid_diffs)
            print(f"Average difference from Stockfish: {avg_diff:.0f} cp")
            print(f"Maximum difference from Stockfish: {max_diff:.0f} cp")

    # Check if model produces varied output
    evals = [r['model_eval'] for r in results]
    eval_std = np.std(evals)
    print(f"\\nModel evaluation std dev: {eval_std:.0f} cp")

    if eval_std < 10:
        print("⚠️  WARNING: Model produces very similar evaluations for different positions!")
        print("   This might indicate an issue with the model or feature extraction.")
    else:
        print("✓  Model produces varied evaluations for different positions.")

    # Final verdict
    print("\\n" + "=" * 80)
    if in_range_count > total_count / 2:
        print("✓ BASIC SANITY CHECK PASSED")
    else:
        print("✗ SANITY CHECK SHOWS POTENTIAL ISSUES")

    if not use_simple_features:
        print("\\nNote: This test used random features. For more accurate testing,")
        print("      implement proper HalfKAv2_hm feature extraction or use")
        print("      the --simple-features flag for basic feature extraction.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Sanity check for KAN NNUE model")
    parser.add_argument("checkpoint", help="Path to .ckpt file")
    parser.add_argument("--stockfish", default="stockfish",
                        help="Path to Stockfish executable")
    parser.add_argument("--simple-features", action="store_true",
                        help="Use simplified feature extraction (more accurate than random)")

    args = parser.parse_args()

    run_sanity_check(args.checkpoint, args.stockfish, args.simple_features)


if __name__ == "__main__":
    main()