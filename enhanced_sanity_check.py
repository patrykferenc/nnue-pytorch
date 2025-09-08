import torch
import model as M
import halfka_v2_hm
import features
import chess
import chess.engine
import chess.pgn
import numpy as np
import pandas as pd
import json
import csv
import sys
import subprocess
import re
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterator
from dataclasses import dataclass
from nnue_dataset import make_sparse_batch_from_fens, destroy_sparse_batch, SparseBatch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class PositionResult:
    """Container for position evaluation results"""
    fen: str
    model_eval: float
    stockfish_eval: Optional[float]
    category: str = "unknown"
    source: str = "manual"
    game_phase: str = "unknown"
    material_balance: int = 0
    piece_count: int = 32


class PositionGenerator:
    """Generate chess positions for testing"""

    @staticmethod
    def from_opening_book(book_path: str, count: int = 100) -> List[str]:
        """Generate positions from opening book"""
        positions = []
        try:
            with open(book_path, 'r') as f:
                lines = f.readlines()
                for line in random.sample(lines, min(count, len(lines))):
                    if line.strip():
                        positions.append(line.strip())
        except FileNotFoundError:
            print(f"Warning: Opening book {book_path} not found")
        return positions

    @staticmethod
    def from_pgn_games(pgn_path: str, count: int = 100,
                       positions_per_game: int = 5) -> List[str]:
        """Extract positions from PGN games"""
        positions = []
        try:
            with open(pgn_path, 'r') as f:
                game_count = 0
                while len(positions) < count and game_count < count // positions_per_game:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break

                    board = game.board()
                    moves = list(game.mainline_moves())

                    if len(moves) > 10:  # Only games with sufficient moves
                        # Sample random positions from the game
                        sample_moves = random.sample(range(5, min(len(moves), 50)),
                                                     min(positions_per_game, len(moves) - 5))

                        temp_board = game.board()
                        for i, move in enumerate(moves):
                            temp_board.push(move)
                            if i in sample_moves:
                                positions.append(temp_board.fen())

                    game_count += 1
        except FileNotFoundError:
            print(f"Warning: PGN file {pgn_path} not found")
        except Exception as e:
            print(f"Warning: Error reading PGN: {e}")

        return positions[:count]

    @staticmethod
    def from_epd_file(epd_path: str, count: int = 100) -> List[str]:
        """Load positions from EPD file"""
        positions = []
        try:
            with open(epd_path, 'r') as f:
                lines = f.readlines()
                for line in random.sample(lines, min(count, len(lines))):
                    # Extract FEN from EPD (first part before operations)
                    fen_part = ' '.join(line.strip().split()[:6])
                    if fen_part:
                        positions.append(fen_part)
        except FileNotFoundError:
            print(f"Warning: EPD file {epd_path} not found")
        return positions

    @staticmethod
    def random_legal_positions(count: int = 100) -> List[str]:
        """Generate random legal positions"""
        positions = []
        for _ in range(count * 10):  # Try more to account for failures
            if len(positions) >= count:
                break

            try:
                # Start from a random opening
                board = chess.Board()

                # Make 10-30 random moves
                moves_count = random.randint(10, 30)
                for _ in range(moves_count):
                    legal_moves = list(board.legal_moves)
                    if not legal_moves or board.is_game_over():
                        break
                    board.push(random.choice(legal_moves))

                # Only add if position is valid and not game over
                if (not board.is_game_over() and
                        len(list(board.legal_moves)) > 0 and
                        not board.is_check()):
                    positions.append(board.fen())
            except:
                continue

        return positions[:count]


class PositionAnalyzer:
    """Analyze chess positions for categorization"""

    @staticmethod
    def analyze_position(fen: str) -> Dict[str, any]:
        """Analyze a position and return characteristics"""
        board = chess.Board(fen)

        # Count pieces
        piece_count = len(board.piece_map())
        white_material = sum(PositionAnalyzer._piece_values()[piece.piece_type]
                             for piece in board.piece_map().values() if piece.color)
        black_material = sum(PositionAnalyzer._piece_values()[piece.piece_type]
                             for piece in board.piece_map().values() if not piece.color)

        material_balance = white_material - black_material

        # Determine game phase
        if piece_count > 24:
            phase = "opening"
        elif piece_count > 12:
            phase = "middlegame"
        else:
            phase = "endgame"

        # Additional characteristics
        has_queens = any(piece.piece_type == chess.QUEEN for piece in board.piece_map().values())
        castling_rights = board.castling_rights != 0
        in_check = board.is_check()

        return {
            'piece_count': piece_count,
            'material_balance': material_balance,
            'game_phase': phase,
            'has_queens': has_queens,
            'castling_rights': castling_rights,
            'in_check': in_check
        }

    @staticmethod
    def _piece_values():
        return {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


class StatisticalAnalyzer:
    """Statistical analysis of evaluation results"""

    @staticmethod
    def correlation_analysis(results: List[PositionResult]) -> Dict:
        """Perform correlation analysis between model and Stockfish"""
        valid_results = [r for r in results if r.stockfish_eval is not None]

        if len(valid_results) < 2:
            return {"error": "Insufficient data for correlation analysis"}

        model_evals = [r.model_eval for r in valid_results]
        sf_evals = [r.stockfish_eval for r in valid_results]

        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(model_evals, sf_evals)

        # Spearman correlation (rank-based)
        spearman_r, spearman_p = stats.spearmanr(model_evals, sf_evals)

        # Kendall's tau
        kendall_tau, kendall_p = stats.kendalltau(model_evals, sf_evals)

        return {
            "pearson": {"r": pearson_r, "p_value": pearson_p},
            "spearman": {"r": spearman_r, "p_value": spearman_p},
            "kendall": {"tau": kendall_tau, "p_value": kendall_p},
            "sample_size": len(valid_results)
        }

    @staticmethod
    def error_analysis(results: List[PositionResult]) -> Dict:
        """Analyze errors between model and Stockfish"""
        valid_results = [r for r in results if r.stockfish_eval is not None]

        if not valid_results:
            return {"error": "No valid data for error analysis"}

        model_evals = np.array([r.model_eval for r in valid_results])
        sf_evals = np.array([r.stockfish_eval for r in valid_results])
        errors = model_evals - sf_evals
        abs_errors = np.abs(errors)

        return {
            "mae": np.mean(abs_errors),
            "mse": np.mean(errors ** 2),
            "rmse": np.sqrt(np.mean(errors ** 2)),
            "mean_error": np.mean(errors),
            "std_error": np.std(errors),
            "max_abs_error": np.max(abs_errors),
            "median_abs_error": np.median(abs_errors),
            "error_percentiles": {
                "p25": np.percentile(abs_errors, 25),
                "p75": np.percentile(abs_errors, 75),
                "p90": np.percentile(abs_errors, 90),
                "p95": np.percentile(abs_errors, 95),
                "p99": np.percentile(abs_errors, 99)
            }
        }

    @staticmethod
    def category_analysis(results: List[PositionResult]) -> Dict:
        """Analyze performance by position categories"""
        analysis = {}

        # Group by categories
        categories = {}
        for result in results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)

        for category, cat_results in categories.items():
            valid_results = [r for r in cat_results if r.stockfish_eval is not None]

            if valid_results:
                model_evals = [r.model_eval for r in valid_results]
                sf_evals = [r.stockfish_eval for r in valid_results]
                errors = np.array(model_evals) - np.array(sf_evals)

                analysis[category] = {
                    "count": len(cat_results),
                    "valid_count": len(valid_results),
                    "mae": np.mean(np.abs(errors)) if len(errors) > 0 else None,
                    "correlation": stats.pearsonr(model_evals, sf_evals)[0] if len(valid_results) > 1 else None
                }

        return analysis


class ScientificSanityCheck:
    """Main class for scientific NNUE evaluation testing"""

    def __init__(self, model_path: str, stockfish_path: str = "stockfish",
                 device: str = "auto"):
        self.model_path = model_path
        self.stockfish_path = stockfish_path

        # Device selection
        if device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load model
        self.feature_set = features.get_feature_set_from_name("HalfKAv2_hm^")
        self.model = M.NNUE(self.feature_set)

        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.model = self.model.to(self.device)

        # Initialize idx_offset for single position evaluation
        self.model.layer_stacks.idx_offset = torch.zeros((1,), dtype=torch.int64, device=self.device)

    def evaluate_position(self, fen: str) -> float:
        """Evaluate position with KAN model"""
        self.model.eval()
        with torch.no_grad():
            # Reset idx_offset for single batch
            self.model.layer_stacks.idx_offset = torch.zeros((1,), dtype=torch.int64, device=self.device)

            # Get features
            features = self._get_features_from_fen(fen)
            us, them, w_idx, w_val, b_idx, b_val, outcome, score_t, psqt_idx, ls_idx = features

            # Forward pass
            output = self.model(us, them, w_idx, w_val, b_idx, b_val, psqt_idx, ls_idx)
            eval_score = output.item() * self.model.nnue2score

            # Adjust for side to move
            board = chess.Board(fen)
            if not board.turn:
                eval_score = -eval_score

            return eval_score

    def _get_features_from_fen(self, fen: str):
        """Extract features from FEN using C++ data loader"""
        batch_ptr = make_sparse_batch_from_fens(
            self.feature_set, [fen], [0], [0], [1]
        )

        if not batch_ptr:
            raise ValueError(f"Failed to create batch from FEN: {fen}")

        try:
            tensors = batch_ptr.contents.get_tensors(self.device)
            return tensors
        finally:
            destroy_sparse_batch(batch_ptr)

    def get_stockfish_eval(self, fen: str) -> Optional[float]:
        """Get Stockfish NNUE evaluation"""
        try:
            engine = subprocess.Popen(
                self.stockfish_path,
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
                if 'uciok' in line.strip():
                    break

            engine.stdin.write('isready\n')
            for line in iter(engine.stdout.readline, ''):
                if 'readyok' in line.strip():
                    break

            # Evaluate position
            engine.stdin.write(f'position fen {fen}\n')
            engine.stdin.write('eval\n')

            nnue_eval = None
            for line in iter(engine.stdout.readline, ''):
                line = line.strip()
                # print(line)
                if 'NNUE evaluation' in line and 'info string' not in line:
                    value = line.split('NNUE evaluation')[1]
                    value = value.split('(white side)')[0].strip()
                    nnue_eval = float(value) * 100
                    break

            engine.stdin.write('quit\n')
            engine.terminate()

            # Adjust for side to move
            board = chess.Board(fen)
            if not board.turn:
                nnue_eval = -nnue_eval

            return nnue_eval

        except Exception as e:
            print("got exception:", e)
            return None

    def run_test_suite(self, positions: List[str],
                       position_source: str = "manual") -> List[PositionResult]:
        """Run evaluation test on a set of positions"""
        results = []

        for i, fen in enumerate(positions):
            try:
                # Analyze position characteristics
                pos_analysis = PositionAnalyzer.analyze_position(fen)
                # print("analysing position #{}: {}, fen: {}".format(i, pos_analysis, fen))

                # Evaluate with model
                model_eval = self.evaluate_position(fen)
                # print("after model eval")

                # Evaluate with Stockfish
                sf_eval = self.get_stockfish_eval(fen)
                # print("after stockfish eval")

                result = PositionResult(
                    fen=fen,
                    model_eval=model_eval,
                    stockfish_eval=sf_eval,
                    category=pos_analysis['game_phase'],
                    source=position_source,
                    game_phase=pos_analysis['game_phase'],
                    material_balance=pos_analysis['material_balance'],
                    piece_count=pos_analysis['piece_count']
                )
                # print("after position result")

                results.append(result)
                # print("after results append")

                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1}/{len(positions)} positions")

            except Exception as e:
                print(f"Error processing position {i}: {e}")
                continue

        return results

    def export_results(self, results: List[PositionResult],
                       output_file: str, format: str = "csv"):
        """Export results to file"""
        if format.lower() == "csv":
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['fen', 'model_eval', 'stockfish_eval', 'category',
                                 'source', 'game_phase', 'material_balance', 'piece_count'])

                for result in results:
                    writer.writerow([
                        result.fen, result.model_eval, result.stockfish_eval,
                        result.category, result.source, result.game_phase,
                        result.material_balance, result.piece_count
                    ])

        elif format.lower() == "json":
            with open(output_file, 'w') as f:
                json.dump([result.__dict__ for result in results], f, indent=2)

    def generate_report(self, results: List[PositionResult]) -> Dict:
        """Generate comprehensive statistical report"""
        # Basic statistics
        total_positions = len(results)
        successful_evaluations = sum(1 for r in results if r.stockfish_eval is not None)

        # Statistical analyses
        correlation_stats = StatisticalAnalyzer.correlation_analysis(results)
        error_stats = StatisticalAnalyzer.error_analysis(results)
        category_stats = StatisticalAnalyzer.category_analysis(results)

        report = {
            "summary": {
                "total_positions": total_positions,
                "successful_evaluations": successful_evaluations,
                "success_rate": successful_evaluations / total_positions if total_positions > 0 else 0
            },
            "correlation_analysis": correlation_stats,
            "error_analysis": error_stats,
            "category_analysis": category_stats,
            "model_statistics": {
                "mean_evaluation": np.mean([r.model_eval for r in results]),
                "std_evaluation": np.std([r.model_eval for r in results]),
                "evaluation_range": {
                    "min": min([r.model_eval for r in results]),
                    "max": max([r.model_eval for r in results])
                }
            }
        }

        return report

    def print_scientific_summary(self, report: Dict):
        """Print concise scientific summary"""
        print("=" * 60)
        print("NNUE MODEL EVALUATION ANALYSIS")
        print("=" * 60)

        summary = report["summary"]
        print(f"Positions evaluated: {summary['total_positions']}")
        print(f"Successful comparisons: {summary['successful_evaluations']}")
        print(f"Success rate: {summary['success_rate']:.3f}")

        if "correlation_analysis" in report and "pearson" in report["correlation_analysis"]:
            corr = report["correlation_analysis"]["pearson"]
            print(f"Pearson correlation: {corr['r']:.4f} (p={corr['p_value']:.2e})")

        if "error_analysis" in report and "mae" in report["error_analysis"]:
            error = report["error_analysis"]
            print(f"Mean absolute error: {error['mae']:.1f} cp")
            print(f"Root mean squared error: {error['rmse']:.1f} cp")
            print(f"Median absolute error: {error['median_abs_error']:.1f} cp")

        if "category_analysis" in report:
            print("\nPerformance by game phase:")
            for phase, stats in report["category_analysis"].items():
                if stats["mae"] is not None:
                    correlation_str = f"{stats['correlation']:.3f}" if stats['correlation'] is not None else "N/A"
                    print(f"  {phase}: MAE={stats['mae']:.1f} cp, r={correlation_str}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Scientific NNUE Model Evaluation")
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument("--stockfish", default="stockfish", help="Stockfish executable path")
    parser.add_argument("--positions", type=int, default=1000, help="Number of positions to test")
    parser.add_argument("--pgn", help="PGN file for position extraction")
    parser.add_argument("--epd", help="EPD file for positions")
    parser.add_argument("--opening-book", help="Opening book file")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--format", choices=["csv", "json"], default="csv", help="Output format")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--report", help="Generate detailed JSON report")

    args = parser.parse_args()

    # Initialize checker
    device = "cpu" if args.cpu else "auto"
    checker = ScientificSanityCheck(args.checkpoint, args.stockfish, device)

    # Generate positions
    positions = []

    if args.pgn:
        print(f"Extracting positions from PGN: {args.pgn}")
        positions.extend(PositionGenerator.from_pgn_games(args.pgn, args.positions // 2))

    if args.epd:
        print(f"Loading positions from EPD: {args.epd}")
        positions.extend(PositionGenerator.from_epd_file(args.epd, args.positions // 2))

    if args.opening_book:
        print(f"Loading positions from opening book: {args.opening_book}")
        positions.extend(PositionGenerator.from_opening_book(args.opening_book, args.positions // 3))

    # Fill remaining with random positions
    remaining = args.positions - len(positions)
    if remaining > 0:
        print(f"Generating {remaining} random positions")
        positions.extend(PositionGenerator.random_legal_positions(remaining))

    positions = positions[:args.positions]  # Trim to requested count

    print(f"Running evaluation on {len(positions)} positions...")

    # Run test suite
    results = checker.run_test_suite(positions, "mixed")

    # Generate report
    report = checker.generate_report(results)

    # Print summary
    checker.print_scientific_summary(report)

    # Export results
    if args.output:
        checker.export_results(results, args.output, args.format)
        print(f"Results exported to: {args.output}")

    # Export detailed report
    if args.report:
        with open(args.report, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Detailed report saved to: {args.report}")


if __name__ == "__main__":
    main()