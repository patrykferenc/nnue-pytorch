#!/usr/bin/env python3
"""
Test PyTorch NNUE models against Stockfish and calculate ELO differences.
This script loads checkpoint files, runs games using c-chess-cli, and analyzes results with ordo.
"""

import os
import sys
import subprocess
import argparse
import json
import time
import shutil
import math
import re
import stat
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages discovery and loading of checkpoint files"""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.checkpoints = []

    def find_checkpoints(self) -> List[Path]:
        """Find all .ckpt files in directory tree"""
        ckpt_files = list(self.root_dir.rglob("*.ckpt"))

        # Filter to only epoch checkpoints
        epoch_pattern = re.compile(r"epoch=(\d+)")
        filtered_ckpts = []

        for ckpt in ckpt_files:
            if epoch_pattern.search(str(ckpt)):
                filtered_ckpts.append(ckpt)

        # Sort by epoch number
        def get_epoch(path):
            match = epoch_pattern.search(str(path))
            return int(match.group(1)) if match else 0

        filtered_ckpts.sort(key=get_epoch)
        self.checkpoints = filtered_ckpts

        logger.info(f"Found {len(self.checkpoints)} checkpoint files")
        for ckpt in self.checkpoints:
            logger.info(f"  - {ckpt.name}")

        return self.checkpoints

    def get_checkpoint_info(self, ckpt_path: Path) -> Dict:
        """Extract metadata from checkpoint path"""
        epoch_match = re.search(r"epoch=(\d+)", str(ckpt_path))
        run_match = re.search(r"run_(\d+)", str(ckpt_path))

        return {
            "path": str(ckpt_path),
            "name": ckpt_path.name,
            "epoch": int(epoch_match.group(1)) if epoch_match else 0,
            "run_id": int(run_match.group(1)) if run_match else 0
        }


class GameRunner:
    """Runs games between engines using c-chess-cli"""

    def __init__(self, c_chess_cli_path: str, stockfish_path: str,
                 book_file: str, output_dir: str):
        self.c_chess_cli = c_chess_cli_path
        self.stockfish = stockfish_path
        self.book_file = book_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Verify executables exist
        if not os.path.exists(self.c_chess_cli):
            raise FileNotFoundError(f"c-chess-cli not found: {self.c_chess_cli}")
        if not os.path.exists(self.stockfish):
            raise FileNotFoundError(f"Stockfish not found: {self.stockfish}")
        if not os.path.exists(self.book_file):
            raise FileNotFoundError(f"Opening book not found: {self.book_file}")

    def run_match(self, pytorch_engine_cmd: str, engine_name: str,
                  games: int = 100, concurrency: int = 4,
                  nodes: int = 5000, hash_mb: int = 16) -> str:
        """Run a match between PyTorch engine and Stockfish"""

        pgn_file = self.output_dir / f"{engine_name}_vs_stockfish.pgn"
        log_file = self.output_dir / f"{engine_name}_match.log"

        # Build c-chess-cli command
        cmd = [self.c_chess_cli]

        # Add concurrency
        cmd += ["-concurrency", str(concurrency)]

        # Add game parameters
        cmd += ["-games", str(games)]
        cmd += ["-rounds", "1"]
        cmd += ["-repeat"]

        # Add opening book
        cmd += ["-openings", f"file={self.book_file}", "order=random", "srand=12345"]

        # Add draw/resign conditions
        cmd += ["-draw", "count=8", "score=10"]
        cmd += ["-resign", "count=3", "score=700"]

        # Add PGN output
        cmd += ["-pgn", str(pgn_file), "0"]

        # Add engine configurations
        cmd += ["-each", f"option.Hash={hash_mb}", "option.Threads=1"]

        # Time control - use nodes for consistency
        cmd += ["-each", f"tc=inf", f"nodes={nodes}"]

        # Add engines
        cmd += ["-engine", f"cmd={self.stockfish}", "name=Stockfish"]
        cmd += ["-engine", f"cmd={pytorch_engine_cmd}", f"name={engine_name}"]

        logger.info(f"Running match: {engine_name} vs Stockfish")
        logger.info(f"Command: {' '.join(cmd)}")

        # Run the match
        with open(log_file, 'w') as log:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )

            # Monitor output
            for line in process.stdout:
                log.write(line)
                if "Score of" in line or "Elo diff" in line:
                    logger.info(line.strip())

            process.wait()

        if process.returncode != 0:
            logger.error(f"c-chess-cli failed with return code {process.returncode}")

        return str(pgn_file)

    def run_gauntlet(self, checkpoint_infos: List[Dict], games_per_model: int,
                     concurrency: int, nodes: int, python_exe: str = sys.executable) -> List[str]:
        """Run a gauntlet tournament - each model plays against Stockfish"""

        pgn_files = []

        # Get the directory containing the UCI engine script
        engine_dir = Path(__file__).parent
        engine_script = engine_dir / "pytorch_uci_engine.py"

        # Create a wrapper script for proper environment setup
        wrapper_script = engine_dir / "engine_wrapper.sh"
        if sys.platform == "win32":
            wrapper_script = engine_dir / "engine_wrapper.bat"
            wrapper_content = f"""@echo off
set PYTHONPATH={engine_dir};%PYTHONPATH%
cd /d "{engine_dir}"
{python_exe} "{engine_script}" %1
"""
        else:
            wrapper_content = f"""#!/bin/bash
export PYTHONPATH="{engine_dir}:$PYTHONPATH"
cd "{engine_dir}"
exec {python_exe} "{engine_script}" "$1"
"""

        # Write the wrapper script
        with open(wrapper_script, 'w') as f:
            f.write(wrapper_content)

        if sys.platform != "win32":
            os.chmod(wrapper_script,
                     stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)  # Make executable on Unix

        for info in checkpoint_infos:
            # Use the wrapper script as the engine command
            engine_cmd = f"{wrapper_script} {info['path']}"
            engine_name = f"pytorch_epoch{info['epoch']}"

            logger.info(f"Starting engine: {engine_name}")
            logger.info(f"Engine command: {engine_cmd}")

            # Run match
            pgn_file = self.run_match(
                engine_cmd,
                engine_name,
                games=games_per_model,
                concurrency=concurrency,
                nodes=nodes
            )
            pgn_files.append(pgn_file)

            logger.info(f"Completed games for {info['name']}")

        # Combine all PGN files
        combined_pgn = self.output_dir / "all_games.pgn"
        with open(combined_pgn, 'w') as out:
            for pgn_file in pgn_files:
                with open(pgn_file, 'r') as inp:
                    out.write(inp.read())
                    out.write("\n\n")

        return str(combined_pgn)


class EloAnalyzer:
    """Analyzes game results and computes ELO ratings"""

    def __init__(self, ordo_path: Optional[str] = None):
        self.ordo_path = ordo_path
        self.results = {}

    def parse_pgn_results(self, pgn_file: str) -> Dict:
        """Parse PGN file and extract results"""
        results = {}

        with open(pgn_file, 'r') as f:
            content = f.read()

        # Parse games
        games = content.split('[Event')
        for game in games[1:]:  # Skip first empty split
            white_match = re.search(r'\[White "([^"]+)"\]', game)
            black_match = re.search(r'\[Black "([^"]+)"\]', game)
            result_match = re.search(r'\[Result "([^"]+)"\]', game)

            if white_match and black_match and result_match:
                white = white_match.group(1)
                black = black_match.group(1)
                result = result_match.group(1)

                # Initialize players
                for player in [white, black]:
                    if player not in results:
                        results[player] = {"wins": 0, "draws": 0, "losses": 0, "games": 0}

                # Update results
                results[white]["games"] += 1
                results[black]["games"] += 1

                if result == "1-0":
                    results[white]["wins"] += 1
                    results[black]["losses"] += 1
                elif result == "0-1":
                    results[white]["losses"] += 1
                    results[black]["wins"] += 1
                elif result == "1/2-1/2":
                    results[white]["draws"] += 1
                    results[black]["draws"] += 1

        self.results = results
        return results

    def calculate_approximate_elo(self, results: Dict) -> Dict:
        """Calculate approximate ELO ratings (anchoring Stockfish at 0)"""
        ratings = {}

        for player, stats in results.items():
            if stats["games"] == 0:
                continue

            # Calculate win rate
            score = stats["wins"] + stats["draws"] * 0.5
            win_rate = score / stats["games"]

            # Convert to ELO difference (using logistic formula)
            # Protect against extreme values
            win_rate = max(0.01, min(0.99, win_rate))

            if player == "Stockfish":
                elo_diff = 0
            else:
                # Calculate ELO difference from expected 50% against Stockfish
                elo_diff = -400 * math.log10(1 / win_rate - 1)

            # Calculate error margin (95% confidence)
            error = 400 / math.sqrt(stats["games"])

            ratings[player] = {
                "elo": elo_diff,
                "error": error,
                "games": stats["games"],
                "score": score,
                "performance": win_rate * 100
            }

        return ratings

    def run_ordo(self, pgn_file: str, output_dir: str, concurrency: int = 1) -> Dict:
        """Run actual ordo for precise ELO calculation"""
        if not self.ordo_path or not os.path.exists(self.ordo_path):
            logger.warning("Ordo not available, using approximate calculation")
            results = self.parse_pgn_results(pgn_file)
            return self.calculate_approximate_elo(results)

        ordo_output = os.path.join(output_dir, "ordo.out")

        cmd = [
            self.ordo_path,
            "-q",  # Quiet mode
            "-g",  # Gauntlet mode
            "-J",  # JSON output
            "-p", pgn_file,
            "-a", "0.0",  # Anchor at 0
            "--anchor=Stockfish",
            "--draw-auto",
            "--white-auto",
            "-s", "100",
            f"--cpus={concurrency}",
            "-o", ordo_output
        ]

        logger.info("Running ordo for precise ELO calculation...")

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)

            # Parse ordo output
            ratings = {}
            with open(ordo_output, 'r') as f:
                for line in f:
                    if "Stockfish" in line or "pytorch" in line:
                        parts = line.split()
                        if len(parts) >= 5:
                            name = parts[1]
                            elo = float(parts[3])
                            error = float(parts[4])
                            ratings[name] = {"elo": elo, "error": error}

            return ratings

        except subprocess.CalledProcessError as e:
            logger.error(f"Ordo failed: {e}")
            # Fallback to approximate
            results = self.parse_pgn_results(pgn_file)
            return self.calculate_approximate_elo(results)

    def generate_report(self, ratings: Dict, output_file: str):
        """Generate a human-readable report"""
        with open(output_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("PyTorch NNUE Model ELO Testing Results\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Base Engine: Stockfish (anchored at 0 ELO)\n\n")

            f.write("Model Rankings:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Rank':<5} {'Model':<30} {'ELO':<10} {'Error':<8} {'Games':<8}\n")
            f.write("-" * 60 + "\n")

            # Sort by ELO
            sorted_players = sorted(ratings.items(),
                                    key=lambda x: x[1].get('elo', -999),
                                    reverse=True)

            rank = 1
            for name, stats in sorted_players:
                if name != "Stockfish":
                    elo = stats.get('elo', 0)
                    error = stats.get('error', 0)
                    games = stats.get('games', 0)
                    f.write(f"{rank:<5} {name:<30} {elo:+7.1f} ±{error:5.1f} {games:>6}\n")
                    rank += 1

            f.write("\n" + "=" * 60 + "\n")

            # Performance details
            if self.results:
                f.write("\nDetailed Performance:\n")
                f.write("-" * 60 + "\n")
                for name, stats in self.results.items():
                    if name != "Stockfish":
                        f.write(f"\n{name}:\n")
                        f.write(f"  Wins:   {stats['wins']}\n")
                        f.write(f"  Draws:  {stats['draws']}\n")
                        f.write(f"  Losses: {stats['losses']}\n")
                        f.write(f"  Score:  {stats['wins'] + stats['draws'] * 0.5:.1f}/{stats['games']}\n")
                        perf = (stats['wins'] + stats['draws'] * 0.5) / stats['games'] * 100
                        f.write(f"  Performance: {perf:.1f}%\n")


def main():
    parser = argparse.ArgumentParser(
        description="Test PyTorch NNUE models against Stockfish and calculate ELO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "checkpoint_dir",
        help="Directory containing .ckpt files to test"
    )

    parser.add_argument(
        "--stockfish",
        default="stockfish",
        help="Path to Stockfish executable"
    )

    parser.add_argument(
        "--c-chess-cli",
        default="c-chess-cli",
        help="Path to c-chess-cli executable"
    )

    parser.add_argument(
        "--ordo",
        default=None,
        help="Path to ordo executable (optional, will use approximation if not provided)"
    )

    parser.add_argument(
        "--book",
        default="book.epd",
        help="Path to opening book file"
    )

    parser.add_argument(
        "--games-per-model",
        type=int,
        default=100,
        help="Number of games each model plays against Stockfish"
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of concurrent games"
    )

    parser.add_argument(
        "--nodes",
        type=int,
        default=5000,
        help="Nodes per move for each engine"
    )

    parser.add_argument(
        "--output-dir",
        default="elo_test_results",
        help="Directory for output files"
    )

    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=None,
        help="Maximum number of checkpoints to test (useful for testing)"
    )

    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for engines"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging to file
    log_file = output_dir / "test_run.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    logger.info("=" * 60)
    logger.info("Starting PyTorch NNUE ELO Testing")
    logger.info("=" * 60)

    try:
        # Find checkpoints
        ckpt_manager = CheckpointManager(args.checkpoint_dir)
        checkpoints = ckpt_manager.find_checkpoints()

        if not checkpoints:
            logger.error("No checkpoint files found!")
            return 1

        # Limit checkpoints if requested
        if args.max_checkpoints:
            checkpoints = checkpoints[:args.max_checkpoints]
            logger.info(f"Limited to {len(checkpoints)} checkpoints for testing")

        # Get checkpoint info
        checkpoint_infos = [ckpt_manager.get_checkpoint_info(ckpt) for ckpt in checkpoints]

        # Run games
        game_runner = GameRunner(
            args.c_chess_cli,
            args.stockfish,
            args.book,
            args.output_dir
        )

        combined_pgn = game_runner.run_gauntlet(
            checkpoint_infos,
            args.games_per_model,
            args.concurrency,
            args.nodes,
            args.python
        )

        logger.info("All games completed!")

        # Analyze results
        analyzer = EloAnalyzer(args.ordo)

        if args.ordo:
            ratings = analyzer.run_ordo(combined_pgn, args.output_dir, args.concurrency)
        else:
            results = analyzer.parse_pgn_results(combined_pgn)
            ratings = analyzer.calculate_approximate_elo(results)

        # Generate reports
        report_file = output_dir / "elo_report.txt"
        analyzer.generate_report(ratings, str(report_file))

        # Save JSON results
        json_file = output_dir / "results.json"
        with open(json_file, 'w') as f:
            json.dump({
                "checkpoints": checkpoint_infos,
                "ratings": {k: v for k, v in ratings.items()},
                "raw_results": analyzer.results,
                "test_parameters": {
                    "games_per_model": args.games_per_model,
                    "nodes_per_move": args.nodes,
                    "concurrency": args.concurrency
                }
            }, f, indent=2, default=str)

        logger.info(f"Results saved to {output_dir}")
        logger.info(f"Report: {report_file}")
        logger.info(f"JSON data: {json_file}")
        logger.info(f"PGN games: {combined_pgn}")

        # Print summary
        print("\n" + "=" * 60)
        print("ELO Testing Complete!")
        print("=" * 60)
        print(f"\nResults saved to: {output_dir}")
        print("\nModel Rankings (vs Stockfish at 0 ELO):")
        print("-" * 60)

        sorted_models = sorted(
            [(k, v) for k, v in ratings.items() if k != "Stockfish"],
            key=lambda x: x[1].get('elo', -999),
            reverse=True
        )

        for name, stats in sorted_models:
            elo = stats.get('elo', 0)
            error = stats.get('error', 0)
            print(f"{name:<30} {elo:+7.1f} ± {error:.1f}")

        return 0

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())