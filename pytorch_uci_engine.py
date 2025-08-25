#!/usr/bin/env python3
"""
UCI-compliant chess engine wrapper for PyTorch NNUE models.
This engine provides a UCI interface to play games with c-chess-cli.
"""

import sys
import os
import torch
import chess
import chess.pgn
import numpy as np
import time
import traceback
from typing import Optional, Tuple
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model as M
import features
from nnue_dataset import make_sparse_batch_from_fens, destroy_sparse_batch

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pytorch_engine.log'),
    ]
)


class PyTorchNNUEEngine:
    """UCI-compliant engine using PyTorch NNUE model"""

    def __init__(self, model_path: str, feature_set_name: str = "HalfKAv2_hm^"):
        self.model_path = model_path
        self.feature_set_name = feature_set_name
        self.board = chess.Board()
        self.model = None
        self.device = None
        self.feature_set = None
        self.search_nodes = 0
        self.max_search_depth = 3  # Simple shallow search

        # Engine info
        self.name = f"PyTorchNNUE_{os.path.basename(model_path)}"
        self.author = "PyTorch NNUE Test Engine"

        self._load_model()

    def _load_model(self):
        """Load the PyTorch model from checkpoint"""
        try:
            # Load feature set
            self.feature_set = features.get_feature_set_from_name(self.feature_set_name)

            # Create model
            self.model = M.NNUE(self.feature_set)

            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.eval()

            # Setup device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)

            # Initialize idx_offset for single position evaluation
            self.model.layer_stacks.idx_offset = torch.zeros((1,), dtype=torch.int64, device=self.device)

            logging.info(f"Model loaded successfully from {self.model_path} on {self.device}")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            traceback.print_exc()
            sys.exit(1)

    def _evaluate_position(self, board: chess.Board) -> float:
        """Evaluate a position using the NNUE model"""
        try:
            fen = board.fen()

            # Create batch from FEN
            batch_ptr = make_sparse_batch_from_fens(
                self.feature_set,
                [fen],
                [0],  # score
                [0],  # ply
                [1]  # result
            )

            if not batch_ptr:
                logging.error(f"Failed to create batch from FEN: {fen}")
                return 0.0

            try:
                with torch.no_grad():
                    # Get tensors from batch
                    tensors = batch_ptr.contents.get_tensors(self.device)
                    us, them, w_idx, w_val, b_idx, b_val, outcome, score_t, psqt_idx, ls_idx = tensors

                    # Forward pass
                    output = self.model(us, them, w_idx, w_val, b_idx, b_val, psqt_idx, ls_idx)

                    # Convert to centipawns
                    eval_score = output.item() * self.model.nnue2score

                    # Adjust for side to move
                    if not board.turn:  # Black to move
                        eval_score = -eval_score

                    return eval_score
            finally:
                destroy_sparse_batch(batch_ptr)

        except Exception as e:
            logging.error(f"Evaluation error: {e}")
            return 0.0

    def _quiescence_search(self, board: chess.Board, alpha: float, beta: float, depth: int = 0) -> float:
        """Simple quiescence search to handle captures"""
        if depth > 4:  # Limit quiescence depth
            return self._evaluate_position(board)

        stand_pat = self._evaluate_position(board)

        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        # Only search captures
        for move in board.legal_moves:
            if board.is_capture(move):
                board.push(move)
                score = -self._quiescence_search(board, -beta, -alpha, depth + 1)
                board.pop()

                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score

        return alpha

    def _minimax_search(self, board: chess.Board, depth: int, alpha: float, beta: float,
                        root: bool = False) -> Tuple[float, Optional[chess.Move]]:
        """Simple minimax search with alpha-beta pruning"""
        self.search_nodes += 1

        # Check for terminal positions
        if board.is_checkmate():
            return -30000 + board.ply(), None
        if board.is_stalemate() or board.is_insufficient_material():
            return 0, None
        if board.can_claim_draw():
            return 0, None

        # Leaf node - use quiescence search
        if depth == 0:
            return self._quiescence_search(board, alpha, beta), None

        best_move = None
        best_score = -float('inf')

        # Generate and order moves
        moves = list(board.legal_moves)

        # Simple move ordering: captures first (MVV-LVA style)
        def move_priority(move):
            if board.is_capture(move):
                captured = board.piece_type_at(move.to_square)
                attacker = board.piece_type_at(move.from_square)
                # MVV-LVA: Most Valuable Victim - Least Valuable Attacker
                victim_value = [0, 100, 300, 300, 500, 900, 0][captured] if captured else 0
                attacker_value = [0, 100, 300, 300, 500, 900, 0][attacker] if attacker else 0
                return victim_value * 10 - attacker_value
            return 0

        moves.sort(key=move_priority, reverse=True)

        for move in moves:
            board.push(move)

            if board.is_repetition(2):
                score = 0
            else:
                score, _ = self._minimax_search(board, depth - 1, -beta, -alpha, False)
                score = -score

            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score

            if alpha >= beta:
                break  # Beta cutoff

        return best_score, best_move

    def search(self, board: chess.Board, nodes: int = 5000) -> chess.Move:
        """Search for the best move with node limit"""
        self.search_nodes = 0
        best_move = None

        # Iterative deepening
        for depth in range(1, self.max_search_depth + 1):
            if self.search_nodes >= nodes:
                break

            score, move = self._minimax_search(
                board, depth, -float('inf'), float('inf'), root=True
            )

            if move:
                best_move = move
                logging.debug(f"Depth {depth}: {move} score {score:.0f} ({self.search_nodes} nodes)")

        # Fallback to first legal move if no move found
        if not best_move and list(board.legal_moves):
            best_move = list(board.legal_moves)[0]

        return best_move

    def handle_uci(self):
        """Handle UCI protocol communication"""
        print(f"id name {self.name}")
        print(f"id author {self.author}")
        print("option name Hash type spin default 16 min 1 max 1024")
        print("option name Threads type spin default 1 min 1 max 1")
        print("uciok")

    def handle_isready(self):
        """Handle isready command"""
        print("readyok")

    def handle_position(self, parts):
        """Handle position command"""
        if parts[1] == "startpos":
            self.board = chess.Board()
            moves_start = 3 if len(parts) > 2 and parts[2] == "moves" else 2
        else:  # fen
            fen_parts = []
            i = 2
            while i < len(parts) and parts[i] != "moves":
                fen_parts.append(parts[i])
                i += 1
            fen = " ".join(fen_parts)
            self.board = chess.Board(fen)
            moves_start = i + 1 if i < len(parts) else len(parts)

        # Apply moves
        for move_str in parts[moves_start:]:
            try:
                move = chess.Move.from_uci(move_str)
                if move in self.board.legal_moves:
                    self.board.push(move)
            except:
                pass

    def handle_go(self, parts):
        """Handle go command"""
        nodes = 5000  # Default
        depth = self.max_search_depth

        # Parse go parameters
        for i, part in enumerate(parts):
            if part == "nodes" and i + 1 < len(parts):
                try:
                    nodes = int(parts[i + 1])
                except:
                    pass
            elif part == "depth" and i + 1 < len(parts):
                try:
                    depth = min(int(parts[i + 1]), 10)
                    self.max_search_depth = depth
                except:
                    pass

        # Search for best move
        best_move = self.search(self.board, nodes)

        if best_move:
            print(f"bestmove {best_move.uci()}")
        else:
            print("bestmove 0000")

    def handle_eval(self):
        """Handle eval command (non-standard but useful for debugging)"""
        score = self._evaluate_position(self.board)
        print(f"info string evaluation: {score:.2f}")

    def run(self):
        """Main UCI loop"""
        while True:
            try:
                line = input().strip()
                if not line:
                    continue

                logging.debug(f"< {line}")
                parts = line.split()

                if not parts:
                    continue

                command = parts[0]

                if command == "uci":
                    self.handle_uci()
                elif command == "isready":
                    self.handle_isready()
                elif command == "position":
                    self.handle_position(parts)
                elif command == "go":
                    self.handle_go(parts)
                elif command == "eval":
                    self.handle_eval()
                elif command == "quit":
                    break
                elif command == "ucinewgame":
                    self.board = chess.Board()

            except EOFError:
                break
            except Exception as e:
                logging.error(f"Error handling command: {e}")
                traceback.print_exc()


def main():
    """Main entry point for UCI engine"""
    if len(sys.argv) < 2:
        print("Usage: python pytorch_uci_engine.py <model.ckpt>", file=sys.stderr)
        sys.exit(1)

    model_path = sys.argv[1]

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    engine = PyTorchNNUEEngine(model_path)
    engine.run()


if __name__ == "__main__":
    main()