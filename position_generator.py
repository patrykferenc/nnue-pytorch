"""
Advanced Position Generation and Dataset Management for NNUE Testing
"""

import chess
import chess.pgn
import chess.engine
import chess.polyglot
import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Iterator
from pathlib import Path
import requests
import zipfile
import io
import json
from dataclasses import dataclass


@dataclass
class PositionSet:
    """Container for a set of test positions with metadata"""
    positions: List[str]  # FEN strings
    name: str
    description: str
    source: str
    expected_characteristics: Dict = None


class AdvancedPositionGenerator:
    """Advanced position generation with specific characteristics"""

    def __init__(self):
        self.opening_positions = []
        self.tactical_positions = []
        self.endgame_positions = []

    def generate_material_imbalanced_positions(self, count: int = 100) -> List[str]:
        """Generate positions with material imbalances"""
        positions = []

        for _ in range(count * 5):  # Try more to get valid positions
            if len(positions) >= count:
                break

            try:
                board = chess.Board()

                # Random opening moves
                for _ in range(random.randint(8, 20)):
                    legal_moves = list(board.legal_moves)
                    if not legal_moves or board.is_game_over():
                        break
                    board.push(random.choice(legal_moves))

                # Force material imbalance by removing pieces
                piece_map = board.piece_map()
                squares_with_pieces = list(piece_map.keys())

                if len(squares_with_pieces) > 10:
                    # Remove 1-3 pieces randomly
                    pieces_to_remove = random.randint(1, min(3, len(squares_with_pieces) - 8))
                    squares_to_clear = random.sample(squares_with_pieces, pieces_to_remove)

                    for square in squares_to_clear:
                        board.remove_piece_at(square)

                    if not board.is_game_over() and list(board.legal_moves):
                        positions.append(board.fen())

            except Exception:
                continue

        return positions[:count]

    def generate_tactical_positions(self, count: int = 100) -> List[str]:
        """Generate positions with tactical motifs"""
        positions = []

        # Known tactical pattern setups
        tactical_setups = [
            # Pin patterns
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R",
            # Fork opportunities
            "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR",
            # Discovered attack setups
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R",
        ]

        for setup_fen in tactical_setups:
            try:
                board = chess.Board(setup_fen)

                # Make 1-3 additional moves to create variations
                for variation in range(count // len(tactical_setups)):
                    temp_board = board.copy()

                    for _ in range(random.randint(1, 3)):
                        legal_moves = list(temp_board.legal_moves)
                        if not legal_moves or temp_board.is_game_over():
                            break
                        temp_board.push(random.choice(legal_moves))

                    if not temp_board.is_game_over():
                        positions.append(temp_board.fen())

            except Exception:
                continue

        return positions[:count]

    def generate_endgame_positions(self, count: int = 100) -> List[str]:
        """Generate endgame positions with specific piece configurations"""
        positions = []

        endgame_types = [
            self._generate_king_pawn_endgames,
            self._generate_rook_endgames,
            self._generate_queen_endgames,
            self._generate_minor_piece_endgames,
        ]

        positions_per_type = count // len(endgame_types)

        for generator in endgame_types:
            positions.extend(generator(positions_per_type))

        return positions[:count]

    def _generate_king_pawn_endgames(self, count: int) -> List[str]:
        """Generate King + Pawn endgames"""
        positions = []

        for _ in range(count * 2):
            if len(positions) >= count:
                break

            try:
                board = chess.Board("8/8/8/8/8/8/8/8 w - - 0 1")

                # Place kings
                white_king = random.choice([sq for sq in chess.SQUARES if chess.square_rank(sq) < 6])
                black_king = random.choice([sq for sq in chess.SQUARES
                                            if chess.square_distance(sq, white_king) >= 2])

                board.set_piece_at(white_king, chess.Piece(chess.KING, chess.WHITE))
                board.set_piece_at(black_king, chess.Piece(chess.KING, chess.BLACK))

                # Place 1-3 pawns
                pawn_count = random.randint(1, 3)
                placed_pawns = 0

                for _ in range(20):  # Try to place pawns
                    if placed_pawns >= pawn_count:
                        break

                    square = random.choice([sq for sq in chess.SQUARES
                                            if 1 <= chess.square_rank(sq) <= 6
                                            and board.piece_at(sq) is None])

                    color = random.choice([chess.WHITE, chess.BLACK])
                    board.set_piece_at(square, chess.Piece(chess.PAWN, color))
                    placed_pawns += 1

                if not board.is_game_over() and list(board.legal_moves):
                    positions.append(board.fen())

            except Exception:
                continue

        return positions[:count]

    def _generate_rook_endgames(self, count: int) -> List[str]:
        """Generate rook endgames"""
        positions = []

        for _ in range(count * 2):
            if len(positions) >= count:
                break

            try:
                board = chess.Board("8/8/8/8/8/8/8/8 w - - 0 1")

                # Place kings
                white_king = random.choice(chess.SQUARES)
                black_king = random.choice([sq for sq in chess.SQUARES
                                            if chess.square_distance(sq, white_king) >= 2])

                board.set_piece_at(white_king, chess.Piece(chess.KING, chess.WHITE))
                board.set_piece_at(black_king, chess.Piece(chess.KING, chess.BLACK))

                # Place rooks
                available_squares = [sq for sq in chess.SQUARES if board.piece_at(sq) is None]

                if len(available_squares) >= 2:
                    rook_squares = random.sample(available_squares, 2)
                    board.set_piece_at(rook_squares[0], chess.Piece(chess.ROOK, chess.WHITE))
                    board.set_piece_at(rook_squares[1], chess.Piece(chess.ROOK, chess.BLACK))

                    # Optionally add pawns
                    if random.random() < 0.5 and len(available_squares) > 2:
                        remaining_squares = [sq for sq in chess.SQUARES
                                             if board.piece_at(sq) is None
                                             and 1 <= chess.square_rank(sq) <= 6]

                        if remaining_squares:
                            pawn_square = random.choice(remaining_squares)
                            color = random.choice([chess.WHITE, chess.BLACK])
                            board.set_piece_at(pawn_square, chess.Piece(chess.PAWN, color))

                    if not board.is_game_over() and list(board.legal_moves):
                        positions.append(board.fen())

            except Exception:
                continue

        return positions[:count]

    def _generate_queen_endgames(self, count: int) -> List[str]:
        """Generate queen endgames"""
        positions = []

        for _ in range(count * 2):
            if len(positions) >= count:
                break

            try:
                board = chess.Board("8/8/8/8/8/8/8/8 w - - 0 1")

                # Place kings
                white_king = random.choice(chess.SQUARES)
                black_king = random.choice([sq for sq in chess.SQUARES
                                            if chess.square_distance(sq, white_king) >= 2])

                board.set_piece_at(white_king, chess.Piece(chess.KING, chess.WHITE))
                board.set_piece_at(black_king, chess.Piece(chess.KING, chess.BLACK))

                # Usually Q vs K or Q vs Q
                available_squares = [sq for sq in chess.SQUARES if board.piece_at(sq) is None]

                if available_squares:
                    queen_square = random.choice(available_squares)
                    board.set_piece_at(queen_square, chess.Piece(chess.QUEEN, chess.WHITE))

                    # Sometimes add opposing queen
                    if random.random() < 0.3:
                        remaining_squares = [sq for sq in chess.SQUARES if board.piece_at(sq) is None]
                        if remaining_squares:
                            black_queen_square = random.choice(remaining_squares)
                            board.set_piece_at(black_queen_square, chess.Piece(chess.QUEEN, chess.BLACK))

                    if not board.is_game_over() and list(board.legal_moves):
                        positions.append(board.fen())

            except Exception:
                continue

        return positions[:count]

    def _generate_minor_piece_endgames(self, count: int) -> List[str]:
        """Generate minor piece endgames"""
        positions = []

        for _ in range(count * 2):
            if len(positions) >= count:
                break

            try:
                board = chess.Board("8/8/8/8/8/8/8/8 w - - 0 1")

                # Place kings
                white_king = random.choice(chess.SQUARES)
                black_king = random.choice([sq for sq in chess.SQUARES
                                            if chess.square_distance(sq, white_king) >= 2])

                board.set_piece_at(white_king, chess.Piece(chess.KING, chess.WHITE))
                board.set_piece_at(black_king, chess.Piece(chess.KING, chess.BLACK))

                # Place minor pieces
                available_squares = [sq for sq in chess.SQUARES if board.piece_at(sq) is None]

                if len(available_squares) >= 2:
                    piece_squares = random.sample(available_squares, 2)
                    pieces = random.choice([
                        (chess.BISHOP, chess.BISHOP),
                        (chess.KNIGHT, chess.KNIGHT),
                        (chess.BISHOP, chess.KNIGHT),
                        (chess.KNIGHT, chess.BISHOP)
                    ])

                    board.set_piece_at(piece_squares[0], chess.Piece(pieces[0], chess.WHITE))
                    board.set_piece_at(piece_squares[1], chess.Piece(pieces[1], chess.BLACK))

                    if not board.is_game_over() and list(board.legal_moves):
                        positions.append(board.fen())

            except Exception:
                continue

        return positions[:count]


class PositionSetManager:
    """Manage collections of test positions"""

    def __init__(self):
        self.position_sets = {}
        self.generator = AdvancedPositionGenerator()

    def create_comprehensive_test_suite(self, positions_per_category: int = 200) -> Dict[str, PositionSet]:
        """Create a comprehensive test suite with multiple categories"""

        # Standard opening positions
        opening_positions = self._get_standard_openings()

        # Generated positions
        tactical_positions = self.generator.generate_tactical_positions(positions_per_category)
        endgame_positions = self.generator.generate_endgame_positions(positions_per_category)
        imbalanced_positions = self.generator.generate_material_imbalanced_positions(positions_per_category)

        position_sets = {
            "openings": PositionSet(
                positions=opening_positions[:positions_per_category],
                name="Standard Openings",
                description="Common opening positions from master games",
                source="theory",
                expected_characteristics={"eval_range": (-100, 100), "piece_count": (26, 32)}
            ),

            "tactical": PositionSet(
                positions=tactical_positions,
                name="Tactical Positions",
                description="Positions with tactical motifs and combinations",
                source="generated",
                expected_characteristics={"eval_range": (-500, 500), "piece_count": (15, 30)}
            ),

            "endgames": PositionSet(
                positions=endgame_positions,
                name="Endgame Positions",
                description="Pure endgame positions with few pieces",
                source="generated",
                expected_characteristics={"eval_range": (-1000, 1000), "piece_count": (3, 10)}
            ),

            "imbalanced": PositionSet(
                positions=imbalanced_positions,
                name="Material Imbalanced",
                description="Positions with material imbalances",
                source="generated",
                expected_characteristics={"eval_range": (-800, 800), "piece_count": (10, 25)}
            )
        }

        return position_sets

    def _get_standard_openings(self) -> List[str]:
        """Get standard opening positions"""
        openings = [
            # Italian Game
            "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",

            # Ruy Lopez
            "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",

            # Sicilian Dragon
            "rnbqkb1r/pp2pp1p/3p1np1/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",

            # Queen's Gambit
            "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2",

            # King's Indian Defense
            "rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQK2R b KQ - 0 6",

            # French Defense
            "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq d6 0 3",

            # Caro-Kann Defense
            "rnbqkbnr/pp2pppp/2p5/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq d6 0 3",

            # English Opening
            "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq c3 0 1",
        ]

        return openings

    def load_positions_from_pgn_database(self, pgn_path: str, max_games: int = 1000) -> List[str]:
        """Extract positions from a PGN database"""
        positions = []

        try:
            with open(pgn_path, 'r') as f:
                games_processed = 0

                while games_processed < max_games:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break

                    # Extract positions from each game
                    board = game.board()
                    move_count = 0

                    for move in game.mainline_moves():
                        board.push(move)
                        move_count += 1

                        # Sample positions at different phases
                        if move_count in [10, 20, 30, 40]:  # Different game phases
                            positions.append(board.fen())

                    games_processed += 1

                    if games_processed % 100 == 0:
                        print(f"Processed {games_processed} games, extracted {len(positions)} positions")

        except Exception as e:
            print(f"Error reading PGN database: {e}")

        return positions

    def export_position_set(self, position_set: PositionSet, filename: str, format: str = "epd"):
        """Export position set to file"""
        if format.lower() == "epd":
            with open(filename, 'w') as f:
                for fen in position_set.positions:
                    f.write(f"{fen}\n")

        elif format.lower() == "json":
            data = {
                "name": position_set.name,
                "description": position_set.description,
                "source": position_set.source,
                "expected_characteristics": position_set.expected_characteristics,
                "positions": position_set.positions
            }

            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)

    def load_position_set_from_json(self, filename: str) -> PositionSet:
        """Load position set from JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)

        return PositionSet(
            positions=data["positions"],
            name=data["name"],
            description=data["description"],
            source=data["source"],
            expected_characteristics=data.get("expected_characteristics", {})
        )


def create_test_datasets():
    """Create comprehensive test datasets for NNUE evaluation"""

    manager = PositionSetManager()

    # Create comprehensive test suite
    print("Creating comprehensive test suite...")
    position_sets = manager.create_comprehensive_test_suite(positions_per_category=500)

    # Export each category
    output_dir = Path("test_positions")
    output_dir.mkdir(exist_ok=True)

    for category, position_set in position_sets.items():
        # Export as EPD for direct use
        epd_file = output_dir / f"{category}_positions.epd"
        manager.export_position_set(position_set, epd_file, "epd")

        # Export as JSON with metadata
        json_file = output_dir / f"{category}_positions.json"
        manager.export_position_set(position_set, json_file, "json")

        print(f"Created {category}: {len(position_set.positions)} positions")
        print(f"  EPD: {epd_file}")
        print(f"  JSON: {json_file}")

    print(f"\nAll test datasets created in {output_dir}")
    return position_sets


if __name__ == "__main__":
    # Create test datasets
    datasets = create_test_datasets()

    # Print summary
    total_positions = sum(len(ps.positions) for ps in datasets.values())
    print(f"\nTotal positions created: {total_positions}")
    print("Ready for scientific evaluation testing!")