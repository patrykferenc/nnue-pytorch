#!/bin/bash

# ELO Testing Runner Script
# This script sets up and runs ELO testing for PyTorch NNUE models

set -e  # Exit on error

echo "=========================================="
echo "PyTorch NNUE Model ELO Testing"
echo "=========================================="

# Default values
CHECKPOINT_DIR=""
STOCKFISH_PATH="stockfish"
C_CHESS_CLI_PATH="c-chess-cli"
ORDO_PATH=""
BOOK_PATH="book.epd"
GAMES_PER_MODEL=100
CONCURRENCY=4
NODES=5000
OUTPUT_DIR="elo_test_results_$(date +%Y%m%d_%H%M%S)"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --stockfish)
            STOCKFISH_PATH="$2"
            shift 2
            ;;
        --c-chess-cli)
            C_CHESS_CLI_PATH="$2"
            shift 2
            ;;
        --ordo)
            ORDO_PATH="$2"
            shift 2
            ;;
        --book)
            BOOK_PATH="$2"
            shift 2
            ;;
        --games)
            GAMES_PER_MODEL="$2"
            shift 2
            ;;
        --concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        --nodes)
            NODES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 --checkpoint-dir DIR [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --checkpoint-dir DIR    Directory containing .ckpt files"
            echo ""
            echo "Options:"
            echo "  --stockfish PATH       Path to Stockfish (default: stockfish)"
            echo "  --c-chess-cli PATH     Path to c-chess-cli (default: c-chess-cli)"
            echo "  --ordo PATH           Path to ordo (optional)"
            echo "  --book PATH           Opening book (default: book.epd)"
            echo "  --games N             Games per model (default: 100)"
            echo "  --concurrency N       Concurrent games (default: 4)"
            echo "  --nodes N             Nodes per move (default: 5000)"
            echo "  --output-dir DIR      Output directory"
            echo "  --help                Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$CHECKPOINT_DIR" ]; then
    echo "Error: --checkpoint-dir is required"
    echo "Use --help for usage information"
    exit 1
fi

# Check if checkpoint directory exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

# Check for required executables
echo "Checking dependencies..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required"
    exit 1
fi

# Check Stockfish
if ! command -v "$STOCKFISH_PATH" &> /dev/null; then
    echo "Warning: Stockfish not found at: $STOCKFISH_PATH"
    echo "Please ensure Stockfish is installed or provide the correct path"
fi

# Check c-chess-cli
if ! command -v "$C_CHESS_CLI_PATH" &> /dev/null; then
    echo "Warning: c-chess-cli not found at: $C_CHESS_CLI_PATH"
    echo "Please ensure c-chess-cli is installed or provide the correct path"
fi

# Check for book file
if [ ! -f "$BOOK_PATH" ]; then
    echo "Warning: Opening book not found at: $BOOK_PATH"
    echo "Attempting to download a book..."

    # Try to download a standard book
    BOOK_URL="https://github.com/official-stockfish/books/raw/master/UHO_Lichess_4852_v1.epd.zip"
    wget -q "$BOOK_URL" -O book.zip
    if [ $? -eq 0 ]; then
        unzip -q book.zip
        rm book.zip
        BOOK_PATH="UHO_Lichess_4852_v1.epd"
        echo "Downloaded opening book: $BOOK_PATH"
    else
        echo "Error: Could not download opening book"
        exit 1
    fi
fi

# Check for ordo (optional)
if [ -n "$ORDO_PATH" ] && ! command -v "$ORDO_PATH" &> /dev/null; then
    echo "Warning: Ordo not found at: $ORDO_PATH"
    echo "Will use approximate ELO calculation"
    ORDO_PATH=""
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo ""
echo "Configuration:"
echo "  Checkpoint directory: $CHECKPOINT_DIR"
echo "  Stockfish: $STOCKFISH_PATH"
echo "  c-chess-cli: $C_CHESS_CLI_PATH"
echo "  Opening book: $BOOK_PATH"
echo "  Games per model: $GAMES_PER_MODEL"
echo "  Concurrency: $CONCURRENCY"
echo "  Nodes per move: $NODES"
echo "  Output directory: $OUTPUT_DIR"
if [ -n "$ORDO_PATH" ]; then
    echo "  Ordo: $ORDO_PATH"
else
    echo "  Ordo: Not available (using approximation)"
fi
echo ""

# Count checkpoints
CHECKPOINT_COUNT=$(find "$CHECKPOINT_DIR" -name "*.ckpt" | wc -l)
echo "Found $CHECKPOINT_COUNT checkpoint files"

if [ $CHECKPOINT_COUNT -eq 0 ]; then
    echo "Error: No .ckpt files found in $CHECKPOINT_DIR"
    exit 1
fi

# Estimate time
TOTAL_GAMES=$((CHECKPOINT_COUNT * GAMES_PER_MODEL))
EST_TIME_MIN=$((TOTAL_GAMES * 15 / CONCURRENCY / 60))  # Assume ~15 seconds per game
echo "Estimated time: ~$EST_TIME_MIN minutes for $TOTAL_GAMES total games"
echo ""

# Ask for confirmation
read -p "Continue with testing? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Testing cancelled"
    exit 0
fi

echo ""
echo "Starting ELO testing..."
echo "=========================================="

# Build command
CMD="python3 test_pytorch_elo.py"
CMD="$CMD \"$CHECKPOINT_DIR\""
CMD="$CMD --stockfish \"$STOCKFISH_PATH\""
CMD="$CMD --c-chess-cli \"$C_CHESS_CLI_PATH\""
CMD="$CMD --book \"$BOOK_PATH\""
CMD="$CMD --games-per-model $GAMES_PER_MODEL"
CMD="$CMD --concurrency $CONCURRENCY"
CMD="$CMD --nodes $NODES"
CMD="$CMD --output-dir \"$OUTPUT_DIR\""

if [ -n "$ORDO_PATH" ]; then
    CMD="$CMD --ordo \"$ORDO_PATH\""
fi

# Run the test
echo "Executing: $CMD"
echo ""
eval $CMD

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Testing completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "View the report:"
    echo "  cat $OUTPUT_DIR/elo_report.txt"
    echo ""
    echo "View the games:"
    echo "  less $OUTPUT_DIR/all_games.pgn"
else
    echo ""
    echo "Testing failed. Check the logs in $OUTPUT_DIR/test_run.log"
    exit 1
fi
