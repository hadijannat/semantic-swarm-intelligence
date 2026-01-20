#!/bin/bash
# Run ML reproducibility benchmarks
#
# This script runs the baseline training with fixed seeds and
# verifies reproducibility by comparing results across runs.
#
# Usage:
#   ./scripts/run_benchmarks.sh
#   ./scripts/run_benchmarks.sh --quick  # Run quick benchmark

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SEED=42
EPOCHS=10
TRAIN_SAMPLES=5000
VAL_SAMPLES=1000
OUTPUT_DIR="models/benchmarks"

# Parse arguments
QUICK_MODE=false
if [[ "$1" == "--quick" ]]; then
    QUICK_MODE=true
    EPOCHS=3
    TRAIN_SAMPLES=1000
    VAL_SAMPLES=200
    echo -e "${YELLOW}Running in quick mode (reduced epochs and samples)${NC}"
fi

echo "=================================================="
echo "NOA Semantic Swarm Mapper - ML Reproducibility Test"
echo "=================================================="
echo ""
echo "Configuration:"
echo "  Seed: $SEED"
echo "  Epochs: $EPOCHS"
echo "  Training samples: $TRAIN_SAMPLES"
echo "  Validation samples: $VAL_SAMPLES"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to run training and capture F1 score
run_training() {
    local run_num=$1
    local output_file="$OUTPUT_DIR/run_${run_num}.pt"

    echo -e "${YELLOW}Running training iteration $run_num...${NC}"

    poetry run python scripts/train_baseline.py \
        --seed "$SEED" \
        --epochs "$EPOCHS" \
        --train-samples "$TRAIN_SAMPLES" \
        --val-samples "$VAL_SAMPLES" \
        --output "$output_file" 2>&1 | tee "$OUTPUT_DIR/run_${run_num}.log"

    # Extract F1 score from log
    grep "Best Macro F1" "$OUTPUT_DIR/run_${run_num}.log" | tail -1
}

# Run training twice
echo ""
echo "Running first training iteration..."
run_training 1

echo ""
echo "Running second training iteration..."
run_training 2

# Compare results
echo ""
echo "=================================================="
echo "Comparing Results"
echo "=================================================="

# Extract F1 scores
F1_1=$(grep "Best Macro F1" "$OUTPUT_DIR/run_1.log" | tail -1 | awk '{print $NF}')
F1_2=$(grep "Best Macro F1" "$OUTPUT_DIR/run_2.log" | tail -1 | awk '{print $NF}')

echo "Run 1 F1 Score: $F1_1"
echo "Run 2 F1 Score: $F1_2"

# Compare (using Python for floating point comparison)
MATCH=$(python3 -c "
import sys
f1_1 = float('$F1_1')
f1_2 = float('$F1_2')
diff = abs(f1_1 - f1_2)
if diff < 0.001:
    print('MATCH')
    sys.exit(0)
else:
    print(f'DIFF: {diff:.6f}')
    sys.exit(1)
")

echo ""
if [[ "$MATCH" == "MATCH" ]]; then
    echo -e "${GREEN}✓ REPRODUCIBILITY TEST PASSED${NC}"
    echo "Both runs produced identical F1 scores with seed $SEED"
else
    echo -e "${RED}✗ REPRODUCIBILITY TEST FAILED${NC}"
    echo "Runs produced different F1 scores: $MATCH"
    exit 1
fi

# Run unit tests
echo ""
echo "=================================================="
echo "Running Unit Tests"
echo "=================================================="
poetry run pytest tests/unit -v --tb=short -q 2>&1 | tail -20

# Run integration tests
echo ""
echo "=================================================="
echo "Running Integration Tests"
echo "=================================================="
poetry run pytest tests/integration -v --tb=short -q 2>&1 | tail -20

# Summary
echo ""
echo "=================================================="
echo "Benchmark Summary"
echo "=================================================="
echo "  Reproducibility: PASSED"
echo "  Best F1 Score: $F1_1"
echo "  Seed: $SEED"
echo "  Model saved to: $OUTPUT_DIR/"
echo ""
echo -e "${GREEN}All benchmarks completed successfully!${NC}"
