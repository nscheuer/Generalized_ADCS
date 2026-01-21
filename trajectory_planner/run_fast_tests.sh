#!/bin/bash
# Run fast trajectory planner tests
#
# Usage:
#   ./run_fast_tests.sh           # Run all fast tests
#   ./run_fast_tests.sh --minimal # Run only minimal test
#   ./run_fast_tests.sh --quick   # Run quick planner tests
#   ./run_fast_tests.sh --bench   # Run benchmark

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEBUG_DIR="$PROJECT_ROOT/debug/debug_controllers/debug_plan_and_track"
TEST_DIR="$PROJECT_ROOT/testing/test_controllers"

# Activate virtual environment if it exists
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# Ensure build is up to date
echo "Checking build..."
cd "$SCRIPT_DIR/build"
if [ -f "build.ninja" ]; then
    ninja tplaunch pysat 2>/dev/null || true
else
    make -j$(nproc) tplaunch pysat 2>/dev/null || true
fi
cd "$PROJECT_ROOT"

echo ""
echo "=========================================="
echo "Fast Trajectory Planner Tests"
echo "=========================================="
echo ""

run_minimal() {
    echo "[1] Running minimal ALTRO test..."
    python "$DEBUG_DIR/minimal_altro_test.py"
}

run_quick() {
    echo "[2] Running quick planner tests..."
    python "$DEBUG_DIR/quick_planner_tests.py"
}

run_fast() {
    echo "[3] Running fast ALTRO test..."
    python "$DEBUG_DIR/fast_altro_test.py" --iterations 3
}

run_ultrafast() {
    echo "[4] Running ultrafast ALTRO test..."
    python "$DEBUG_DIR/ultrafast_altro_test.py" --iterations 10
}

run_benchmark() {
    echo "[5] Running benchmark..."
    python "$DEBUG_DIR/benchmark_altro.py" --quick
}

run_pytest() {
    echo "[6] Running pytest fast tests..."
    pytest "$TEST_DIR/test_plan_and_track_fast.py" -v --tb=short
}

# Parse arguments
case "${1:-all}" in
    --minimal|-m)
        run_minimal
        ;;
    --quick|-q)
        run_quick
        ;;
    --fast|-f)
        run_fast
        ;;
    --ultrafast|-u)
        run_ultrafast
        ;;
    --bench|-b)
        run_benchmark
        ;;
    --pytest|-p)
        run_pytest
        ;;
    --all|all)
        run_minimal
        echo ""
        run_quick
        echo ""
        run_fast
        ;;
    --full)
        run_minimal
        echo ""
        run_quick
        echo ""
        run_fast
        echo ""
        run_ultrafast
        echo ""
        run_benchmark
        echo ""
        run_pytest
        ;;
    --help|-h)
        echo "Usage: $0 [option]"
        echo ""
        echo "Options:"
        echo "  --minimal, -m    Run minimal ALTRO test only"
        echo "  --quick, -q      Run quick planner tests"
        echo "  --fast, -f       Run fast ALTRO test"
        echo "  --ultrafast, -u  Run ultrafast ALTRO test"
        echo "  --bench, -b      Run benchmark"
        echo "  --pytest, -p     Run pytest tests"
        echo "  --all, all       Run minimal + quick + fast (default)"
        echo "  --full           Run all tests including benchmark"
        echo "  --help, -h       Show this help"
        exit 0
        ;;
    *)
        echo "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Tests completed!"
echo "=========================================="
