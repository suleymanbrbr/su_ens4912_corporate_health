#!/usr/bin/env bash
# =============================================================================
# run_all_tests.sh — SUT Corporate Health Full Test Suite Runner
# =============================================================================
# Usage:
#   cd tests
#   chmod +x run_all_tests.sh
#   ./run_all_tests.sh              # All unit + integration
#   ./run_all_tests.sh --unit       # Unit tests only
#   ./run_all_tests.sh --integration # Integration only (requires DB)
#   ./run_all_tests.sh --rag        # RAG quality tests (requires backend)
#   ./run_all_tests.sh --stress     # Stress test (headless, 60s)
#   ./run_all_tests.sh --all        # Everything
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$PROJECT_ROOT/backend"
REPORT_DIR="$SCRIPT_DIR/reports"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p "$REPORT_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

banner() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

check_deps() {
    banner "Checking dependencies..."
    if ! python -c "import pytest" 2>/dev/null; then
        echo -e "${YELLOW}Installing test dependencies...${NC}"
        pip install -r "$SCRIPT_DIR/requirements-test.txt" -q
    fi
    echo -e "${GREEN}✓ Dependencies OK${NC}"
}

run_unit() {
    banner "🧪 Running Unit Tests"
    cd "$SCRIPT_DIR"
    python -m pytest unit/ \
        --cov="$BACKEND_DIR" \
        --cov-report=html:"$REPORT_DIR/coverage_${TIMESTAMP}" \
        --cov-report=term-missing \
        --html="$REPORT_DIR/unit_report_${TIMESTAMP}.html" \
        --self-contained-html \
        -v \
        "$@"
    echo -e "\n${GREEN}✓ Unit tests complete. Coverage report: $REPORT_DIR/coverage_${TIMESTAMP}/index.html${NC}"
}

run_integration() {
    banner "🔗 Running Integration Tests"
    cd "$SCRIPT_DIR"
    python -m pytest integration/ \
        --html="$REPORT_DIR/integration_report_${TIMESTAMP}.html" \
        --self-contained-html \
        -v \
        "$@"
    echo -e "\n${GREEN}✓ Integration tests complete${NC}"
}

run_rag() {
    banner "🤖 Running RAG Quality Tests"
    cd "$SCRIPT_DIR"
    python -m pytest rag_quality/ \
        -m slow \
        --html="$REPORT_DIR/rag_quality_report_${TIMESTAMP}.html" \
        --self-contained-html \
        -v -s \
        "$@"
    echo -e "\n${GREEN}✓ RAG quality tests complete. Results: $SCRIPT_DIR/rag_quality/rag_quality_results.json${NC}"
}

run_stress() {
    banner "⚡ Running Stress Tests (60s, 50 users)"
    STRESS_REPORT="$REPORT_DIR/stress_report_${TIMESTAMP}.html"
    locust \
        -f "$SCRIPT_DIR/stress/locustfile.py" \
        --host=http://localhost:8000 \
        --headless \
        -u 50 \
        -r 10 \
        -t 60s \
        --html="$STRESS_REPORT"
    echo -e "\n${GREEN}✓ Stress test complete. Report: $STRESS_REPORT${NC}"
}

# ── Parse arguments ───────────────────────────────────────────────────────────
check_deps

ARGS=("$@")
MODE="${1:-}"

case "$MODE" in
    --unit)
        run_unit "${ARGS[@]:1}"
        ;;
    --integration)
        run_integration "${ARGS[@]:1}"
        ;;
    --rag)
        run_rag "${ARGS[@]:1}"
        ;;
    --stress)
        run_stress
        ;;
    --all)
        run_unit
        run_integration
        run_rag
        run_stress
        ;;
    "")
        # Default: unit + integration
        run_unit
        run_integration
        ;;
    *)
        echo "Unknown option: $MODE"
        echo "Usage: $0 [--unit | --integration | --rag | --stress | --all]"
        exit 1
        ;;
esac

banner "✅ Test run complete! Reports saved to: $REPORT_DIR"
