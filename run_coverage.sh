#!/usr/bin/env bash
# run_coverage.sh — local coverage report generator
# Mirrors the coverage_report job in .github/workflows/analysis.yaml
#
# Usage:
#   ./run_coverage.sh [options]
#
# Options:
#   --skip-configure   Skip cmake configure step
#   --skip-build       Skip cmake build + install steps
#   --skip-cpp         Skip C++ test run
#   --skip-python      Skip Python test run
#   --skip-gcovr       Skip gcovr report generation
#   --open             Open coverage report in browser when done
#   --build-dir DIR    Build directory (default: build)
#   --out-dir DIR      Output directory for reports (default: covreport)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────────
BUILD_DIR="$REPO_ROOT/build"
OUT_DIR="$REPO_ROOT/covreport"
SKIP_CONFIGURE=0
SKIP_BUILD=0
SKIP_CPP=0
SKIP_PYTHON=0
SKIP_GCOVR=0
OPEN_BROWSER=0

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-configure) SKIP_CONFIGURE=1 ;;
    --skip-build)     SKIP_BUILD=1 ;;
    --skip-cpp)       SKIP_CPP=1 ;;
    --skip-python)    SKIP_PYTHON=1 ;;
    --skip-gcovr)     SKIP_GCOVR=1 ;;
    --open)           OPEN_BROWSER=1 ;;
    --build-dir)      BUILD_DIR="$2"; shift ;;
    --out-dir)        OUT_DIR="$2"; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
  shift
done

CPP_OUT="$OUT_DIR/cpp"
PY_OUT="$OUT_DIR/python"

echo "=== stablebear local coverage ==="
echo "Repo:      $REPO_ROOT"
echo "Build dir: $BUILD_DIR"
echo "Out dir:   $OUT_DIR"
echo

# ── Environment (mirrors CI) ──────────────────────────────────────────────────
export BUILD_WITH_CUDA=0
export SKIP_STUBGEN=1
export SKIP_BACK_COPY=1
export ENABLE_COVERAGE=1

# ── Configure ─────────────────────────────────────────────────────────────────
if [[ $SKIP_CONFIGURE -eq 0 ]]; then
  echo "--- Configure ---"
  cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Debug "$REPO_ROOT"
  echo
fi

# ── Build + install ───────────────────────────────────────────────────────────
if [[ $SKIP_BUILD -eq 0 ]]; then
  echo "--- Build ---"
  cmake --build "$BUILD_DIR" -j "$(nproc)"
  echo
  echo "--- Install ---"
  cmake --install "$BUILD_DIR"
  echo
fi

# ── C++ tests ─────────────────────────────────────────────────────────────────
if [[ $SKIP_CPP -eq 0 ]]; then
  echo "--- C++ tests ---"
  (cd "$REPO_ROOT/test" && "$BUILD_DIR/sb_test") || echo "(some C++ tests failed — coverage data still collected)"
  echo
fi

# ── Python tests ──────────────────────────────────────────────────────────────
if [[ $SKIP_PYTHON -eq 0 ]]; then
  echo "--- Python tests ---"
  mkdir -p "$PY_OUT"
  (cd "$REPO_ROOT/test" && python -m pytest . \
    --cov=stablebear \
    --cov-config="$REPO_ROOT/pyproject.toml" \
    --cov-report="html:$PY_OUT" \
    --cov-report="json:$PY_OUT/coverage.json" \
    --cov-report=term) || echo "(some Python tests failed — coverage data still collected)"
  touch "$PY_OUT/.keep"
  [[ -f "$PY_OUT/index.html" ]]   || python -m coverage html --directory="$PY_OUT" || true
  [[ -f "$PY_OUT/coverage.json" ]] || python -m coverage json -o "$PY_OUT/coverage.json" || true
  echo
fi

# ── gcovr ─────────────────────────────────────────────────────────────────────
if [[ $SKIP_GCOVR -eq 0 ]]; then
  echo "--- gcovr ---"
  #python -c "import gcovr" &>/dev/null || pip install "gcovr>=7.0"
  mkdir -p "$CPP_OUT"
  # Run from CPP_OUT so gcovr resolves relative paths the same way as CI
  (cd "$CPP_OUT" && python -m gcovr \
    --gcov-ignore-parse-errors=negative_hits.warn_once_per_file \
    -r . \
    --object-directory "$BUILD_DIR" \
    --filter "$REPO_ROOT/include/" \
    --filter "$REPO_ROOT/src/" \
    --html-details coverage.html \
    --json-summary coverage-summary.json \
    --print-summary)
  echo
fi

# ── AI export ─────────────────────────────────────────────────────────────────
AI_EXPORT="$OUT_DIR/ai-export.txt"
python3 - "$CPP_OUT/coverage-summary.json" "$PY_OUT/coverage.json" "$AI_EXPORT" \
  "$(cd "$REPO_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo unknown)" \
  "$(cd "$REPO_ROOT" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)" \
<<'PYEOF'
import json, sys, os

cpp_path, py_path, out_path, sha, branch = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]

def fmt_cpp(data):
    lines = ["=== C++ Coverage ==="]
    lines.append(f"Line coverage:     {data.get('line_percent', 'N/A'):.1f}%" if data.get('line_percent') is not None else "Line coverage:     N/A")
    lines.append(f"Function coverage: {data.get('function_percent', 'N/A'):.1f}%" if data.get('function_percent') is not None else "Function coverage: N/A")
    lines.append(f"Branch coverage:   {data.get('branch_percent', 'N/A'):.1f}%" if data.get('branch_percent') is not None else "Branch coverage:   N/A")
    if data.get('line_covered') is not None:
        lines.append(f"Lines covered:     {data['line_covered']} / {data['line_total']}")
    if data.get('function_covered') is not None:
        lines.append(f"Functions covered: {data['function_covered']} / {data['function_total']}")
    files = sorted(data.get('files', []), key=lambda f: f.get('filename', ''))
    if files:
        lines.append("")
        lines.append("Per-file breakdown:")
        for f in files:
            pct = f.get('line_percent')
            cov = f.get('line_covered', '?')
            tot = f.get('line_total', '?')
            name = f.get('filename', '').replace('/home/runner/work/stablebear/stablebear/', '')
            # strip local absolute prefix too
            for prefix in [os.path.expanduser('~'), '/NOBACKUP']:
                if name.startswith(prefix):
                    idx = name.find('stablebear/')
                    if idx != -1:
                        name = name[idx:]
            pct_str = f"{pct:.1f}%" if pct is not None else "?"
            lines.append(f"  {pct_str:7s}  {cov:4}/{tot:4}  {name}")
    return "\n".join(lines)

def fmt_py(data):
    lines = ["=== Python Coverage ==="]
    totals = data.get('totals', {})
    pct = totals.get('percent_covered')
    lines.append(f"Total: {pct:.1f}%" if pct is not None else "Total: N/A")
    lines.append(f"Covered lines: {totals.get('covered_lines', 'N/A')}")
    lines.append(f"Missing lines: {totals.get('missing_lines', 'N/A')}")
    lines.append("")
    lines.append("Per-file breakdown:")
    for path, info in sorted(data.get('files', {}).items()):
        summary = info.get('summary', {})
        p = summary.get('percent_covered')
        pct_str = f"{p:.1f}%" if p is not None else "?"
        display = path
        for prefix in [os.path.expanduser('~'), '/NOBACKUP']:
            if display.startswith(prefix):
                idx = display.find('stablebear/')
                if idx != -1:
                    display = display[idx:]
        lines.append(f"  {display}: {pct_str} ({summary.get('covered_lines',0)}/{summary.get('num_statements',0)} stmts)")
        missing = info.get('missing_lines', [])
        if missing:
            lines.append(f"    missing lines: {', '.join(str(x) for x in missing)}")
    return "\n".join(lines)

out = [f"# stablebear coverage report", f"SHA: {sha}", f"Branch: {branch}", ""]

try:
    with open(cpp_path) as f:
        out.append(fmt_cpp(json.load(f)))
except Exception as e:
    out.append(f"=== C++ Coverage ===\nFailed to load: {e}")

out.append("")

try:
    with open(py_path) as f:
        out.append(fmt_py(json.load(f)))
except Exception as e:
    out.append(f"=== Python Coverage ===\nFailed to load: {e}")

text = "\n".join(out)
with open(out_path, 'w') as f:
    f.write(text)
print(text)
PYEOF

# ── Summary ───────────────────────────────────────────────────────────────────
echo "=== Done ==="
echo "C++ report:    $CPP_OUT/coverage.html"
echo "Python report: $PY_OUT/index.html"
echo "AI export:     $AI_EXPORT"
echo

if [[ $OPEN_BROWSER -eq 1 ]]; then
  if command -v xdg-open &>/dev/null; then
    xdg-open "$CPP_OUT/coverage.html" &
    xdg-open "$PY_OUT/index.html" &
  elif command -v open &>/dev/null; then
    open "$CPP_OUT/coverage.html"
    open "$PY_OUT/index.html"
  else
    echo "(--open: no browser opener found)"
  fi
fi
