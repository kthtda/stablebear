#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
ROOT=$(realpath "${SCRIPT_DIR}/..")

usage() {
  cat << EOF
Usage: $(basename "$0") [OPTIONS]

Run valgrind or helgrind on the stablebear test suite and open an HTML report.

Options:
  -r, --root    <dir>            Project root directory (default: $ROOT)
  -b, --build   <dir>            Build directory containing sb_test
                                 (default: <root>/../cmake-build-debug)
  -o, --output  <dir>            Output directory for the HTML report
                                 (default: \$TMPDIR/<tool>_<suite>_report)
  -t, --tool    valgrind|helgrind  Analysis tool to use (default: helgrind)
  -s, --suite   pytest|gtest       Test suite to run (default: gtest)
  -h, --help                     Show this help message and exit
EOF
}

PARSED=$(getopt \
  --options r:b:o:t:s:h \
  --longoptions root:,build:,output:,tool:,suite:,help \
  --name "$(basename "$0")" \
  -- "$@")

if [ $? -ne 0 ]; then
  usage >&2; exit 1
fi

eval set -- "$PARSED"

BUILD=""
SUITE="gtest"
TOOL="helgrind"
OUT_DIR=""

while true; do
  case $1 in
    -r|--root)   ROOT=$(realpath "$2");    shift 2 ;;
    -b|--build)  BUILD=$(realpath "$2");   shift 2 ;;
    -o|--output) OUT_DIR=$(realpath "$2"); shift 2 ;;
    -t|--tool)   TOOL=$2;                  shift 2 ;;
    -s|--suite)  SUITE=$2;                 shift 2 ;;
    -h|--help)   usage; exit 0 ;;
    --) shift; break ;;
    *)  echo "Unexpected option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ "$TOOL" != "valgrind" && "$TOOL" != "helgrind" ]]; then
  echo "Error: --tool must be valgrind or helgrind." >&2; usage >&2; exit 1
fi

if [[ "$SUITE" != "pytest" && "$SUITE" != "gtest" ]]; then
  echo "Error: --suite must be pytest or gtest." >&2; usage >&2; exit 1
fi

BUILD=${BUILD:-$(realpath "${ROOT}/cmake-build-debug")}
OUT_DIR=${OUT_DIR:-"${TMPDIR:-/tmp}/${TOOL}_${SUITE}_report"}

SB_TEST="${BUILD}/sb_test"
if [ ! -x "$SB_TEST" ]; then
  echo "Error: $SB_TEST not found or not executable." >&2
  exit 1
fi

XML="${TMPDIR:-/tmp}/${TOOL}-${SUITE}.xml"

rm -rf "$OUT_DIR" "$XML"

if [ "$TOOL" = "valgrind" ]; then
  VG_TOOL="memcheck"
else
  VG_TOOL="$TOOL"
fi
VALGRIND_CMD=(valgrind --tool="$VG_TOOL" --xml=yes --xml-file="$XML" --num-callers=20 --history-level=full)
echo "XML file: ${XML}"
if [ "$SUITE" = "pytest" ]; then
  PYTHONMALLOC=malloc "${VALGRIND_CMD[@]}" python -m pytest "${ROOT}/test/" -x -q || true
else
  "${VALGRIND_CMD[@]}" "$SB_TEST" || true
fi

valgrind-ci "$XML" --output-dir="$OUT_DIR" --source-dir="${ROOT}" --summary
echo ""
echo "Report: file://${OUT_DIR}/index.html"

if command -v xdg-open &> /dev/null; then
  xdg-open "${OUT_DIR}/index.html"
elif command -v open &> /dev/null; then
  open "${OUT_DIR}/index.html"
fi
