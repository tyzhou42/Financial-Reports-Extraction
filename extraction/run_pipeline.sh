#!/usr/bin/env bash
if [[ -z "${BASH_VERSION:-}" ]]; then
  exec /usr/bin/env bash "$0" "$@"
fi
set -euo pipefail

# Minimal viability run for the extraction pipeline.
# Adjust the values in the "Defaults" section to run larger batches.

# ----------------------------
# Defaults (edit here)
# ----------------------------
PYTHON_BIN="${PYTHON_BIN:-python3}"
INPUT_DIR="data/reports"
ANALYST_REPORTS_DIR="datasets/analyst_reports"
	TXT_DIR="data/txts"
	TXT_NO_DISCLAIMER_DIR="data/txts_no_disclaimer"
	MAX_WORKERS=5
	# Leave blank to process all datasets/tickers under the input directory.
	DATASETS=""
	COMPANIES=""
	START_DATE=""
	END_DATE=""
POSTPROCESS=true
TEMPLATE_REMOVAL=true
# Set to true to run the full LLM/VLM pipeline.
# Set to false to run text-only extraction mode.
USE_API=true

# ----------------------------
# Resolve paths
# ----------------------------
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
INPUT_DIR="${ROOT_DIR}/${INPUT_DIR}"
ANALYST_REPORTS_DIR="${ROOT_DIR}/${ANALYST_REPORTS_DIR}"
TXT_DIR="${ROOT_DIR}/${TXT_DIR}"
TXT_NO_DISCLAIMER_DIR="${ROOT_DIR}/${TXT_NO_DISCLAIMER_DIR}"

mkdir -p "${INPUT_DIR}"

# ----------------------------
# Run pipeline
# ----------------------------
echo "Running extraction pipeline..."
echo "Input directory: ${INPUT_DIR}"
echo "Analyst reports output: ${ANALYST_REPORTS_DIR}"
echo "TXT directory: ${TXT_DIR}"
echo "TXT no disclaimer: ${TXT_NO_DISCLAIMER_DIR}"
echo "Max workers: ${MAX_WORKERS}"
echo "Datasets filter: ${DATASETS:-all}"
echo "Companies filter: ${COMPANIES:-all}"
echo "Date range: ${START_DATE:-min} -> ${END_DATE:-max}"
echo "Postprocess: ${POSTPROCESS}"
echo "Template removal: ${TEMPLATE_REMOVAL}"
echo "Use API (LLM/VLM): ${USE_API}"
(
  cd "${ROOT_DIR}"
  ARGS=()
  if [[ "${POSTPROCESS}" == "true" ]]; then
    ARGS+=("--postprocess")
  else
    ARGS+=("--no-postprocess")
  fi
  if [[ "${TEMPLATE_REMOVAL}" == "true" ]]; then
    ARGS+=("--template-removal")
  else
    ARGS+=("--no-template-removal")
  fi

  if [[ "${USE_API,,}" == "true" ]]; then
    ARGS+=("--no-text-only")
  else
    ARGS+=("--text-only")
  fi

  EXTRACTION_INPUT_DIRECTORY="${INPUT_DIR}" \
  EXTRACTION_ANALYST_REPORTS_DIRECTORY="${ANALYST_REPORTS_DIR}" \
  EXTRACTION_TXT_DIRECTORY="${TXT_DIR}" \
  EXTRACTION_TXT_NO_DISCLAIMER_DIRECTORY="${TXT_NO_DISCLAIMER_DIR}" \
  EXTRACTION_MAX_WORKERS="${MAX_WORKERS}" \
  EXTRACTION_DATASETS="${DATASETS}" \
  EXTRACTION_COMPANIES="${COMPANIES}" \
  EXTRACTION_START_DATE="${START_DATE}" \
  EXTRACTION_END_DATE="${END_DATE}" \
  "${PYTHON_BIN}" "extraction/main.py" "${ARGS[@]}"
)
