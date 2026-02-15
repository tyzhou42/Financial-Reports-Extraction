# `extraction/` overview

This folder contains the code that turns analyst report PDFs into CSV datasets.

## How the pipeline works (high level)

1. Find PDFs under the input directory (by default `data/reports/`).
2. For each PDF:
   - Extract text (direct PDF text first, then OCR if needed).
   - Optionally extract EPS tables and call an LLM to fill the target fields.
3. Write one CSV per dataset folder under `datasets/analyst_reports/`.
4. Optionally run a post-processing cleaner to standardize dates, prices, EPS formats, and names.

## Main entrypoint

- `main.py`
  - The only real pipeline runner.
  - Supports:
    - `--text-only` / `--no-text-only`: skip or enable the LLM/table feature extraction.
    - `--postprocess`: run `postprocessor.py` after writing CSVs.
    - `--template-removal`: generate `txts_no_disclaimer/` and use that text in outputs.
  - Runs reports in parallel with `ProcessPoolExecutor`.
  - Writes logs to `logs/extraction.log` (and also prints log lines to the console).

## Extraction modules (split by responsibility)

- `text_extractor.py`
  - Extracts plain text from a PDF.
  - Tries direct text extraction first (`pypdf`), then falls back to OCR (`pdf2image` + `pytesseract`).

- `table_extractor.py`
  - Extracts EPS-related tables from PDFs and returns them as Markdown.
  - Uses Camelot when possible.
  - Has a VLM fallback for EPS tables when Camelot fails (renders pages and asks a vision model to transcribe a table).

- `feature_extractor.py`
  - Fills target fields using LLM prompts.
  - Uses a cheap model for non-EPS fields and a strong model for EPS-related fields.
  - Also contains a regex-based extractor (currently not wired into `main.py`).

- `image_extractor.py`
  - Utilities for extracting embedded images from PDFs and running VLM-based feature extraction on them.
  - Not currently enabled by default in `main.py`.

- `extractor.py`
  - Compatibility facade that re-exports functions from the modules above.
  - Exists so older imports like `from extractor import get_text_from_document` still work.

## Template removal (optional)

- `template_removal.py`
  - Removes repeated disclaimer/boilerplate paragraphs from `.txt` exports.
  - Used when `--template-removal` is enabled; outputs cleaned files into `txts_no_disclaimer/`.

## LLM/VLM client code

- `llm_client.py`
  - Provider adapters (Gemini and OpenAI) behind a small interface.
  - You should consider turn to cheaper LLM apis like Deepseek.
  - Returns a provider-specific client based on config.
  - Also contains `required_api_keys()` to tell `main.py` which environment variables are needed.

- `utils.py`
  - Shared helpers used across the pipeline:
    - LLM call wrapper with retries and JSON parsing.
    - VLM call wrapper with retries and JSON parsing.
    - Directory setup/cleanup helpers.
    - A helper to export PDFs to `.txt` files in bulk.

## Configuration and selection

- `config.py`
  - Constants and settings only:
    - Directory paths (input/output/cache/logs).
    - Model names and providers.
    - Target feature list and prompt templates.

- `report_selection.py`
  - Finds PDFs and builds a list of work items (`ReportRecord`).
  - Expects a directory structure like: `dataset/ticker/report.pdf`.
  - Optional filters are driven by environment variables (dataset, ticker, date range).

## Post-processing (cleaning outputs)

- `postprocessor.py`
  - Reads an extracted CSV and produces a cleaned CSV:
    - Normalizes dates to `YYYY-MM-DD`.
    - Parses numeric target prices and extracts currency codes.
    - Normalizes EPS numbers and EPS period labels.
    - Standardizes bank, analyst, and company names.
  - Loads bank name mappings from `resources/bank_mappings.json`.

## Resources and generated artifacts

- `resources/bank_mappings.json`
  - A dictionary of “messy bank name” -> “standard bank name”.
  - Used by `postprocessor.py`.

- `project_exporter.py`
  - Helper that generates a Markdown dump of the code for auditing/review.

- `project_dump.md`
  - The generated output from `project_exporter.py`.

## Logging and run scripts

- `logging_utils.py`
  - One place to configure Python `logging` consistently for console + file.

- `run_pipeline.sh`
  - Convenience script to run the pipeline with environment variables set.
  - Controls whether the run is full mode or text-only mode.

- `logs/`
  - Log files and run summaries.

## Local-only files

- `.env`
  - Local environment variables (API keys).
  - Keep secrets out of git.
