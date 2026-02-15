"""Main execution script for the analyst report extraction pipeline."""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from config import (
    ANALYST_REPORTS_DIRECTORY,
    INPUT_DIRECTORY,
    LOG_DIRECTORY,
    TABLE_KEYWORDS,
    TARGET_FEATURES,
    TXT_DIRECTORY,
    TXT_NO_DISCLAIMER_DIRECTORY,
    VLM_PROVIDER,
)
from llm_client import required_api_keys
from logging_utils import setup_logging
from postprocessor import run_postprocessing
from text_extractor import get_text_from_document
from utils import clear_directory_contents, setup_directories

try:
    from .report_selection import ReportRecord, parse_date_input, select_reports
except ImportError:
    from report_selection import ReportRecord, parse_date_input, select_reports


logger = logging.getLogger(__name__)


def process_report_pipeline(record: ReportRecord, text_only: bool = False) -> Dict[str, Any]:
    """Execute extraction for a single report."""
    file_path = record.path
    logger.info("Processing report: %s", file_path.name)

    document_text = get_text_from_document(file_path)
    if document_text:
        txt_output_path = TXT_DIRECTORY / f"{file_path.stem}.txt"
        try:
            TXT_DIRECTORY.mkdir(parents=True, exist_ok=True)
            txt_output_path.write_text(document_text, encoding="utf-8")
            logger.info("Saved text export: %s", txt_output_path.name)
        except Exception as exc:
            logger.warning("Unable to write text export for %s: %s", file_path.name, exc)

    final_features: Dict[str, Any] = {}
    if not text_only:
        from feature_extractor import fill_missing_features_with_llm
        from table_extractor import extract_relevant_tables_as_markdown

        table_keywords = [keyword for keywords in TABLE_KEYWORDS.values() for keyword in keywords]
        table_markdown = extract_relevant_tables_as_markdown(file_path, table_keywords)

        features: Dict[str, Any] = {key: None for key in TARGET_FEATURES}
        if document_text:
            final_features = fill_missing_features_with_llm(document_text, table_markdown, features)
        else:
            logger.warning("No text extracted for %s; skipping LLM feature extraction", file_path.name)
    else:
        final_features = {key: None for key in TARGET_FEATURES}

    found_count = sum(1 for key in TARGET_FEATURES if final_features.get(key))
    logger.info(
        "Completed %s: found %s/%s features (text_only=%s)",
        file_path.name,
        found_count,
        len(TARGET_FEATURES),
        text_only,
    )

    result: Dict[str, Any] = {
        "dataset_name": record.dataset,
        "company_ticker": record.ticker,
        "report_name": record.report_name,
        "report_text": document_text or "",
    }
    for feature in TARGET_FEATURES:
        result[feature] = final_features.get(feature)
    return result


def _parse_env_list(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    tokens = [token.strip() for token in re.split(r"[,\s]+", value) if token.strip()]
    return tokens or None


def _env_truthy(value: Optional[str]) -> bool:
    if not value:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the financial report extraction pipeline.")
    parser.add_argument(
        "--text-only",
        action=argparse.BooleanOptionalAction,
        default=_env_truthy(os.getenv("EXTRACTION_TEXT_ONLY")),
        help="Skip LLM/VLM extraction and export text-only outputs.",
    )
    parser.add_argument(
        "--postprocess",
        action=argparse.BooleanOptionalAction,
        default=_env_truthy(os.getenv("EXTRACTION_POSTPROCESS")),
        help="Run CSV postprocessing after extraction.",
    )
    parser.add_argument(
        "--template-removal",
        action=argparse.BooleanOptionalAction,
        default=_env_truthy(os.getenv("EXTRACTION_TEMPLATE_REMOVAL")),
        help="Generate txts_no_disclaimer and use cleaned text in outputs.",
    )
    return parser.parse_args()


def _run_template_removal(corpus_dir: Path, output_dir: Path) -> None:
    try:
        from template_removal import RemovalConfig, process_corpus
    except Exception:
        logger.exception("Template removal import failed")
        return

    config = RemovalConfig(
        tail_start_ratio=0.55,
        min_score=1.0,
        extra_keywords=(),
    )
    process_corpus(
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        marked_dir=None,
        report_csv=None,
        config=config,
    )


def _validate_api_keys(text_only: bool) -> bool:
    if text_only:
        return True

    required_keys = set(required_api_keys().values())
    if VLM_PROVIDER == "gemini":
        required_keys.add("GOOGLE_API_KEY")

    missing_keys = [key for key in sorted(required_keys) if not os.getenv(key)]
    if missing_keys:
        logger.error(
            "Missing required API keys: %s. Set them in .env before running.",
            ", ".join(missing_keys),
        )
        return False
    return True


def _load_cleaned_texts() -> Dict[str, str]:
    cleaned_text_map: Dict[str, str] = {}
    if not TXT_NO_DISCLAIMER_DIRECTORY.exists():
        return cleaned_text_map

    for path in TXT_NO_DISCLAIMER_DIRECTORY.glob("*.txt"):
        try:
            cleaned_text_map[path.stem] = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            logger.warning("Unable to read cleaned text %s: %s", path.name, exc)
    return cleaned_text_map


def main() -> None:
    load_dotenv()
    log_path = LOG_DIRECTORY / "extraction.log"
    setup_logging(log_file=log_path)
    args = _parse_args()

    logger.info("Starting extraction run (text_only=%s)", args.text_only)
    if not _validate_api_keys(args.text_only):
        return

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key and not args.text_only:
        import google.generativeai as genai

        genai.configure(api_key=google_api_key)

    setup_directories([ANALYST_REPORTS_DIRECTORY, TXT_DIRECTORY, TXT_NO_DISCLAIMER_DIRECTORY, LOG_DIRECTORY])

    if not INPUT_DIRECTORY.exists():
        logger.error("Input directory not found: %s", INPUT_DIRECTORY.resolve())
        return

    dataset_filters = _parse_env_list(os.getenv("EXTRACTION_DATASETS") or os.getenv("EXTRACTION_DATASET"))
    company_filters = _parse_env_list(os.getenv("EXTRACTION_COMPANIES") or os.getenv("EXTRACTION_COMPANY"))
    start_date_raw = os.getenv("EXTRACTION_START_DATE")
    end_date_raw = os.getenv("EXTRACTION_END_DATE")
    start_date = parse_date_input(start_date_raw)
    end_date = parse_date_input(end_date_raw)
    if start_date_raw and start_date is None:
        logger.warning("Invalid EXTRACTION_START_DATE=%r; ignoring", start_date_raw)
    if end_date_raw and end_date is None:
        logger.warning("Invalid EXTRACTION_END_DATE=%r; ignoring", end_date_raw)

    records = select_reports(
        INPUT_DIRECTORY,
        dataset_filters=dataset_filters,
        company_filters=company_filters,
        start_date=start_date,
        end_date=end_date,
    )
    if not records:
        logger.warning("No documents found in %s for the requested filters", INPUT_DIRECTORY)
        return

    clear_directory_contents(TXT_DIRECTORY)
    clear_directory_contents(TXT_NO_DISCLAIMER_DIRECTORY)

    max_workers_env = os.getenv("EXTRACTION_MAX_WORKERS")
    if max_workers_env:
        try:
            max_workers = max(1, int(max_workers_env))
        except ValueError:
            logger.warning("Invalid EXTRACTION_MAX_WORKERS=%r; using default 8", max_workers_env)
            max_workers = 8
    else:
        max_workers = 8

    logger.info("Found %s documents to process", len(records))
    logger.info("Using up to %s workers", max_workers)

    all_results: List[Dict[str, Any]] = []
    success_count = 0
    exception_count = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_record = {
            executor.submit(process_report_pipeline, record, args.text_only): record for record in records
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_record),
            total=len(records),
            desc="Processing Reports",
        ):
            record = future_to_record[future]
            try:
                result = future.result()
                all_results.append(result)
                success_count += 1
            except Exception as exc:
                logger.exception("Critical error processing %s: %s", record.report_name, exc)
                exception_count += 1

    logger.info("Run summary: success=%s, exception=%s", success_count, exception_count)
    if not all_results:
        logger.warning("Processing finished, but no data was extracted")
        return

    if args.template_removal:
        logger.info("Running template removal")
        _run_template_removal(TXT_DIRECTORY, TXT_NO_DISCLAIMER_DIRECTORY)

    cleaned_text_map = _load_cleaned_texts() if args.template_removal else {}

    results_by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for result in all_results:
        dataset_name = result.get("dataset_name") or "unknown"
        if args.template_removal:
            report_name = result.get("report_name") or ""
            stem = Path(report_name).stem
            cleaned_text = cleaned_text_map.get(stem)
            if cleaned_text is not None:
                result["report_text"] = cleaned_text
        results_by_dataset[dataset_name].append(result)

    for dataset_name, rows in results_by_dataset.items():
        dataframe = pd.DataFrame(rows)
        for column in ("dataset_name", "company_ticker"):
            if column in dataframe.columns:
                dataframe = dataframe.drop(columns=column)

        column_order = ["report_name", "report_text"] + TARGET_FEATURES
        dataframe = dataframe.reindex(columns=[column for column in column_order if column in dataframe.columns])
        dataframe.insert(0, "index", range(len(dataframe)))

        output_path = ANALYST_REPORTS_DIRECTORY / f"{dataset_name}.csv"
        dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info("Saved dataset output: %s", output_path)

        if args.postprocess:
            cleaned_output_path = ANALYST_REPORTS_DIRECTORY / f"{dataset_name}_cleaned.csv"
            run_postprocessing(output_path, cleaned_output_path)


if __name__ == "__main__":
    main()
