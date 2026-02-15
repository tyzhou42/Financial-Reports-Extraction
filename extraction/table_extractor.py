"""Table extraction helpers focused on EPS content."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

import camelot
import fitz

from config import VLM_EPS_TABLE_PROMPT, get_feature_rules_subset
from utils import call_vlm_api


logger = logging.getLogger(__name__)


_QUARTER_PATTERN = re.compile(
    r"(\b(?:(?:Q[1-4]|[1-4]Q)(?:\s*|')(?:(?:FY|CY)?\s*(?:'\d{2}|\d{2,4})[EA]?)|(?:(?:FY|CY)?\s*(?:'\d{2}|\d{2,4})[EA]?)\s+(?:Q[1-4]|[1-4]Q)|(?:Q[1-4]|[1-4]Q)|(?:qtr\D*[1-4])|(?:quarter\D*[1-4])|(?:q\D*[1-4]))\b)",
    re.IGNORECASE,
)


def _contains_quarterly_eps_markers(markdown_text: str) -> bool:
    """Return True when markdown appears to contain quarterly EPS table content."""
    if not markdown_text:
        return False
    lowered = markdown_text.lower()
    if "eps" not in lowered and "earnings per share" not in lowered:
        return False
    return bool(_QUARTER_PATTERN.search(lowered))


def extract_eps_tables_with_vlm(file_path: Path, keywords: List[str], max_pages: int = 5) -> str:
    """Use VLM to recover EPS table markdown when Camelot cannot provide it."""
    try:
        doc = fitz.open(file_path)
    except Exception as exc:
        logger.warning("Unable to open PDF for VLM table extraction %s: %s", file_path.name, exc)
        return ""

    try:
        keywords_lower = [keyword.lower() for keyword in keywords]
        candidate_pages: List[int] = []

        for page_index in range(len(doc)):
            try:
                page = doc.load_page(page_index)
            except Exception as page_exc:
                logger.warning("Failed to load page %s for %s: %s", page_index + 1, file_path.name, page_exc)
                continue

            page_text = page.get_text("text") or ""
            normalized = page_text.lower()
            if any(keyword in normalized for keyword in keywords_lower):
                candidate_pages.append(page_index)
            if len(candidate_pages) >= max_pages:
                break

        if not candidate_pages:
            candidate_pages = list(range(min(max_pages, len(doc))))

        rules = get_feature_rules_subset(["current_period_eps", "next_period_eps"])

        for page_index in candidate_pages:
            try:
                page = doc.load_page(page_index)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                image_bytes = pix.tobytes("png")
            except Exception as render_exc:
                logger.warning(
                    "Failed to render page %s for %s: %s",
                    page_index + 1,
                    file_path.name,
                    render_exc,
                )
                continue

            prompt = VLM_EPS_TABLE_PROMPT.format(page_number=page_index + 1, feature_rules=rules)
            vlm_result = call_vlm_api(image_bytes, prompt, response_mime_type="application/json")
            table_markdown = vlm_result.get("table_markdown") if vlm_result else None
            if not table_markdown:
                continue

            if _contains_quarterly_eps_markers(table_markdown):
                logger.info("VLM recovered EPS table from page %s (%s)", page_index + 1, file_path.name)
                return f"Table from Page {page_index + 1}:\n{table_markdown}"

            logger.info(
                "VLM page %s for %s lacked quarterly EPS markers",
                page_index + 1,
                file_path.name,
            )

        logger.info("VLM could not recover EPS tables for %s", file_path.name)
        return ""
    finally:
        doc.close()


def extract_relevant_tables_as_markdown(file_path: Path, keywords: List[str]) -> str:
    """Extract document tables and return merged EPS-focused markdown."""
    logger.info("Extracting relevant tables from %s", file_path.name)
    final_tables = []

    logger.info("Trying Camelot stream mode for %s", file_path.name)
    try:
        tables_stream = camelot.read_pdf(str(file_path), pages="all", flavor="stream")
        if tables_stream.n > 0 and any(tbl.parsing_report["accuracy"] >= 90 for tbl in tables_stream):
            logger.info("Camelot stream found %s tables for %s", tables_stream.n, file_path.name)
            final_tables = tables_stream
    except Exception as exc:
        logger.warning("Stream extraction failed for %s: %s", file_path.name, exc)

    if not final_tables:
        logger.info("Falling back to Camelot lattice mode for %s", file_path.name)
        try:
            tables_lattice = camelot.read_pdf(str(file_path), pages="all", flavor="lattice")
            if tables_lattice.n > 0:
                logger.info("Camelot lattice found %s tables for %s", tables_lattice.n, file_path.name)
                final_tables = tables_lattice
        except Exception as exc:
            logger.warning("Lattice extraction failed for %s: %s", file_path.name, exc)

    if not final_tables:
        logger.info("No Camelot tables extracted for %s", file_path.name)
        if any("eps" in keyword.lower() for keyword in keywords):
            logger.info("Trying VLM EPS fallback for %s", file_path.name)
            vlm_tables = extract_eps_tables_with_vlm(file_path, keywords)
            if vlm_tables:
                return vlm_tables
        return ""

    relevant_tables = []
    unique_tables = set()
    use_keyword_filter = False

    for table in final_tables:
        dataframe = table.df
        dataframe_as_string = dataframe.to_string()
        dataframe_lower = dataframe_as_string.lower()

        if use_keyword_filter:
            matched_keyword = any(keyword.lower() in dataframe_lower for keyword in keywords)
            if not matched_keyword:
                continue

        if not _contains_quarterly_eps_markers(dataframe_as_string):
            continue

        table_hash = tuple(dataframe.iloc[0].tolist())
        if table_hash not in unique_tables:
            relevant_tables.append((table.page, dataframe))
            unique_tables.add(table_hash)

    if not relevant_tables:
        logger.info("No Camelot tables passed quarterly EPS checks for %s", file_path.name)
        if any("eps" in keyword.lower() for keyword in keywords):
            logger.info("Trying VLM EPS fallback after Camelot filtering for %s", file_path.name)
            vlm_tables = extract_eps_tables_with_vlm(file_path, keywords)
            if vlm_tables:
                return vlm_tables
        return ""

    logger.info("Converting %s relevant tables to markdown for %s", len(relevant_tables), file_path.name)
    return "\n\n---\n\n".join(
        [f"Table from Page {page}:\n" + dataframe.to_markdown(index=False) for page, dataframe in relevant_tables]
    )

