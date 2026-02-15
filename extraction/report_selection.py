from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
import re
from typing import Iterable, List, Optional, Sequence


_DATE_PATTERN = re.compile(
    r"(?<!\d)((?:19|20)\d{2})[-_.]?(0?[1-9]|1[0-2])[-_.]?(0?[1-9]|[12]\d|3[01])(?!\d)"
)


@dataclass(frozen=True)
class ReportRecord:
    dataset: str
    ticker: str
    path: Path
    report_name: str
    report_date: Optional[date]


def parse_date_input(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y_%m_%d", "%Y.%m.%d"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def extract_report_date(report_name: str) -> Optional[date]:
    for match in _DATE_PATTERN.finditer(report_name):
        year, month, day = match.groups()
        try:
            return date(int(year), int(month), int(day))
        except ValueError:
            continue
    return None


def _normalize_filters(values: Optional[Iterable[str]]) -> Optional[set[str]]:
    if not values:
        return None
    normalized = {str(value).strip().lower() for value in values if str(value).strip()}
    return normalized or None


def _normalize_substrings(values: Optional[Iterable[str]]) -> Optional[List[str]]:
    if not values:
        return None
    normalized = [str(value).strip().lower() for value in values if str(value).strip()]
    return normalized or None


def select_reports(
    root: Path,
    dataset_filters: Optional[Iterable[str]] = None,
    company_filters: Optional[Iterable[str]] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    bank_filters: Optional[Iterable[str]] = None,
) -> List[ReportRecord]:
    dataset_filter_set = _normalize_filters(dataset_filters)
    company_filter_set = _normalize_filters(company_filters)
    bank_filter_list = _normalize_substrings(bank_filters)

    records: List[ReportRecord] = []
    pdf_paths = sorted(root.rglob("*.pdf"))
    for pdf_path in pdf_paths:
        try:
            rel_parts = pdf_path.relative_to(root).parts
        except ValueError:
            continue
        if len(rel_parts) < 3:
            continue

        dataset = rel_parts[0]
        ticker = rel_parts[1]
        if dataset_filter_set and dataset.lower() not in dataset_filter_set:
            continue
        if company_filter_set and ticker.lower() not in company_filter_set:
            continue

        report_name = pdf_path.name
        report_date = extract_report_date(report_name)
        if start_date and (report_date is None or report_date < start_date):
            continue
        if end_date and (report_date is None or report_date > end_date):
            continue

        if bank_filter_list:
            stem_lower = pdf_path.stem.lower()
            if not any(token in stem_lower for token in bank_filter_list):
                continue

        records.append(
            ReportRecord(
                dataset=dataset,
                ticker=ticker,
                path=pdf_path,
                report_name=report_name,
                report_date=report_date,
            )
        )

    return records
