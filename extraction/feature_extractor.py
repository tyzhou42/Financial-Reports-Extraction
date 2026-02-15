"""Feature extraction helpers using regex and LLM passes."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from config import (
    CHEAP_LLM_MODEL_NAME,
    LLM_FEATURE_EXTRACTION_PROMPT_CHEAP,
    LLM_FEATURE_EXTRACTION_PROMPT_STRONG,
    STRONG_LLM_MODEL_NAME,
    TARGET_FEATURES,
    get_feature_rules_subset,
)
from utils import call_llm_api


logger = logging.getLogger(__name__)


def extract_features_with_regex(text: str) -> Dict[str, Optional[Any]]:
    """Extract a first-pass set of features with regex heuristics."""
    logger.info("Attempting regex feature extraction")
    extracted_data: Dict[str, Optional[Any]] = {key: None for key in TARGET_FEATURES}

    date_pattern = re.compile(
        r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})\b",
        re.IGNORECASE,
    )
    date_match = date_pattern.search(text)
    if date_match:
        extracted_data["report_date"] = date_match.group(0).strip()

    analyst_pattern = re.compile(
        r"(?:(?:Research\s)?Analyst|Prepared by|Author)[:\s]*\n?([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,2})(?:,?\s*(.*?))\n",
        re.IGNORECASE,
    )
    analyst_match = analyst_pattern.search(text)
    if analyst_match:
        extracted_data["analyst_name"] = analyst_match.group(1).strip()
        if analyst_match.group(2) and len(analyst_match.group(2)) < 50:
            extracted_data["analyst_title"] = analyst_match.group(2).strip()

    price_pattern = re.compile(
        r"(?:Target\sPrice|Price\sTarget|PT)\s*[:\-]?\s*([$€£]?\s?\d{1,4}(?:\.\d{1,2})?)",
        re.IGNORECASE,
    )
    price_match = price_pattern.search(text)
    if price_match:
        extracted_data["target_price"] = price_match.group(1).strip()

    eps_pattern = re.compile(
        r"(?:FY|CY|20)\d{2}(?:E|A)?\s+EPS\s*[:\-]?\s*([$€£]?\s?\d{1,3}(?:\.\d{1,2})?)",
        re.IGNORECASE,
    )
    eps_match = eps_pattern.search(text)
    if eps_match:
        extracted_data["next_period_eps"] = eps_match.group(1).strip()

    if not extracted_data["company_name"]:
        for line in text.split("\n")[:15]:
            line = line.strip()
            if re.match(r"^[A-Z][A-Z\s.&-]{5,40}[A-Z.]$", line):
                if not any(keyword in line.lower() for keyword in ["research", "report", "equity", "analysis"]):
                    extracted_data["company_name"] = line
                    break

    found_count = sum(1 for value in extracted_data.values() if value is not None)
    logger.info("Regex found %s/%s features", found_count, len(TARGET_FEATURES))
    for key, value in extracted_data.items():
        if value:
            logger.debug("Regex feature %s=%s", key, value)
    return extracted_data


def fill_missing_features_with_llm(
    text: str,
    table_markdown: str,
    existing_features: Dict[str, Optional[Any]],
) -> Dict[str, Optional[Any]]:
    """Fill missing fields with a cheap pass then a strong EPS-focused pass."""
    missing_features = [key for key, value in existing_features.items() if value is None]

    if not missing_features:
        logger.info("No missing features; skipping LLM calls")
        return existing_features

    strong_only_fields = {
        "current_period_eps",
        "current_period_eps_period",
        "next_period_eps",
        "next_period_eps_period",
    }

    cheap_targets = [key for key in missing_features if key not in strong_only_fields]
    if cheap_targets:
        logger.info("Attempting cheap LLM extraction for %s fields", len(cheap_targets))
        cheap_rules = get_feature_rules_subset(cheap_targets)
        cheap_prompt = LLM_FEATURE_EXTRACTION_PROMPT_CHEAP.format(
            features_to_find=", ".join(cheap_targets),
            text_chunk=text[:16000],
            table_markdown="",
            feature_rules=cheap_rules,
        )
        cheap_results = call_llm_api(cheap_prompt, CHEAP_LLM_MODEL_NAME)

        updates = 0
        if cheap_results:
            for key in cheap_targets:
                value = cheap_results.get(key)
                if value:
                    existing_features[key] = value
                    updates += 1
        logger.info("Cheap LLM populated %s fields", updates)

    strong_targets = [
        key
        for key in (
            "current_period_eps",
            "current_period_eps_period",
            "next_period_eps",
            "next_period_eps_period",
        )
        if existing_features.get(key) is None
    ]
    if strong_targets:
        logger.info("Attempting strong LLM extraction for fields: %s", ", ".join(strong_targets))
        eps_rules = get_feature_rules_subset(strong_targets)
        strong_prompt = LLM_FEATURE_EXTRACTION_PROMPT_STRONG.format(
            features_to_find=", ".join(strong_targets),
            text_chunk=text[:16000],
            table_markdown=table_markdown,
            feature_rules=eps_rules,
        )
        strong_results = call_llm_api(strong_prompt, STRONG_LLM_MODEL_NAME)

        updates = 0
        if strong_results:
            for key in strong_targets:
                value = strong_results.get(key)
                if value:
                    existing_features[key] = value
                    updates += 1
        logger.info("Strong LLM populated %s fields", updates)

    return existing_features

