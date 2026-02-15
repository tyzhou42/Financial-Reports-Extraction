"""Configuration constants for the extraction pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

from dotenv import load_dotenv


# Load environment variables when this module is imported.
load_dotenv()


PROVIDER_API_KEY_ENV: Dict[str, str] = {
    "gemini": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
}

_DEFAULT_MODELS: Dict[str, Dict[str, str]] = {
    "gemini": {"cheap": "gemini-3-flash-preview", "strong": "gemini-3-pro-preview"},
    "openai": {"cheap": "gpt-4o-mini", "strong": "gpt-4o"},
}


def _normalize_provider(provider: Optional[str], fallback: str) -> str:
    candidate = (provider or fallback).strip().lower()
    return candidate if candidate in _DEFAULT_MODELS else fallback


def _resolve_model(env_var: str, provider: str, tier: str) -> str:
    explicit = os.environ.get(env_var)
    if explicit:
        return explicit
    return _DEFAULT_MODELS[provider][tier]


CHEAP_LLM_PROVIDER = _normalize_provider(
    os.environ.get("EXTRACTION_CHEAP_LLM_PROVIDER"), "gemini"
)
STRONG_LLM_PROVIDER = _normalize_provider(
    os.environ.get("EXTRACTION_STRONG_LLM_PROVIDER"), "gemini"
)
VLM_PROVIDER = _normalize_provider(os.environ.get("EXTRACTION_VLM_PROVIDER"), "gemini")

CHEAP_LLM_MODEL_NAME = _resolve_model(
    "EXTRACTION_CHEAP_LLM_MODEL", CHEAP_LLM_PROVIDER, "cheap"
)
STRONG_LLM_MODEL_NAME = _resolve_model(
    "EXTRACTION_STRONG_LLM_MODEL", STRONG_LLM_PROVIDER, "strong"
)
VLM_MODEL_NAME = _resolve_model("EXTRACTION_VLM_MODEL", VLM_PROVIDER, "strong")

ACTIVE_LLM_PROVIDERS: Set[str] = {
    CHEAP_LLM_PROVIDER,
    STRONG_LLM_PROVIDER,
}

LLM_MODEL_REGISTRY: Dict[str, str] = {
    CHEAP_LLM_MODEL_NAME: CHEAP_LLM_PROVIDER,
    STRONG_LLM_MODEL_NAME: STRONG_LLM_PROVIDER,
}

for provider, models in _DEFAULT_MODELS.items():
    for model in models.values():
        LLM_MODEL_REGISTRY.setdefault(model, provider)


def _env_path(var_name: str, default: str) -> Path:
    value = os.environ.get(var_name)
    return Path(value) if value else Path(default)


INPUT_DIRECTORY = _env_path("EXTRACTION_INPUT_DIRECTORY", "../data/reports")
OUTPUT_DIRECTORY = _env_path("EXTRACTION_OUTPUT_DIRECTORY", "../data/output")
ANALYST_REPORTS_DIRECTORY = _env_path(
    "EXTRACTION_ANALYST_REPORTS_DIRECTORY", "../datasets/analyst_reports"
)
PROCESSED_DIRECTORY = _env_path("EXTRACTION_PROCESSED_DIRECTORY", "../data/processed")
ERROR_DIRECTORY = _env_path("EXTRACTION_ERROR_DIRECTORY", "../data/error")
TXT_DIRECTORY = _env_path("EXTRACTION_TXT_DIRECTORY", "../data/txts")
TXT_NO_DISCLAIMER_DIRECTORY = _env_path(
    "EXTRACTION_TXT_NO_DISCLAIMER_DIRECTORY", "../data/txts_no_disclaimer"
)
LOG_DIRECTORY = _env_path("EXTRACTION_LOG_DIRECTORY", "logs")


TARGET_FEATURES = [
    "company_name",
    "report_date",
    "bank_name",
    "analyst_name",
    "analyst_title",
    "target_price",
    "current_period_eps",
    "current_period_eps_period",
    "next_period_eps",
    "next_period_eps_period",
]

FEATURE_EXTRACTION_RULES = {
    "ticker_name": "Extract the exact stock ticker symbol from the report. Look for the ticker on the title page, coverage header, or company overview section. The ticker is typically shown in parentheses after the company name, in all caps, or near the price/rating information. Return only the ticker symbol itself without any exchange prefix, suffix, or additional formatting (e.g., 'AAPL' not 'NASDAQ:AAPL' or 'AAPL US'). If multiple tickers are present (e.g., for dual-listed companies), extract the primary ticker shown most prominently. Preserve the exact case as shown in the document. If no explicit ticker is found in the report, infer the most commonly known ticker symbol based on the company name (e.g., 'Apple Inc.' → 'AAPL', 'Microsoft Corporation' → 'MSFT', 'Tesla, Inc.' → 'TSLA'). Only return null if neither an explicit ticker nor a company name suitable for inference is available.",
    "report_date": "Extract the publication date of this report (not cited sources or prior versions). Prefer the title page masthead or the first-page banner. Normalize to 'YYYY-MM-DD'. Handle formats like '25 Sep 2025', 'Sep 25, 2025', '25/09/2025', '09/25/2025'. If multiple dates appear, choose the one clearly marking publication; otherwise the most prominent date on the title page. If only month/year is present, return null.",
    "bank_name": "Extract the common brand name of the issuing financial institution. The goal is a short, recognizable name for easy post-processing and matching. 1.  **Simplify:** Remove legal suffixes (e.g., 'LLC', 'Inc.', '& Co.') and departmental names (e.g., 'Research', 'Securities'). 2.  **Standardize:** Convert well-known abbreviations to their full brand name (e.g., 'BofA' to 'Bank of America'). **Examples to follow:** - Input: 'Goldman Sachs & Co. LLC' -> Output: 'Goldman Sachs' - Input: 'Morgan Stanley Research' -> Output: 'Morgan Stanley' - Input: 'J.P. Morgan Securities LLC' -> Output: 'J.P. Morgan' - Input: 'BofA Global Research' -> Output: 'Bank of America' - Input: 'UBS Investment Bank' -> Output: 'UBS'",
    "analyst_name": "Extract the first leading analyst’s full name. Look for labels such as 'Analyst', 'Prepared by', 'Lead/Primary Analyst', or the byline near rating. If multiple analysts are shown, choose the first lead/primary (often marked with an asterisk or with contact info). Exclude associates/assistants unless no lead is present. Do not include title of the analyst.",
    "analyst_title": "Extract the job title of the selected analyst, typically adjacent to their name (e.g., 'Senior Equity Analyst', 'Research Analyst', 'Managing Director', 'Head of Research'). If multiple credentials/titles appear, return the core role title; omit certifications unless they are the only title.",
    "target_price": "Extract the current explicit price target/price objective for the stock with currency symbol (e.g., $150.50, €25.00, £12.30, C$45.00, HK$88). Accept labels 'Price Target', 'Target Price', 'Price Objective', or 'PT'. If multiple values appear, prefer the first-page rating/overview banner or the value tied to the current rating action. If the number lacks a currency symbol, return null.",
    "current_period_eps": "Extract the analyst's specific Earnings Per Share (EPS) for the current fiscal quarter (the quarter that is ongoing as of report_date, or the most recent quarter just before report_date if the current quarter is not explicitly labeled). Follow these rules precisely:\n\n## Search Priority:\n1.  **Analyst's Model/Estimates Table:** The primary source is the detailed financial model table.\n2.  **Summary 'Estimates' Panel:** If not in the main model, look for a summary forecast box.\n3.  **Forecast Exhibits:** As a last resort, check other charts or tables containing forecasts.\n\n## Extraction Criteria:\n* **Value Type:** Prioritize 'Adjusted' or 'Non-GAAP' EPS. If not available or specified, use the primary/reported EPS figure.\n* **Source:** Must be the **analyst's own estimate or actual**. Do not use 'Consensus', 'Street', or 'Mean' estimates. If only consensus values are found, return null.\n* **Time Period:** Identify the **current fiscal quarter** associated with the report date. If the table separates Actual (A) and Estimate (E), prefer the quarter marked as actual on or immediately before report_date. If only estimates are available, use the estimate for the quarter containing report_date.\n* **Quarter Formats:** Recognize various formats like 'Q1 FY2026', '2Q26', 'Mar-26 (quarter)', 'CQ3 2025', etc.\n* **Exclusion Rule:** If only **annual** EPS values are available for the current period, return null. The target must be a **quarterly** value.\n\n## Output Format:\n* **On Success:** Return a numeric value formatted as a string (e.g., \"0.85\", \"-0.12\").\n* **On Failure:** Return null if no quarterly analyst EPS value can be found according to the criteria.",
    "current_period_eps_period": "Return the fiscal period label associated with `current_period_eps`. Normalize whitespace and output the fiscal quarter in the standardized format 'Q#-YYYY' (e.g., 'Q3-2025'). Remove fiscal prefixes (FY/CY/CQ) and remove markers like 'E'/'A' and month-based formats (convert 'Mar-26E' → 'Q1-2026'). If multiple quarter labels appear, choose the one explicitly tied to the analyst EPS value. If no quarterly value is available, return null.",
    "next_period_eps": "Extract the analyst's specific Earnings Per Share (EPS) estimate for the single next fiscal quarter on or after the report_date. Follow these rules precisely:\n\n## Search Priority:\n1.  **Analyst's Model/Estimates Table:** The primary source is the detailed financial model table.\n2.  **Summary 'Estimates' Panel:** If not in the main model, look for a summary forecast box.\n3.  **Forecast Exhibits:** As a last resort, check other charts or tables containing forecasts.\n\n## Extraction Criteria:\n* **Value Type:** Prioritize 'Adjusted' or 'Non-GAAP' EPS. If not available or specified, use the primary/reported EPS figure.\n* **Source:** Must be the **analyst's own estimate**. Do not use 'Consensus', 'Street', or 'Mean' estimates. If only consensus values are found, return null.\n* **Time Period:** Identify the **first** upcoming quarter explicitly marked as an estimate (often with an 'E'). This column typically follows the last quarter of 'Actual' data (often marked with an 'A').\n* **Quarter Formats:** Recognize various formats like 'Q1 FY2026', '2Q26', 'Mar-26 (quarter)', 'CQ3 2025', etc.\n* **Exclusion Rule:** If only **annual** EPS estimates are available for the next period, return null. The target must be a **quarterly** value.\n\n## Output Format:\n* **On Success:** Return a numeric value formatted as a string (e.g., \"0.85\", \"-0.12\").\n* **On Failure:** Return null if no quarterly analyst estimate can be found according to the criteria.",
    "next_period_eps_period": "Return the fiscal period label associated with `next_period_eps`. Normalize whitespace and output the fiscal quarter in the standardized format 'Q#-YYYY' (e.g., 'Q3-2025'). Remove fiscal prefixes (FY/CY/CQ) and remove estimate markers like 'E' and month-based formats (convert 'Mar-26E' → 'Q1-2026'). If multiple quarter labels appear, choose the one explicitly tied to the analyst estimate. If no quarterly estimate is available, return null.",
}

TABLE_KEYWORDS = {
    "current_period_eps": ["EPS", "Earnings Per Share"],
    "next_period_eps": ["EPS", "Earnings Per Share"],
}

VLM_TEXT_TRANSCRIPTION_PROMPT = """
Transcribe the text from this image of a financial report page.
Be as accurate as possible. Do not summarize or interpret the content.
Return the result as a JSON object with a single key "text".
"""

LLM_FEATURE_EXTRACTION_PROMPT_CHEAP = """
You are an expert financial data extraction system.
From the following text of a financial report, extract ONLY the following fields: {features_to_find}.

You have been provided with two sources of information:
1.  **Full Text Context**: The raw text from the financial report.
2.  **Relevant Extracted Tables**: Tables that have been identified as potentially containing key data, provided in Markdown format.

**Instructions**:
- Use both sources to find the most accurate information.
- **Prioritize data from the 'Relevant Extracted Tables'** for numerical values like prices or EPS, as they are structured.
- Use the 'Full Text Context' to find information not in the tables (like analyst names) and to understand the context of the table data.
- Adhere strictly to the Feature Extraction Rules provided below.
- Return a single, valid JSON object. Use null for any fields you cannot find.

--- TEXT (first 16000 characters) ---
{text_chunk}

--- RELEVANT EXTRACTED TABLES ---
{table_markdown}

Feature Extraction Rules:
{feature_rules}
"""

LLM_FEATURE_EXTRACTION_PROMPT_STRONG = """
You are a world-class financial analyst specializing in data extraction.
From the following text (in Markdown format), perform a high-accuracy extraction for these specific fields: {features_to_find}.

You have been provided with two sources of information:
1.  **Full Text Context**: The raw text from the financial report.
2.  **Relevant Extracted Tables**: Tables that have been identified as potentially containing key data, provided in Markdown format.

**Instructions**:
- Use both sources to find the most accurate information.
- **Prioritize data from the 'Relevant Extracted Tables'** for numerical values like prices or EPS, as they are structured.
- Use the 'Full Text Context' to find information not in the tables (like analyst names) and to understand the context of the table data.
- Adhere strictly to the Feature Extraction Rules provided below.
- Return a single, valid JSON object. Use null for any fields you cannot find.

--- TEXT (first 16000 characters) ---
{text_chunk}

--- RELEVANT EXTRACTED TABLES ---
{table_markdown}

Feature Extraction Rules:
{feature_rules}
"""

VLM_FEATURE_EXTRACTION_PROMPT = """
Analyze this image from a financial report.
Extract any of the following fields if they are clearly visible: {target_features}.
Focus on data presented in tables, charts, or headers.
Return the result as a single, valid JSON object. Use null for fields not present.

Feature Extraction Rules:
{feature_rules}
"""

VLM_EPS_TABLE_PROMPT = """
You are transcribing an EPS table from a financial report page image (page {page_number}).
Return a JSON object with a single key "table_markdown" containing a clean Markdown table focused on analyst EPS estimates.

Guidelines:
- Preserve column headers and quarter labels exactly as they appear.
- Include only rows or columns relevant to EPS estimates; omit pricing, revenue, or unrelated metrics.
- Normalize whitespace inside table cells, but keep currency symbols and minus signs.
- If no EPS content is present, return {{"table_markdown": null}}.

Feature Extraction Rules:
{feature_rules}
"""


def get_feature_rules_subset(feature_names: Iterable[str]) -> str:
    """Return formatted extraction rules for selected features."""
    lines = []
    for name in feature_names:
        rule = FEATURE_EXTRACTION_RULES.get(name)
        if rule:
            lines.append(f"- {name}: {rule}")
    return "\n".join(lines)

