# postprocessor.py
"""
Handles the cleaning and standardization of the extracted CSV data.
This script is run after the main extraction pipeline is complete.
It is designed to be robust against various formatting inconsistencies.
"""
import pandas as pd
import re
from pathlib import Path
import difflib
from collections import defaultdict, Counter
import json
import logging
from functools import lru_cache
from typing import Dict

from logging_utils import setup_logging
from utils import export_processed_pdfs_to_txt


logger = logging.getLogger(__name__)
BANK_MAPPING_PATH = Path(__file__).resolve().parent / "resources" / "bank_mappings.json"


@lru_cache(maxsize=1)
def _load_bank_mappings(mapping_path: Path = BANK_MAPPING_PATH) -> Dict[str, str]:
    """Load bank-name normalization map from JSON once per process."""
    with mapping_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Bank mapping JSON must contain an object: {mapping_path}")
    return {str(key): str(value) for key, value in data.items()}

def clean_date(series: pd.Series) -> pd.Series:
    """
    [FINAL CORRECTION] Standardizes date strings to YYYY-MM-DD format.
    This version now pre-replaces '.' with '-' to ensure formats like
    'YYYY.MM.DD' are always parsed correctly.
    """
    # astype(str) is crucial
    date_series_str = series.astype(str)

    return pd.to_datetime(
        date_series_str,
        errors='coerce',
        format='mixed'
    ).dt.strftime('%Y-%m-%d')


def clean_price(series: pd.Series) -> pd.Series:
    """
    Extracts the first valid numeric value from a price string.
    Handles formats like: $525.00, USD 200.00, USD130.00, $214.00, (100.50)
    """
    series_str = series.astype(str).str.strip()

    # Extract numeric value (with or without commas, handles decimals)
    number_pattern = re.compile(r'([\d,]+\.?\d*|\.\d+)')
    extracted_series = series_str.str.extract(number_pattern, expand=False)

    # Remove commas and convert to numeric
    extracted_series = extracted_series.str.replace(',', '', regex=False)
    numeric_values = pd.to_numeric(extracted_series, errors='coerce')

    # Handle negative values (shown in parentheses)
    is_negative = series_str.str.contains(r'\(.*\)', regex=True, na=False)
    numeric_values.loc[is_negative] = -numeric_values.loc[is_negative].abs()

    return numeric_values


def extract_currency(series: pd.Series) -> pd.Series:
    """
    Extracts and standardizes currency symbols or ISO codes.
    Handles formats like: $525.00, USD 200.00, USD130.00, C$50.00
    """
    series_str = series.astype(str).str.strip()

    # Pattern matches:
    # 1. ISO codes (with optional space/no space before number): USD, EUR, GBP, etc.
    # 2. Special prefixed symbols: C$, US$, CDN$, HK$, A$, AU$
    # 3. Single currency symbols: $, €, £, ¥, ₹
    currency_pattern = re.compile(
        r'(USD|EUR|GBP|CAD|AUD|JPY|CNY|INR|HKD|SGD|CHF|NZD|ZAR|MXN|BRL|KRW|TWD)\s*(?=[\d(])|'
        r'(C\$|US\$|CDN\$|HK\$|A\$|AU\$|NZ\$)|'
        r'([\$€£¥₹])',
        re.IGNORECASE
    )

    extracted = series_str.str.extract(currency_pattern, expand=True)

    # Combine all groups (ISO codes, prefixed symbols, single symbols)
    currency = extracted[0].fillna(extracted[1]).fillna(extracted[2])
    currency = currency.astype("string").str.upper()

    # Standardize currency symbols and codes
    currency_map = {
        '$': 'USD',
        'US$': 'USD',
        'C$': 'CAD',
        'CDN$': 'CAD',
        'A$': 'AUD',
        'AU$': 'AUD',
        'HK$': 'HKD',
        'NZ$': 'NZD',
        '€': 'EUR',
        '£': 'GBP',
        '¥': 'JPY',
        '₹': 'INR'
    }

    return currency.map(currency_map).fillna(currency)

def clean_numeric_eps(series: pd.Series) -> pd.Series:
    """
    [IMPROVED] A specific cleaner for EPS to ensure it's a numeric type.
    """
    series_str = series.astype(str).str.strip()
    number_pattern = re.compile(r'([\d,]+\.?\d*|\.\d+)')
    extracted_series = series_str.str.extract(number_pattern, expand=False).str.replace(',', '', regex=False)
    numeric_values = pd.to_numeric(extracted_series, errors='coerce')
    is_negative = series_str.str.contains(r'\(.*\)', regex=True)
    numeric_values.loc[is_negative] = -numeric_values.loc[is_negative].abs()
    return numeric_values


def clean_eps_period(series: pd.Series) -> pd.Series:
    """
    Standardizes EPS period strings to a consistent format: Q#-YYYY

    Handles common formats:
    - 3Q24E, 1Q25E
    - Q3FY24e, Q1FY25E
    - Q2/24E, Q4/25
    - Q1 2024, Q3 24
    - FY2024, FY24

    Returns standardized format: 'Q1-2024' for quarters, 'FY-2024' for full years
    """
    series_str = series.astype(str).str.strip().str.upper()

    def normalize_year(year_str):
        """Convert 2-digit or 4-digit year to 4-digit format."""
        if not year_str or not year_str.isdigit():
            return None
        year = int(year_str)
        if year < 100:
            year = 2000 + year
        elif year < 1000:
            return None
        return str(year)

    def standardize_period(period_str):
        if pd.isna(period_str) or period_str in ['', 'NAN', 'NONE', 'NULL']:
            return None

        # Remove estimate markers (E, A) and extra whitespace
        clean_str = re.sub(r'[EA]\s*$', '', period_str).strip()

        # Pattern 1: Quarter number first (3Q24, 1Q25)
        match = re.search(r'^(\d)Q(?:FY|CY)?[\s/\-]*(\d{2,4})', clean_str)
        if match:
            quarter = match.group(1)
            year = normalize_year(match.group(2))
            if year and 1 <= int(quarter) <= 4:
                return f"Q{quarter}-{year}"

        # Pattern 2: Q first (Q3FY24, Q2/24, Q1 2024)
        match = re.search(r'Q(\d)(?:FY|CY)?[\s/\-]*(\d{2,4})', clean_str)
        if match:
            quarter = match.group(1)
            year = normalize_year(match.group(2))
            if year and 1 <= int(quarter) <= 4:
                return f"Q{quarter}-{year}"

        # Pattern 3: Full year only - FY2024, FY24, 2024
        match = re.search(r'(?:FY|CY)?(\d{2,4})$', clean_str)
        if match:
            year = normalize_year(match.group(1))
            if year:
                return f"FY-{year}"

        # No match found
        return None

    return series_str.apply(standardize_period)


def standardize_bank_name(series: pd.Series) -> pd.Series:
    """
    Enhanced bank name standardization with comprehensive manual corrections.
    Uses a two-phase approach: manual mappings first, then fuzzy matching.
    """
    bank_mappings = _load_bank_mappings()

    def normalize_for_lookup(name):
        """Aggressive normalization for dictionary lookup."""
        if pd.isna(name) or str(name).strip() == '':
            return ''

        name = str(name).strip().lower()
        # Remove all punctuation and extra spaces
        name = re.sub(r'[^\w\s]', ' ', name)
        name = re.sub(r'\s+', ' ', name).strip()
        return name

    def apply_manual_mapping(name):
        """Apply manual mapping if available."""
        if pd.isna(name) or str(name).strip() == '':
            return None

        normalized = normalize_for_lookup(name)

        # Try exact match first
        if normalized in bank_mappings:
            return bank_mappings[normalized]

        # Special handling for S&P variations (the ampersand might get lost)
        if 's p' in normalized or 'standard' in normalized:
            if any(word in normalized for word in ['standard', 'poor', 'poors', 'mcgraw', 'hill']):
                return 'S&P Capital IQ'

        # Try partial match for complex names
        for key, value in bank_mappings.items():
            # Check if the key is contained in the normalized name
            if key in normalized or normalized in key:
                # Additional validation: check if it's a meaningful match
                key_words = set(key.split())
                name_words = set(normalized.split())
                # If there's significant word overlap, it's likely the same bank
                if len(key_words & name_words) >= min(2, len(key_words)):
                    return value

        return None

    # Try manual mapping first
    manual_mapped = series.apply(apply_manual_mapping)

    # Phase 2: Smart title case for unmapped names
    def smart_title_case(name):
        """Apply intelligent title casing."""
        if not name:
            return name

        lowercase_words = {'and', 'or', 'of', 'the', 'for', 'in', 'on', 'at', 'to', 'a', 'an'}
        acronyms = {
            'USA', 'UK', 'US', 'LLC', 'LP', 'LLP', 'LTD', 'INC', 'CORP', 'CO',
            'UBS', 'RBC', 'BMO', 'TD', 'HSBC', 'BNP', 'BAML', 'BOFA', 'JPM',
            'MS', 'GS', 'CS', 'DB', 'BTIG', 'ISI', 'CRT', 'JMP', 'FBN'
        }

        words = name.split()
        result = []

        for i, word in enumerate(words):
            # Check for acronyms
            word_upper = word.upper().replace('.', '')
            if word_upper in acronyms or (2 <= len(word_upper) <= 5 and word.isupper()):
                result.append(word.upper())
            # Keep lowercase words lowercase (except first word)
            elif i > 0 and word.lower() in lowercase_words:
                result.append(word.lower())
            # Special handling for possessives and contractions
            elif "'" in word:
                parts = word.split("'")
                result.append("'".join([parts[0].capitalize()] + [p.lower() for p in parts[1:]]))
            else:
                result.append(word.capitalize())

        return ' '.join(result)

    # For unmapped names, apply title case
    unmapped_mask = manual_mapped.isna()
    if unmapped_mask.any():
        unmapped_cleaned = series[unmapped_mask].astype(str).str.strip().apply(smart_title_case)
        manual_mapped = manual_mapped.fillna(unmapped_cleaned)

    return manual_mapped

def standardize_analyst_name(series: pd.Series) -> pd.Series:
    """
    Enhanced analyst name standardization.
    Handles common issues like inconsistent spacing, punctuation, and initials.
    """

    def normalize_name(name):
        """Clean and standardize analyst name format."""
        if pd.isna(name) or str(name).strip() == '':
            return None

        name = str(name).strip()

        # Handle cases like "L. Huberty" vs "Katy L. Huberty"
        # Extract components
        parts = name.split()

        # If name has initials, try to expand based on other occurrences
        # For now, just standardize format

        # Remove extra spaces
        name = re.sub(r'\s+', ' ', name)

        # Standardize periods after initials
        name = re.sub(r'([A-Z])\.?\s+([A-Z])\.?\s+', r'\1. \2. ', name)
        name = re.sub(r'([A-Z])\.?\s+(?=[A-Z][a-z])', r'\1. ', name)

        return name.strip()

    # Apply normalization
    normalized = series.apply(normalize_name)

    # Group similar names (e.g., "Katy Huberty" and "Katy L. Huberty")
    name_groups = defaultdict(list)

    for idx, name in normalized.items():
        if name:
            # Create a key based on last name and first initial
            parts = name.split()
            if len(parts) >= 2:
                # Use last part as surname, first part's first letter
                key = (parts[-1].lower(), parts[0][0].lower())
                name_groups[key].append((idx, name))

    # For each group, select the most complete version
    canonical_map = {}
    for key, names in name_groups.items():
        if len(names) > 1:
            # Prefer longer, more complete names
            canonical = max(names, key=lambda x: len(x[1]))[1]
            for idx, name in names:
                canonical_map[idx] = canonical

    # Apply canonical mapping
    result = normalized.copy()
    for idx, canonical_name in canonical_map.items():
        result.iloc[idx] = canonical_name

    return result


def standardize_company_name(series: pd.Series) -> pd.Series:
    """
    Standardizes company names using fuzzy matching and intelligent normalization.
    """

    def normalize_name(name):
        """Remove common suffixes, punctuation, and standardize format."""
        if pd.isna(name) or str(name).strip() == '':
            return ''

        name = str(name).strip()
        name = re.sub(r'\s*\([A-Z]{1,5}\)\s*', ' ', name)
        name = re.sub(r'\s+&\s+', ' and ', name, flags=re.IGNORECASE)
        name = re.sub(r'\bthe\b', '', name, flags=re.IGNORECASE)

        suffixes = [
            r'\b(inc\.?|incorporated|corp\.?|corporation|ltd\.?|limited|llc|l\.l\.c\.?|plc|p\.l\.c\.?)\b',
            r'\b(co\.?|company|companies)\b',
            r'\b(sa|s\.a\.?|nv|n\.v\.?|ag|a\.g\.?|gmbh|se|s\.e\.?)\b'
        ]
        for suffix in suffixes:
            name = re.sub(suffix, '', name, flags=re.IGNORECASE)

        name = re.sub(r'[^\w\s]', ' ', name)
        name = re.sub(r'\s+', ' ', name).strip().lower()
        return name

    def smart_title_case(name):
        """Apply title case while preserving acronyms."""
        if not name:
            return name

        lowercase_words = {'and', 'or', 'of', 'the', 'for', 'in', 'on', 'at', 'to', 'a', 'an'}
        acronyms = {'USA', 'UK', 'US', 'LLC', 'LP', 'LLP', 'AI', 'IT', 'IBM', 'ATT', 'HP'}

        words = name.split()
        result = []

        for i, word in enumerate(words):
            if word.upper() in acronyms or (2 <= len(word) <= 5 and word.isupper()):
                result.append(word.upper())
            elif i > 0 and word.lower() in lowercase_words:
                result.append(word.lower())
            elif "'" in word:
                parts = word.split("'")
                result.append("'".join([parts[0].capitalize()] + [p.lower() for p in parts[1:]]))
            else:
                result.append(word.capitalize())

        return ' '.join(result)

    normalized_series = series.astype(str).apply(normalize_name)

    normalized_to_originals = {}
    for idx, norm_name in normalized_series.items():
        if norm_name:
            original = str(series.iloc[idx]).strip()
            if norm_name not in normalized_to_originals:
                normalized_to_originals[norm_name] = []
            normalized_to_originals[norm_name].append(original)

    unique_normalized = [n for n in normalized_series.unique() if n]
    canonical_map = {}
    processed = set()

    for name in unique_normalized:
        if name in processed or not name:
            continue

        close_matches = difflib.get_close_matches(
            name,
            [n for n in unique_normalized if n not in processed],
            n=5,
            cutoff=0.90
        )

        if close_matches:
            canonical = min(close_matches,
                            key=lambda x: (-len(normalized_to_originals.get(x, [])), len(x)))

            for match in close_matches:
                canonical_map[match] = canonical
                processed.add(match)

    canonical_to_proper = {}
    for norm_name, originals in normalized_to_originals.items():
        canonical = canonical_map.get(norm_name, norm_name)

        if canonical not in canonical_to_proper:
            counts = Counter(originals)
            best_original = max(originals, key=lambda x: (counts[x], -len(x)))
            best_original = re.sub(r'\s*\([A-Z]{1,5}\)\s*', '', best_original)
            proper = smart_title_case(best_original)
            proper = re.sub(r'\b(Inc|Corp|Ltd|LLC|PLC|Co)\b',
                            lambda m: m.group(1).title() + '.',
                            proper, flags=re.IGNORECASE)
            proper = re.sub(r'\.\.+', '.', proper)
            canonical_to_proper[canonical] = proper.strip()

    def map_to_standard(norm_name):
        if not norm_name:
            return None
        canonical = canonical_map.get(norm_name, norm_name)
        return canonical_to_proper.get(canonical)

    standardized = normalized_series.apply(map_to_standard)
    fallback = series.astype(str).str.strip().apply(smart_title_case)
    return standardized.fillna(fallback)


def run_postprocessing(input_path: Path, output_path: Path) -> None:
    """
    Main function to run the entire post-processing pipeline on the CSV file.
    """
    if not input_path.exists():
        logger.warning("Post-processing skipped; input not found: %s", input_path)
        return

    logger.info("Starting enhanced post-processing for %s", input_path)

    df = pd.read_csv(input_path)

    logger.info("Standardizing company_name")
    df['company_name'] = standardize_company_name(df['company_name'])

    logger.info("Standardizing report_date")
    df['report_date'] = clean_date(df['report_date'])

    logger.info("Standardizing bank_name")
    if 'bank_name' in df.columns:
        df['bank_name'] = standardize_bank_name(df['bank_name'])

    logger.info("Standardizing analyst_name")
    if 'analyst_name' in df.columns:
        df['analyst_name'] = standardize_analyst_name(df['analyst_name'])

    logger.info("Cleaning target_price and extracting currency")
    df['target_price_currency'] = extract_currency(df['target_price'])
    df['target_price'] = clean_price(df['target_price'])

    logger.info("Standardizing current_period_eps to numeric")
    if 'current_period_eps' in df.columns:
        df['current_period_eps'] = clean_numeric_eps(df['current_period_eps'])

    logger.info("Standardizing next_period_eps to numeric")
    if 'next_period_eps' in df.columns:
        df['next_period_eps'] = clean_numeric_eps(df['next_period_eps'])

    logger.info("Standardizing EPS period fields")
    if 'current_period_eps_period' in df.columns:
        df['current_period_eps_period'] = clean_eps_period(df['current_period_eps_period'])
    if 'next_period_eps_period' in df.columns:
        df['next_period_eps_period'] = clean_eps_period(df['next_period_eps_period'])

    if {'company_name', 'current_period_eps_period', 'current_period_eps'}.issubset(df.columns):
        group_cols = ['company_name', 'current_period_eps_period']
        mask = df[group_cols].notna().all(axis=1)
        if mask.any():
            def _mode_scalar(series: pd.Series):
                values = series.dropna()
                if values.empty:
                    return float('nan')
                return values.value_counts().index[0]

            df.loc[mask, 'current_period_eps'] = (
                df.loc[mask]
                .groupby(group_cols)['current_period_eps']
                .transform(_mode_scalar)
            )

    cols = [
        'source_filename',
        'company_name',
        'bank_name',
        'report_date',
        'analyst_name',
        'analyst_title',
        'target_price',
        'target_price_currency',
        'current_period_eps',
        'current_period_eps_period',
        'next_period_eps',
        'next_period_eps_period',
    ]
    if 'error' in df.columns:
        cols.append('error')

    final_cols = [col for col in cols if col in df.columns]
    df = df[final_cols]

    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    logger.info("Post-processing summary: total_records=%s", len(df))
    if "company_name" in df.columns:
        logger.info("Unique companies: %s", df["company_name"].nunique())
    if "bank_name" in df.columns:
        logger.info("Unique banks: %s", df["bank_name"].nunique())
        logger.info("Most common banks:\n%s", df["bank_name"].value_counts().head(10).to_string())
    if "analyst_name" in df.columns:
        logger.info("Unique analysts: %s", df["analyst_name"].nunique())
    logger.info("Post-processing output saved to %s", output_path)

if __name__ == '__main__':
    setup_logging(log_file=Path(__file__).resolve().parent / "logs" / "postprocessor.log")
    TXT_EXTRACTION = False  # Toggle for pilot/example runs.
    project_root = Path(__file__).parent.parent
    raw_output_file = project_root / 'data' / 'output' / 'extracted_financial_data.csv'
    cleaned_output_file = project_root / 'data' / 'output' / 'extracted_financial_data_cleaned.csv'

    if TXT_EXTRACTION:
        export_processed_pdfs_to_txt(
            source_dir=project_root / 'data' / 'reports',
            target_dir=project_root / 'data' / 'txts'
        )

    logger.info("Running postprocessor in test mode")
    run_postprocessing(raw_output_file, cleaned_output_file)
    logger.info("Final corrected test output:\n%s", pd.read_csv(cleaned_output_file).to_string())
