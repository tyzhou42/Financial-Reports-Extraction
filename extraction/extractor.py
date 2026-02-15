"""Compatibility facade for extraction helpers.

Historically this project kept all extraction logic in this module.
The implementation now lives in focused modules and is re-exported here
to avoid breaking existing imports.
"""

from feature_extractor import extract_features_with_regex, fill_missing_features_with_llm
from image_extractor import extract_features_from_images, extract_images_from_pdf
from table_extractor import extract_eps_tables_with_vlm, extract_relevant_tables_as_markdown
from text_extractor import extract_text_directly, extract_text_with_ocr, get_text_from_document

__all__ = [
    "extract_features_from_images",
    "extract_features_with_regex",
    "extract_images_from_pdf",
    "extract_relevant_tables_as_markdown",
    "extract_eps_tables_with_vlm",
    "extract_text_directly",
    "extract_text_with_ocr",
    "fill_missing_features_with_llm",
    "get_text_from_document",
]

