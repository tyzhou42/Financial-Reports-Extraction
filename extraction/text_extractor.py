"""Text extraction helpers for PDF documents."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def extract_text_directly(file_path: Path) -> Optional[str]:
    """Extract text directly from a text-based PDF."""
    logger.info("Direct text extraction: %s", file_path.name)
    try:
        import pypdf

        text = ""
        with open(file_path, "rb") as handle:
            reader = pypdf.PdfReader(handle)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"

        if text.strip() and len(text.strip()) > 100:
            return text

        logger.info("Direct extraction yielded little text for %s", file_path.name)
        return None
    except Exception as exc:
        logger.warning("Direct extraction failed for %s: %s", file_path.name, exc)
        return None


def extract_text_with_ocr(file_path: Path) -> Optional[str]:
    """Fallback OCR-based text extraction."""
    logger.info("OCR text extraction: %s", file_path.name)
    full_text = ""
    try:
        import pytesseract
        from pdf2image import convert_from_path

        images = convert_from_path(file_path)
        logger.info("OCR rendered %s pages for %s", len(images), file_path.name)

        for index, image in enumerate(images):
            logger.debug("OCR page %s/%s for %s", index + 1, len(images), file_path.name)
            page_text = pytesseract.image_to_string(image, lang="eng")
            if page_text:
                full_text += page_text + "\n\n--- Page Break ---\n\n"

        if full_text.strip():
            return full_text

        logger.info("OCR produced no text for %s", file_path.name)
        return None
    except Exception as exc:
        logger.warning("OCR extraction failed for %s: %s", file_path.name, exc)
        return None


def get_text_from_document(file_path: Path) -> Optional[str]:
    """Extract text using direct extraction first, then OCR fallback."""
    text = extract_text_directly(file_path)
    if text:
        logger.info("Direct extraction succeeded for %s", file_path.name)
        return text

    logger.info("Falling back to OCR for %s", file_path.name)
    text = extract_text_with_ocr(file_path)
    if text:
        logger.info("OCR extraction succeeded for %s", file_path.name)
        return text

    logger.warning("All text extraction methods failed for %s", file_path.name)
    return None
