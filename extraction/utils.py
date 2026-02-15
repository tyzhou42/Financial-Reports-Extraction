# utils.py
"""Utility functions for API calls and filesystem helpers."""
import re
import json
import io
import time
import random
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Iterable

from PIL import Image

try:
    from .config import (
        LLM_MODEL_REGISTRY,
        VLM_MODEL_NAME,
        INPUT_DIRECTORY,
        TXT_DIRECTORY,
    )
    from .llm_client import get_llm_agent
except ImportError:  # Allow running as a script without package context.
    from config import (  # type: ignore[no-redef]
        LLM_MODEL_REGISTRY,
        VLM_MODEL_NAME,
        INPUT_DIRECTORY,
        TXT_DIRECTORY,
    )
    from llm_client import get_llm_agent  # type: ignore[no-redef]


_VLM_MODEL_CACHE: Dict[str, Any] = {}
logger = logging.getLogger(__name__)


def _get_vlm_model(response_mime_type: str) -> Any:
    """Reuse VLM model instances keyed by response MIME type."""
    cached = _VLM_MODEL_CACHE.get(response_mime_type)
    if cached is None:
        import google.generativeai as genai

        cached = genai.GenerativeModel(
            VLM_MODEL_NAME,
            generation_config={"response_mime_type": response_mime_type}
        )
        _VLM_MODEL_CACHE[response_mime_type] = cached
    return cached


def setup_directories(dirs: Iterable[Path]) -> None:
    """Creates necessary directories if they don't exist."""
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    logger.info("Directories are set up")


def clear_directory_contents(dir_path: Path) -> None:
    """Remove all files under dir_path while keeping the directory structure."""
    if not dir_path.exists():
        return
    for entry in sorted(dir_path.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if entry.is_file() or entry.is_symlink():
            entry.unlink()


def _clean_llm_response(response_text: str) -> Optional[Dict[str, Any]]:
    """A helper to parse JSON from an LLM's text response, handling markdown."""
    match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = response_text

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error("Failed to decode JSON from LLM response: %s", e)
        logger.debug("LLM raw response:\n%s", response_text)
        return None


def call_llm_api(prompt: str, model_name: str, provider: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    A centralized function to handle API calls to Large Language Models.
    Includes retry logic with exponential backoff for rate limiting.
    """
    resolved_provider = (provider or LLM_MODEL_REGISTRY.get(model_name) or "gemini")
    logger.info("Calling LLM API with model=%s provider=%s", model_name, resolved_provider)

    try:
        agent = get_llm_agent(resolved_provider, model_name)
    except Exception as exc:
        logger.error("Unable to initialize LLM agent for %s: %s", resolved_provider, exc)
        return None

    max_retries = 3
    backoff_factor = 2  # Initial wait time in seconds

    for attempt in range(max_retries):
        try:
            response_text = agent.generate(prompt)
            return _clean_llm_response(response_text)
        except Exception as e:
            error_text = str(e).lower()
            if any(token in error_text for token in ("rate limit", "resource has been exhausted", "429", "overloaded")):
                if attempt < max_retries - 1:
                    wait_time = (backoff_factor ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        "LLM rate limit hit; retrying in %.2f seconds (attempt %s/%s)",
                        wait_time,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(wait_time)
                else:
                    logger.error("LLM rate limit persisted after %s attempts", max_retries)
                    return None
            else:
                logger.error("Unexpected exception during LLM API call: %s", e)
                return None
    return None


def call_vlm_api(
        image_data: bytes,
        prompt: str,
        response_mime_type: str = "application/json"
) -> Optional[Dict[str, Any]]:
    """
    A centralized function to handle API calls to Vision-Language Models.
    Includes retry logic with exponential backoff for rate limiting.
    """
    logger.info("Calling VLM API for an image")
    max_retries = 3
    backoff_factor = 2  # Initial wait time in seconds

    for attempt in range(max_retries):
        try:
            image = Image.open(io.BytesIO(image_data))
            model = _get_vlm_model(response_mime_type)
            response = model.generate_content([prompt, image])
            return _clean_llm_response(response.text)
        except Exception as e:
            error_text = str(e).lower()
            if "rate limit" in error_text or "resource has been exhausted" in error_text or "429" in error_text:
                if attempt < max_retries - 1:
                    wait_time = (backoff_factor ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        "VLM rate limit hit; retrying in %.2f seconds (attempt %s/%s)",
                        wait_time,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(wait_time)
                else:
                    logger.error("VLM rate limit persisted after %s attempts", max_retries)
                    return None
            else:
                logger.error("Unexpected exception during VLM API call: %s", e)
                return None
    return None


def export_processed_pdfs_to_txt(
        source_dir: Optional[Path] = None,
        target_dir: Optional[Path] = None,
        overwrite: bool = True
) -> None:
    """
    Convert all PDFs in the source directory into plain-text files using direct extraction.

    Args:
        source_dir: Optional override for the PDF directory (default: data/reports).
        target_dir: Optional override for where to save the text outputs.
        overwrite: When False, skip writing txt files that already exist.
    """
    from text_extractor import extract_text_directly

    source_dir_path = Path(source_dir) if source_dir else INPUT_DIRECTORY
    txt_dir = Path(target_dir) if target_dir else TXT_DIRECTORY
    if not source_dir_path.exists():
        logger.warning("Source directory not found: %s", source_dir_path.resolve())
        return

    txt_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(source_dir_path.rglob('*.pdf'))
    if not pdf_paths:
        logger.warning("No PDF files found in %s", source_dir_path.resolve())
        return

    logger.info("Exporting text for %s PDFs from %s", len(pdf_paths), source_dir_path.resolve())

    for pdf_path in pdf_paths:
        txt_path = txt_dir / (pdf_path.stem + '.txt')
        if not overwrite and txt_path.exists():
            logger.info("Skipping %s; %s already exists", pdf_path.name, txt_path.name)
            continue

        try:
            extracted_text = extract_text_directly(pdf_path)
            if not extracted_text:
                logger.info("Skipping %s; no text extracted", pdf_path.name)
                continue

            txt_path.write_text(extracted_text, encoding='utf-8')
            logger.info("Wrote %s", txt_path.name)
        except Exception as exc:
            logger.error("Failed to export %s: %s", pdf_path.name, exc)
