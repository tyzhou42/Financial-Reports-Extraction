"""Image extraction and VLM feature extraction helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz

from config import TARGET_FEATURES, VLM_FEATURE_EXTRACTION_PROMPT, get_feature_rules_subset
from utils import call_vlm_api


logger = logging.getLogger(__name__)


def extract_images_from_pdf(file_path: Path) -> List[bytes]:
    """Extract all embedded images from a PDF."""
    logger.info("Extracting images from %s", file_path.name)
    images: List[bytes] = []
    try:
        document = fitz.open(file_path)
        for page_number in range(len(document)):
            page = document.load_page(page_number)
            for image in page.get_images(full=True):
                xref = image[0]
                base_image = document.extract_image(xref)
                image_bytes = base_image["image"]
                images.append(image_bytes)
        document.close()
    except Exception as exc:
        logger.warning("Image extraction failed for %s: %s", file_path.name, exc)

    logger.info("Found %s images in %s", len(images), file_path.name)
    return images


def extract_features_from_images(
    images: List[bytes],
    features_to_find: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run VLM extraction on each image and merge first non-null values."""
    if not images:
        return {}

    logger.info("Analyzing %s images with VLM", len(images))
    merged_image_features: Dict[str, Any] = {}

    targets = features_to_find or TARGET_FEATURES
    rules = get_feature_rules_subset(targets)
    prompt = VLM_FEATURE_EXTRACTION_PROMPT.format(
        target_features=", ".join(targets),
        feature_rules=rules,
    )

    for index, image_data in enumerate(images):
        logger.debug("Analyzing image %s/%s", index + 1, len(images))
        vlm_result = call_vlm_api(image_data, prompt)

        if vlm_result:
            for key, value in vlm_result.items():
                if key in targets and value and key not in merged_image_features:
                    merged_image_features[key] = value
                    logger.debug("VLM found %s in image %s", key, index + 1)

    logger.info("VLM found %s unique features across images", len(merged_image_features))
    return merged_image_features

