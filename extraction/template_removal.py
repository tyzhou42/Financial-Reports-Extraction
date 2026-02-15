"""Remove boilerplate disclaimer paragraphs from English analyst reports.

This module was originally developed under `baseline/` and is now part of the
production extraction pipeline. It operates on `.txt` exports produced by the
PDF text extractor and writes cleaned versions with disclaimer-like paragraphs
removed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
from collections import Counter
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple, TypeVar

from tqdm.auto import tqdm


logger = logging.getLogger(__name__)


def iter_documents(directory: Path) -> Iterator[Tuple[str, str]]:
    """Yield (document_id, text) pairs for every `.txt` file in a directory."""
    for path in sorted(directory.glob("*.txt")):
        yield path.stem, path.read_text(encoding="utf-8", errors="ignore")


# ---------------------------------------------------------------------------
# Patterns and heuristics tuned for English-language financial disclaimers
# ---------------------------------------------------------------------------

DISCLAIMER_HEADINGS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^\s*important (?:disclosures?|information)\b", re.I), "heading:important_disclosures"),
    (re.compile(r"^\s*important u\.?s\.?(?: regulatory)? disclosures\b", re.I), "heading:us_reg_disclosures"),
    (re.compile(r"^\s*disclaimer(s)?\b", re.I), "heading:disclaimer"),
    (re.compile(r"^\s*analyst certification(s)?\b", re.I), "heading:analyst_certification"),
    (re.compile(r"^\s*this report is intended\b", re.I), "heading:this_report_is_intended"),
    (re.compile(r"^\s*distribution of this report\b", re.I), "heading:distribution"),
    (re.compile(r"^\s*required disclosures\b", re.I), "heading:required_disclosures"),
)

DISCLAIMER_REGEXES: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\bdoes not (?:constitute|represent)\b.{0,80}\b(?:investment|legal)\b.{0,40}\b(?:advice|recommendation)\b",
            re.I,
        ),
        "regex:does_not_constitute",
    ),
    (
        re.compile(r"\bshall not\b.{0,40}\b(?:be )?liable\b", re.I),
        "regex:shall_not_be_liable",
    ),
    (
        re.compile(r"\bmay not be reproduced\b", re.I),
        "regex:may_not_be_reproduced",
    ),
    (
        re.compile(r"\bpast performance\b.{0,40}\bnot\b.{0,20}\bindicative\b", re.I),
        "regex:past_performance",
    ),
    (
        re.compile(r"\bfor institutional investors? only\b", re.I),
        "regex:institutional_only",
    ),
    (
        re.compile(r"\breg(?:ulation)?\.?\s*ac\b", re.I),
        "regex:reg_ac",
    ),
)

DEFAULT_KEY_PHRASES: Tuple[str, ...] = (
    "this report is intended for",
    "this report has been prepared by",
    "this report is provided solely",
    "for informational purposes only",
    "should not be construed as investment advice",
    "does not constitute investment advice",
    "does not constitute an offer",
    "past performance is not indicative of future results",
    "the information contained herein",
    "before making an investment decision",
    "independent financial advice",
    "without the prior written consent",
    "may not be reproduced or redistributed",
    "distribution of this report",
    "analyst certification",
    "important disclosures",
    "conflicts of interest",
    "affiliates may have positions",
    "investment involves risks",
    "subject to change without notice",
    "neither the firm nor any affiliate",
    "non-public information",
    "this communication is confidential",
    "intended solely for the addressee",
    "financial promotion",
    "regulation ac",
    "accurate and complete in all material respects",
    "firm may seek investment banking business",
    "redistribution or reproduction is prohibited",
    "standard & poor's equity research services",
    "standard & poor's financial services",
    "this material is not intended as an offer",
    "obligation to update its opinions",
    "warranties of merchantability",
    "fitness for a particular purpose",
    "no event shall",
    "may receive compensation",
    "solicitation for the purchase or sale",
)

TAIL_KEY_PHRASES: Tuple[str, ...] = (
    "standard & poor's",
    "capital iq",
    "redistribution or reproduction",
    "no event shall",
    "no guarantee of future performance",
    "warranties of merchantability",
    "fitness for a particular purpose",
    "obligation to update",
    "incorporates both actual and estimated variables",
    "monetary authority",
    "financial services authority",
    "solicitation for the purchase",
    "jurisdictions where standard & poor's",
    "professional advice",
    "fiduciary",
    "non-public information",
    "indexes are unmanaged",
    "does not take into account the reinvestment of dividends",
    "some of the stars equities",
    "consulting with a financial advisor",
    "only be made after consulting",
    "currency fluctuations and controls",
    "for residents of australia",
    "prepared for use by retail investors",
    "before making any decision or recommendation",
    "financial services licence number",
    "financial services guide",
)

LEGAL_TERMS: Tuple[str, ...] = (
    "solicitation",
    "warranties",
    "merchantability",
    "liability",
    "fiduciary",
    "jurisdiction",
    "monetary authority",
    "financial services authority",
    "securities commission",
    "australian securities & investments commission",
    "reproduction",
    "redistribution",
    "copyright",
    "permission",
    "compensation",
    "s&p parties",
    "standard & poor's",
    "capital iq",
    "cross-border investment advisory",
    "not intended as an offer",
)

_HASH_TOKEN_RE = re.compile(r"[a-z0-9']+")

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class RemovalConfig:
    """Runtime configuration for disclaimer removal."""

    tail_start_ratio: float = 0.5
    min_score: float = 0.9
    keyword_weight: float = 0.7
    heading_weight: float = 1.1
    uppercase_weight: float = 0.3
    tail_weight: float = 0.35
    final_weight: float = 0.35
    near_final_weight: float = 0.2
    short_penalty: float = 0.35
    short_length_threshold: int = 35
    extra_keywords: Tuple[str, ...] = ()
    enable_hash_matching: bool = True
    hash_bits: int = 64
    hash_band_size: int = 16
    hash_match_threshold: int = 6
    hash_min_cluster_size: int = 3
    hash_min_unique_docs: int = 2
    hash_min_tokens: int = 12
    hash_min_chars: int = 80
    hash_score_boost: float = 2.5
    hash_score_growth: float = 0.4
    hash_min_rule_score: float = 0.1
    hash_workers: int = 10
    hash_cluster_workers: int = 10


@dataclass
class HashStats:
    """Similarity metadata computed from paragraph simhash clustering."""

    fingerprint: int | None
    cluster_size: int
    unique_docs: int
    match_count: int


@dataclass(frozen=True)
class ParagraphHashEntry:
    """Internal representation of a hashed paragraph."""

    doc_id: str
    index: int
    fingerprint: int


@dataclass
class ParagraphEvaluation:
    """Paragraph-level decision data."""

    doc_id: str
    index: int
    raw: str
    normalized: str
    uppercase_ratio: float
    is_tail: bool
    is_last: bool
    score: float
    keyword_hits: int
    triggers: List[str]
    hash_fingerprint: int | None
    hash_cluster_size: int
    hash_unique_docs: int
    hash_match_count: int
    hash_score_bonus: float
    is_disclaimer: bool


_SENTENCE_END_RE = re.compile(r"[.!?]['\"]?$")


def _is_bullet_line(line: str) -> bool:
    stripped = line.lstrip()
    if not stripped:
        return False
    bullet_prefixes = ("-", "•", "·", "*", "➤", "►", "→")
    if stripped.startswith(bullet_prefixes):
        return True
    return bool(re.match(r"^(\(?\d+[\).]|[A-Z]\.|[ivx]+\.)\s", stripped))


def _is_heading_line(line: str) -> bool:
    stripped = line.strip(":- ").strip()
    if not stripped or len(stripped) > 80:
        return False
    letters = [c for c in stripped if c.isalpha()]
    if not letters:
        return False
    ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    return ratio > 0.75


def _merge_buffer(buffer: List[str]) -> str:
    if not buffer:
        return ""
    merged: List[str] = []
    for line in buffer:
        if merged and merged[-1].endswith("-"):
            merged[-1] = merged[-1][:-1] + line.lstrip()
        else:
            merged.append(line)
    joined = " ".join(merged)
    return re.sub(r"\s+", " ", joined).strip()


def _should_flush_paragraph(buffer: List[str], current_line: str, next_line: str | None) -> bool:
    if not next_line:
        return True
    if _SENTENCE_END_RE.search(current_line):
        return True
    if len(" ".join(buffer)) >= 400:
        return True
    if _is_bullet_line(next_line) or _is_heading_line(next_line):
        return True
    return False


def _reflow_block(lines: List[str]) -> List[str]:
    paragraphs: List[str] = []
    buffer: List[str] = []
    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            if buffer:
                paragraphs.append(_merge_buffer(buffer))
                buffer.clear()
            continue
        if _is_bullet_line(line) or _is_heading_line(line):
            if buffer:
                paragraphs.append(_merge_buffer(buffer))
                buffer.clear()
            paragraphs.append(line)
            continue
        buffer.append(line)
        next_line = lines[idx + 1] if idx + 1 < len(lines) else None
        if _should_flush_paragraph(buffer, line, next_line.strip() if next_line else None):
            paragraphs.append(_merge_buffer(buffer))
            buffer.clear()
    if buffer:
        paragraphs.append(_merge_buffer(buffer))
    return paragraphs


def split_into_paragraphs(text: str) -> List[str]:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = [part for part in re.split(r"\n{2,}", cleaned) if part.strip()]
    if not parts and cleaned.strip():
        parts = [cleaned]

    paragraphs: List[str] = []
    for part in parts:
        lines = [line for line in part.split("\n")]
        paragraphs.extend([para for para in _reflow_block(lines) if para])
    return paragraphs


def normalize_paragraph(paragraph: str) -> str:
    text = paragraph.strip().lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\d[\d,\.]*", "<num>", text)
    return text


def uppercase_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    uppercase = sum(1 for c in letters if c.isupper())
    return uppercase / len(letters)


def _parallel_map_with_progress(
    func: Callable[[T], R],
    items: Iterable[T],
    *,
    total: int,
    desc: str,
    unit: str,
    max_workers: int,
    executor_cls: type[Executor] | None = None,
) -> List[R]:
    if total <= 0:
        return []

    items_list = list(items)
    max_workers = max(1, max_workers)
    executor_cls = executor_cls or ThreadPoolExecutor

    if max_workers == 1:
        results: List[R] = []
        with tqdm(total=total, desc=desc, unit=unit, leave=False) as progress:
            for item in items_list:
                results.append(func(item))
                progress.update(1)
        return results

    results_with_index: List[Tuple[int, R]] = []
    with executor_cls(max_workers=max_workers) as executor:
        futures = {executor.submit(func, item): idx for idx, item in enumerate(items_list)}
        with tqdm(total=total, desc=desc, unit=unit, leave=False) as progress:
            for future in as_completed(futures):
                idx = futures[future]
                results_with_index.append((idx, future.result()))
                progress.update(1)
    results_with_index.sort(key=lambda pair: pair[0])
    return [result for _, result in results_with_index]


def _hash_worker(item: Tuple[str, int, Sequence[str]], *, num_bits: int) -> ParagraphHashEntry:
    doc_id, idx, tokens = item
    fingerprint = compute_simhash(tokens, num_bits=num_bits)
    return ParagraphHashEntry(doc_id=doc_id, index=idx, fingerprint=fingerprint)


def _cluster_worker(
    idx: int,
    *,
    entries: Sequence[ParagraphHashEntry],
    band_slices: Sequence[Tuple[int, int, int]],
    band_to_entries: Mapping[Tuple[int, int], Sequence[int]],
    match_threshold: int,
) -> Tuple[str, int, HashStats]:
    entry = entries[idx]
    similar_docs: set[str] = set()
    match_count = 0

    for band, shift, mask in band_slices:
        band_key = (band, (entry.fingerprint >> shift) & mask)
        candidates = band_to_entries.get(band_key, ())
        for candidate_idx in candidates:
            if candidate_idx == idx:
                continue
            candidate = entries[candidate_idx]
            if hamming_distance(entry.fingerprint, candidate.fingerprint) <= match_threshold:
                match_count += 1
                similar_docs.add(candidate.doc_id)

    unique_docs = len(similar_docs | {entry.doc_id})
    stats = HashStats(
        fingerprint=entry.fingerprint,
        cluster_size=match_count + 1,
        unique_docs=unique_docs,
        match_count=match_count,
    )
    return entry.doc_id, entry.index, stats


def _tokenize_for_hashing(text: str) -> List[str]:
    if not text:
        return []
    tokens: List[str] = []
    for token in _HASH_TOKEN_RE.findall(text):
        if token == "num":
            continue
        if any(char.isdigit() for char in token):
            continue
        tokens.append(token)
    return tokens


def compute_simhash(tokens: Sequence[str], *, num_bits: int = 64) -> int:
    if not tokens:
        return 0
    bit_length = max(16, num_bits)
    byte_length = max(1, (bit_length + 7) // 8)
    vector = [0] * bit_length
    token_weights = Counter(tokens)
    for token, weight in token_weights.items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=byte_length).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(bit_length):
            if value & (1 << bit):
                vector[bit] += weight
            else:
                vector[bit] -= weight

    fingerprint = 0
    for bit, balance in enumerate(vector):
        if balance >= 0:
            fingerprint |= 1 << bit
    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def build_phrase_list(config: RemovalConfig) -> Tuple[str, ...]:
    phrases = list(DEFAULT_KEY_PHRASES)
    if config.extra_keywords:
        phrases.extend(config.extra_keywords)
    return tuple(phrases)


def _score_keywords(normalized: str, phrases: Sequence[str]) -> Tuple[int, float, List[str]]:
    hits = 0
    triggers: List[str] = []
    for phrase in phrases:
        if phrase and phrase in normalized:
            hits += 1
            triggers.append(f"keyword:{phrase}")
    score = hits / max(1, len(phrases))
    return hits, score, triggers


def _score_headings(raw: str) -> Tuple[float, List[str]]:
    triggers: List[str] = []
    score = 0.0
    stripped = raw.strip()
    for pattern, label in DISCLAIMER_HEADINGS:
        if pattern.search(stripped):
            score += 1.0
            triggers.append(label)
    for pattern, label in DISCLAIMER_REGEXES:
        if pattern.search(stripped):
            score += 0.6
            triggers.append(label)
    return score, triggers


def evaluate_paragraph(
    *,
    doc_id: str,
    raw: str,
    index: int,
    total: int,
    tail_start: int,
    config: RemovalConfig,
    phrases: Sequence[str],
    hash_stats: HashStats | None,
) -> ParagraphEvaluation:
    normalized = normalize_paragraph(raw)
    upper_ratio = uppercase_ratio(raw)
    is_tail = index >= tail_start
    is_last = index == total - 1

    keyword_hits, keyword_score, keyword_triggers = _score_keywords(normalized, phrases)
    heading_score, heading_triggers = _score_headings(raw)

    score = (
        keyword_score * config.keyword_weight
        + heading_score * config.heading_weight
        + upper_ratio * config.uppercase_weight
    )

    triggers = []
    triggers.extend(keyword_triggers)
    triggers.extend(heading_triggers)

    if is_tail:
        score += config.tail_weight
        for phrase in TAIL_KEY_PHRASES:
            if phrase in normalized:
                score += 0.25
                triggers.append(f"tail:{phrase}")
    if is_last:
        score += config.final_weight
    if total >= 2 and index >= total - 2:
        score += config.near_final_weight

    if len(normalized.split()) <= config.short_length_threshold:
        score -= config.short_penalty
        triggers.append("penalty:short")

    hash_fingerprint = None
    hash_cluster_size = 1
    hash_unique_docs = 1
    hash_match_count = 0
    hash_score_bonus = 0.0
    if hash_stats and hash_stats.fingerprint is not None:
        hash_fingerprint = hash_stats.fingerprint
        hash_cluster_size = hash_stats.cluster_size
        hash_unique_docs = hash_stats.unique_docs
        hash_match_count = hash_stats.match_count
        if hash_cluster_size >= config.hash_min_cluster_size and hash_unique_docs >= config.hash_min_unique_docs:
            growth = max(0, hash_cluster_size - config.hash_min_cluster_size)
            hash_score_bonus = config.hash_score_boost + growth * config.hash_score_growth
            score += hash_score_bonus
            triggers.append("hash:cluster")

    is_disclaimer = score >= config.min_score
    return ParagraphEvaluation(
        doc_id=doc_id,
        index=index,
        raw=raw,
        normalized=normalized,
        uppercase_ratio=upper_ratio,
        is_tail=is_tail,
        is_last=is_last,
        score=score,
        keyword_hits=keyword_hits,
        triggers=triggers,
        hash_fingerprint=hash_fingerprint,
        hash_cluster_size=hash_cluster_size,
        hash_unique_docs=hash_unique_docs,
        hash_match_count=hash_match_count,
        hash_score_bonus=hash_score_bonus,
        is_disclaimer=is_disclaimer,
    )


def build_hash_statistics(
    paragraphs_by_doc: Mapping[str, Sequence[str]],
    config: RemovalConfig,
    rule_scores_by_doc: Mapping[str, Sequence[float]],
) -> Dict[str, Dict[int, HashStats]]:
    if not paragraphs_by_doc:
        return {}

    doc_stats: Dict[str, Dict[int, HashStats]] = {doc_id: {} for doc_id in paragraphs_by_doc}
    if not config.enable_hash_matching:
        for doc_id, paragraphs in paragraphs_by_doc.items():
            for idx in range(len(paragraphs)):
                doc_stats[doc_id][idx] = HashStats(
                    fingerprint=None,
                    cluster_size=1,
                    unique_docs=1,
                    match_count=0,
                )
        return doc_stats

    bit_length = max(16, config.hash_bits)
    band_size = max(1, config.hash_band_size)
    match_threshold = max(0, config.hash_match_threshold)
    min_tokens = max(1, config.hash_min_tokens)
    min_chars = max(0, config.hash_min_chars)

    entries: List[ParagraphHashEntry] = []
    hash_candidates: List[Tuple[str, int, Sequence[str]]] = []
    total_paragraphs = sum(len(paragraphs) for paragraphs in paragraphs_by_doc.values())
    scan_progress = (
        tqdm(total=total_paragraphs, desc="Scanning paragraphs", unit="para", leave=False)
        if total_paragraphs
        else None
    )

    try:
        for doc_id, paragraphs in paragraphs_by_doc.items():
            doc_map = doc_stats.setdefault(doc_id, {})
            doc_rule_scores = rule_scores_by_doc.get(doc_id)
            for idx, raw in enumerate(paragraphs):
                doc_map[idx] = HashStats(fingerprint=None, cluster_size=1, unique_docs=1, match_count=0)
                base_score = None
                if doc_rule_scores and idx < len(doc_rule_scores):
                    base_score = doc_rule_scores[idx]
                if base_score is not None and base_score >= config.hash_min_rule_score:
                    normalized = normalize_paragraph(raw)
                    tokens = _tokenize_for_hashing(normalized)
                    if len(tokens) >= min_tokens and len(normalized) >= min_chars:
                        hash_candidates.append((doc_id, idx, tokens))
                if scan_progress:
                    scan_progress.update(1)
    finally:
        if scan_progress:
            scan_progress.close()

    if hash_candidates:
        hash_func = partial(_hash_worker, num_bits=bit_length)
        hashed_entries = _parallel_map_with_progress(
            hash_func,
            hash_candidates,
            total=len(hash_candidates),
            desc="Hashing paragraphs",
            unit="para",
            max_workers=config.hash_workers,
            executor_cls=ProcessPoolExecutor,
        )
        entries.extend(hashed_entries)

    if not entries:
        return doc_stats

    num_bands = max(1, (bit_length + band_size - 1) // band_size)
    band_slices: List[Tuple[int, int, int]] = []
    for band in range(num_bands):
        shift = band * band_size
        if shift >= bit_length:
            continue
        bits_in_band = min(band_size, bit_length - shift)
        mask = (1 << bits_in_band) - 1
        band_slices.append((band, shift, mask))

    band_to_entries: Dict[Tuple[int, int], List[int]] = {}
    for idx, entry in enumerate(entries):
        for band, shift, mask in band_slices:
            band_key = (band, (entry.fingerprint >> shift) & mask)
            band_to_entries.setdefault(band_key, []).append(idx)

    cluster_func = partial(
        _cluster_worker,
        entries=entries,
        band_slices=band_slices,
        band_to_entries=band_to_entries,
        match_threshold=match_threshold,
    )
    cluster_results = _parallel_map_with_progress(
        cluster_func,
        range(len(entries)),
        total=len(entries),
        desc="Clustering hashes",
        unit="para",
        max_workers=config.hash_cluster_workers,
        executor_cls=ThreadPoolExecutor,
    )

    for doc_id, index, stats in cluster_results:
        if doc_id in doc_stats:
            doc_stats[doc_id][index] = stats
    return doc_stats


def clean_document(
    doc_id: str,
    text: str,
    config: RemovalConfig,
    phrases: Sequence[str],
    *,
    paragraphs: Sequence[str] | None = None,
    hash_stats: Mapping[int, HashStats] | None = None,
) -> Tuple[str, str, List[dict[str, object]]]:
    paragraph_list = list(paragraphs) if paragraphs is not None else split_into_paragraphs(text)
    total = len(paragraph_list)
    if total == 0:
        return "", "", []
    tail_start = min(int(total * config.tail_start_ratio), max(total - 1, 0))

    evaluations: List[ParagraphEvaluation] = []
    for index, paragraph in enumerate(paragraph_list):
        evaluation = evaluate_paragraph(
            doc_id=doc_id,
            raw=paragraph,
            index=index,
            total=total,
            tail_start=tail_start,
            config=config,
            phrases=phrases,
            hash_stats=hash_stats.get(index) if hash_stats else None,
        )
        evaluations.append(evaluation)

    cleaned: List[str] = []
    marked: List[str] = []
    for evaluation, paragraph in zip(evaluations, paragraph_list):
        if evaluation.is_disclaimer:
            marked.append(f"[REMOVED] {paragraph}")
        else:
            marked.append(paragraph)
            cleaned.append(paragraph)

    cleaned_text = "\n\n".join(cleaned).strip()
    marked_text = "\n\n".join(marked).strip()

    records = [
        {
            "doc_id": evaluation.doc_id,
            "paragraph_index": evaluation.index,
            "is_tail": evaluation.is_tail,
            "is_last": evaluation.is_last,
            "char_length": len(evaluation.raw),
            "uppercase_ratio": round(evaluation.uppercase_ratio, 3),
            "keyword_hits": evaluation.keyword_hits,
            "score": round(evaluation.score, 3),
            "is_disclaimer": evaluation.is_disclaimer,
            "triggers": "|".join(evaluation.triggers),
            "hash_cluster_size": evaluation.hash_cluster_size,
            "hash_unique_docs": evaluation.hash_unique_docs,
            "hash_match_count": evaluation.hash_match_count,
            "hash_score_bonus": round(evaluation.hash_score_bonus, 3),
            "hash_fingerprint": f"{evaluation.hash_fingerprint:016x}"
            if evaluation.hash_fingerprint is not None
            else "",
            "text_preview": evaluation.raw[:120].replace("\n", " "),
        }
        for evaluation in evaluations
    ]
    return cleaned_text, marked_text, records


def write_report(report_path: Path, records: Sequence[dict[str, object]]) -> None:
    if not records:
        return
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def process_corpus(
    corpus_dir: Path,
    output_dir: Path,
    marked_dir: Path | None,
    report_csv: Path | None,
    config: RemovalConfig,
) -> dict[str, object]:
    phrases = build_phrase_list(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    if marked_dir:
        marked_dir.mkdir(parents=True, exist_ok=True)

    documents: List[Tuple[str, str, List[str]]] = []
    for doc_id, text in iter_documents(corpus_dir):
        paragraph_list = split_into_paragraphs(text)
        documents.append((doc_id, text, paragraph_list))

    paragraphs_by_doc: Dict[str, Sequence[str]] = {doc_id: paragraphs for doc_id, _, paragraphs in documents}
    rule_scores_by_doc: Dict[str, List[float]] = {}

    if config.enable_hash_matching and documents:
        rule_score_config = replace(config, enable_hash_matching=False)
        for doc_id, _, paragraphs in documents:
            total = len(paragraphs)
            if total == 0:
                rule_scores_by_doc[doc_id] = []
                continue
            tail_start = min(int(total * config.tail_start_ratio), max(total - 1, 0))
            doc_scores: List[float] = []
            for index, paragraph in enumerate(paragraphs):
                evaluation = evaluate_paragraph(
                    doc_id=doc_id,
                    raw=paragraph,
                    index=index,
                    total=total,
                    tail_start=tail_start,
                    config=rule_score_config,
                    phrases=phrases,
                    hash_stats=None,
                )
                doc_scores.append(evaluation.score)
            rule_scores_by_doc[doc_id] = doc_scores

    hash_stats_by_doc = (
        build_hash_statistics(paragraphs_by_doc, config, rule_scores_by_doc) if documents else {}
    )

    all_records: List[dict[str, object]] = []
    document_count = 0
    total_paragraphs = 0
    removed_paragraphs = 0

    for doc_id, text, paragraphs in documents:
        doc_hash_stats = hash_stats_by_doc.get(doc_id) if hash_stats_by_doc else None
        cleaned_text, marked_text, records = clean_document(
            doc_id,
            text,
            config,
            phrases,
            paragraphs=paragraphs,
            hash_stats=doc_hash_stats,
        )
        document_count += 1
        total_paragraphs += len(records)
        removed_paragraphs += sum(1 for record in records if record["is_disclaimer"])

        output_path = output_dir / f"{doc_id}.txt"
        output_path.write_text(cleaned_text, encoding="utf-8")

        if marked_dir:
            marked_path = marked_dir / f"{doc_id}.txt"
            marked_path.write_text(marked_text, encoding="utf-8")

        for record in records:
            record["output_file"] = str(output_path)
        all_records.extend(records)

    if report_csv:
        write_report(report_csv, all_records)

    summary = {
        "n_documents": document_count,
        "n_total_paragraphs": total_paragraphs,
        "n_removed_paragraphs": removed_paragraphs,
        "removal_ratio": (removed_paragraphs / total_paragraphs) if total_paragraphs else 0.0,
        "output_dir": str(output_dir),
        "marked_dir": str(marked_dir) if marked_dir else None,
        "report_csv": str(report_csv) if report_csv else None,
        "threshold": config.min_score,
        "tail_start_ratio": config.tail_start_ratio,
        "extra_keywords": list(config.extra_keywords),
        "hash_enabled": config.enable_hash_matching,
        "hash_min_cluster_size": config.hash_min_cluster_size if config.enable_hash_matching else None,
        "hash_min_unique_docs": config.hash_min_unique_docs if config.enable_hash_matching else None,
        "hash_match_threshold": config.hash_match_threshold if config.enable_hash_matching else None,
        "hash_min_rule_score": config.hash_min_rule_score if config.enable_hash_matching else None,
        "hash_workers": config.hash_workers if config.enable_hash_matching else None,
        "hash_cluster_workers": config.hash_cluster_workers if config.enable_hash_matching else None,
    }
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove disclaimer-style paragraphs from English analyst reports."
    )
    parser.add_argument("--corpus-dir", type=Path, required=True, help="Directory containing input .txt files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for cleaned output files.")
    parser.add_argument("--marked-dir", type=Path, help="Optional directory for marked copies.")
    parser.add_argument("--report", type=Path, help="Optional CSV file for paragraph decisions.")
    parser.add_argument(
        "--extra-keyword",
        action="append",
        default=[],
        help="Additional lowercase keyword or phrase to flag; can be supplied multiple times.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Score threshold for classifying a paragraph as a disclaimer.",
    )
    parser.add_argument(
        "--tail-ratio",
        type=float,
        default=0.55,
        help="Ratio of the document (0-1) considered the trailing section.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    tail_ratio = max(0.0, min(0.95, args.tail_ratio))
    extra_keywords = tuple(
        keyword.strip().lower()
        for keyword in args.extra_keyword
        if isinstance(keyword, str) and keyword.strip()
    )

    config = RemovalConfig(
        tail_start_ratio=tail_ratio,
        min_score=args.threshold,
        extra_keywords=extra_keywords,
    )

    summary = process_corpus(
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        marked_dir=args.marked_dir,
        report_csv=args.report,
        config=config,
    )
    logger.info("Template removal summary:\n%s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

