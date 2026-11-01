#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_report.py — Bidirectional converter for DETECTION.md reports.

Transforms an existing DETECTION.md file between:
  - 'oneline' : Compact single-line per image format.
  - 'expand'  : Full multi-line section + Markdown table per image format.

No model inference or network request is needed.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import scan

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("convert-report")

# Regex patterns for parsing expanded and oneline entries
LANG_CODE_HEADER_REGEX = re.compile(r"\(`([a-z]{2})`\)\s*$", re.MULTILINE)
TABLE_ROW_REGEX = re.compile(
    r"^\|\s*#\d+\s*\|\s*\*\*([^*]+)\*\*\s*\|\s*([^|]+?)\s*\|\s*`([^`]+)`\s*\|\s*([\d.]+)%\s*\|$",
    re.MULTILINE,
)
EXPANDED_ERROR_REGEX = re.compile(r"^-\s*⚠️\s*\*\*[^:]+:\*\*\s*`([^`]+)`", re.MULTILINE)

ONELINE_PRED_REGEX = re.compile(
    r"^(?P<rel_path>[^\n#-][^\n]*?)\s+:\s+"
    r"\*\*\[(?P<sci>[^\]]+)\]\*\*\s+\(\*(?P<common>[^*]+)\*\)\s+"
    r"\(\*\*[^:]+:\*\*\s*`(?P<family>[^`]+)`\)\s+—\s+"
    r"\*\*[^:]+:\s*(?P<conf>[\d.]+)%\*\*"
    r"(?:\s+/\s+\*\*[^:]+:\*\*\s+(?P<alts>.+))?$"
)
ONELINE_ALT_ITEM_REGEX = re.compile(
    r"\*(?P<sci>[^*]+)\*\s+\((?P<common>.+?),\s*(?P<family>[^,—]+?)\s*—\s*(?P<conf>[\d.]+)%\)"
)
ONELINE_WARN_REGEX = re.compile(
    r"^(?P<rel_path>[^\n#-][^\n]*?)\s+:\s+⚠️\s+(?:\*\*[^:]+:\*\*\s*`(?P<error>[^`]+)`|(?P<low>.+))$"
)


def detect_report_language(content: str, override_lang: Optional[str] = None) -> str:
    """Detect the language code ('en', 'fr', 'id', 'es', 'de') of an existing DETECTION.md."""
    if override_lang:
        return override_lang.strip().lower()

    match = LANG_CODE_HEADER_REGEX.search(content)
    if match:
        return match.group(1).lower()

    # Fallback heuristic for reports generated before the language header field existed
    if "Famille taxonomique" in content or "Nom vernaculaire" in content or "Dossier analysé" in content:
        return "fr"
    if "Famili taksonomi" in content or "Folder yang dianalisis" in content:
        return "id"
    if "Familia taxonómica" in content or "Carpeta analizada" in content:
        return "es"
    if "Taxonomische Familie" in content or "Analysierter Ordner" in content:
        return "de"
    return scan.DEFAULT_LANGUAGE


def split_header_and_body(content: str) -> Tuple[str, str]:
    """Split `DETECTION.md` into `(header_with_separator, body)`."""
    separator = "\n---\n\n"
    if separator in content:
        header_part, body = content.split(separator, 1)
        return header_part + separator, body
    return "", content


def _parse_expanded_section(section_text: str) -> Optional[scan.ImagePredictionResult]:
    """Parse a single expanded (`### 📷 ...`) Markdown section into `ImagePredictionResult`."""
    header_match = scan.ENTRY_HEADER_REGEX.search(section_text)
    if not header_match:
        return None

    rel_path = header_match.group(1).strip()

    err_match = EXPANDED_ERROR_REGEX.search(section_text)
    if err_match:
        return scan.ImagePredictionResult(rel_path=rel_path, error=err_match.group(1).strip())

    hypotheses: List[scan.SpeciesHypothesis] = []
    for sci, common, family, conf_str in TABLE_ROW_REGEX.findall(section_text):
        hypotheses.append(
            scan.SpeciesHypothesis(
                scientific_name=sci.strip(),
                common_name=common.strip(),
                family=family.strip(),
                confidence=float(conf_str),
            )
        )

    return scan.ImagePredictionResult(rel_path=rel_path, hypotheses=hypotheses)


def _parse_oneline_line(line: str) -> Optional[scan.ImagePredictionResult]:
    """Parse a single compact `--oneline` line into `ImagePredictionResult`."""
    stripped = line.strip()
    if not stripped:
        return None

    pred_match = ONELINE_PRED_REGEX.match(stripped)
    if pred_match:
        rel_path = pred_match.group("rel_path").strip()
        hypotheses = [
            scan.SpeciesHypothesis(
                scientific_name=pred_match.group("sci").strip(),
                common_name=pred_match.group("common").strip(),
                family=pred_match.group("family").strip(),
                confidence=float(pred_match.group("conf")),
            )
        ]
        alts_str = pred_match.group("alts")
        if alts_str:
            for alt_m in ONELINE_ALT_ITEM_REGEX.finditer(alts_str):
                hypotheses.append(
                    scan.SpeciesHypothesis(
                        scientific_name=alt_m.group("sci").strip(),
                        common_name=alt_m.group("common").strip(),
                        family=alt_m.group("family").strip(),
                        confidence=float(alt_m.group("conf")),
                    )
                )
        return scan.ImagePredictionResult(rel_path=rel_path, hypotheses=hypotheses)

    warn_match = ONELINE_WARN_REGEX.match(stripped)
    if warn_match:
        rel_path = warn_match.group("rel_path").strip()
        error_msg = warn_match.group("error")
        if error_msg:
            return scan.ImagePredictionResult(rel_path=rel_path, error=error_msg.strip())
        return scan.ImagePredictionResult(rel_path=rel_path, hypotheses=[])

    return None


def parse_detection_md_results(body: str) -> List[scan.ImagePredictionResult]:
    """
    Parse all image entries from the body of a `DETECTION.md` file,
    supporting both expanded multi-line sections and `--oneline` lines.
    """
    results: List[scan.ImagePredictionResult] = []

    if "### 📷 `" in body:
        # Split on `### 📷 ` while keeping the header marker
        chunks = re.split(r"(?=^###\s+📷\s+`)", body, flags=re.MULTILINE)
        for chunk in chunks:
            if not chunk.strip():
                continue
            parsed = _parse_expanded_section(chunk)
            if parsed is not None:
                results.append(parsed)
    else:
        for line in body.splitlines():
            parsed = _parse_oneline_line(line)
            if parsed is not None:
                results.append(parsed)

    return results


def convert_markdown_content(
    content: str,
    target_mode: str,
    lang: Optional[str] = None,
    min_confidence: float = scan.DEFAULT_MIN_CONFIDENCE,
) -> Tuple[str, int]:
    """
    Convert the full text of a `DETECTION.md` file to `target_mode` ('oneline' or 'expand').
    Returns `(converted_markdown, num_entries_converted)`.
    """
    if target_mode not in ("oneline", "expand"):
        raise ValueError(f"Invalid target_mode: {target_mode}")

    resolved_lang = detect_report_language(content, override_lang=lang)
    header, body = split_header_and_body(content)
    results = parse_detection_md_results(body)

    oneline = target_mode == "oneline"
    formatted_entries = [
        scan.format_prediction_entry(
            res,
            lang=resolved_lang,
            min_confidence=min_confidence,
            oneline=oneline,
        )
        for res in results
    ]

    return header + "".join(formatted_entries), len(results)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an existing DETECTION.md report between expanded (multi-line + table) "
            "and compact (--oneline) formats without re-running model inference."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--folder",
        "-f",
        type=Path,
        help="Path to the folder containing DETECTION.md.",
    )
    target_group.add_argument(
        "--file",
        "-i",
        type=Path,
        help="Direct path to a DETECTION.md file.",
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--oneline",
        action="store_true",
        help="Transform DETECTION.md into the compact single-line format.",
    )
    mode_group.add_argument(
        "--expand",
        action="store_true",
        help="Transform DETECTION.md into the detailed multi-line + Markdown table format.",
    )
    mode_group.add_argument(
        "--to",
        choices=["oneline", "expand"],
        help="Target format ('oneline' or 'expand').",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional output file path (defaults to updating DETECTION.md in-place).",
    )
    parser.add_argument(
        "--lang",
        "-l",
        type=str,
        default=None,
        help="Override report language code (auto-detected from DETECTION.md by default).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=scan.DEFAULT_MIN_CONFIDENCE,
        help="Minimum confidence percentage (0-100) required to keep a hypothesis.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    if args.file:
        md_path = args.file.expanduser().resolve()
    else:
        md_path = args.folder.expanduser().resolve() / "DETECTION.md"

    if not md_path.exists() or not md_path.is_file():
        logger.error("DETECTION.md file not found: %s", md_path)
        sys.exit(1)

    target_mode = "oneline" if args.oneline else ("expand" if args.expand else args.to)
    output_path = args.output.expanduser().resolve() if args.output else md_path

    content = md_path.read_text(encoding="utf-8")
    converted_text, count = convert_markdown_content(
        content=content,
        target_mode=target_mode,
        lang=args.lang,
        min_confidence=args.min_confidence,
    )

    output_path.write_text(converted_text, encoding="utf-8")
    logger.info(
        "Converted %d image entries to '%s' format in: %s",
        count,
        target_mode,
        output_path,
    )


if __name__ == "__main__":
    main()
