#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan.py — Recursive underwater photo species classifier (North Sulawesi: Lembeh & Bunaken).
Optimized for macOS Apple Silicon (M1/M2/M3/M4 via PyTorch MPS backend).

Generates a DETECTION.md report at the root of the target folder.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import logging
import os
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageOps, UnidentifiedImageError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("underwater-scan")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_LOCAL_MODEL = "hf-hub:timm/eva02_large_patch14_clip_336.merged2b_ft_inat21"


# ==============================================================================
# Data Structures
# ==============================================================================


@dataclass
class SpeciesHypothesis:
    scientific_name: str
    common_name: str
    family: str
    confidence: float  # Percentage between 0.0 and 100.0


@dataclass
class ImagePredictionResult:
    rel_path: str
    hypotheses: List[SpeciesHypothesis] = field(default_factory=list)
    error: Optional[str] = None


# ==============================================================================
# File Discovery & Image Preprocessing
# ==============================================================================


def discover_images(root_folder: Path) -> List[Path]:
    """
    Recursively walk `root_folder` and return a sorted list of valid image paths.
    Ignores hidden directories and macOS resource fork files (`._*`, `.DS_Store`).
    """
    images: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        dirnames[:] = [d for d in sorted(dirnames) if not d.startswith(".")]

        for fname in sorted(filenames):
            if fname.startswith(".") or fname.startswith("._"):
                continue
            ext = Path(fname).suffix.lower()
            if ext in SUPPORTED_EXTENSIONS:
                images.append(Path(dirpath) / fname)

    return images


def load_and_resize_image(image_path: Path, max_size: int = 1024) -> Image.Image:
    """
    Load an image into memory, apply EXIF orientation, and resize proportionally
    so neither dimension exceeds `max_size` to preserve unified RAM on Apple Silicon.
    Raises an exception if the image file is corrupted.
    """
    with Image.open(image_path) as img:
        img.verify()

    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return img.copy()


def image_to_jpeg_bytes(img: Image.Image, quality: int = 88) -> bytes:
    """Encode a PIL Image into compressed JPEG bytes."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


# ==============================================================================
# Taxonomic Enrichment (Public iNaturalist API with In-Memory Cache)
# ==============================================================================


class INaturalistTaxonomyResolver:
    """
    Queries the free public iNaturalist API (no API key required) to resolve
    the exact taxonomic family and vernacular (common) name from a scientific name.
    Uses an in-memory cache to avoid redundant network requests.
    """

    def __init__(self, enabled: bool = True, locale: str = "en") -> None:
        self.enabled = enabled
        self.locale = locale
        self._cache: Dict[Tuple[str, str], Tuple[str, str]] = {}

    def resolve(self, scientific_name: str, fallback_desc: str = "") -> Tuple[str, str]:
        """Return `(common_name, family)` for a given `scientific_name`."""
        cache_key = (scientific_name, self.locale)
        if cache_key in self._cache:
            return self._cache[cache_key]

        fallback_common = fallback_desc.split(",")[0].strip() if fallback_desc else "Unknown"
        fallback_family = (
            fallback_desc.split(",")[1].strip()
            if (fallback_desc and "," in fallback_desc)
            else "Unknown"
        )

        if not self.enabled:
            return fallback_common, fallback_family

        try:
            query = urllib.parse.urlencode(
                {"q": scientific_name, "per_page": 1, "locale": self.locale}
            )
            url = f"https://api.inaturalist.org/v1/taxa?{query}"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "FishVitDetection/1.0 (macOS Apple Silicon)"},
            )
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            results = data.get("results", [])
            if not results:
                self._cache[cache_key] = (fallback_common, fallback_family)
                return fallback_common, fallback_family

            taxon = results[0]
            common_name = (
                taxon.get("preferred_common_name")
                or taxon.get("english_common_name")
                or fallback_common
            )

            family = fallback_family
            for anc in taxon.get("ancestors", []):
                if anc.get("rank") == "family":
                    family = anc.get("name", fallback_family)
                    break

            if family == fallback_family and taxon.get("id"):
                taxon_id = taxon["id"]
                detail_url = (
                    f"https://api.inaturalist.org/v1/taxa/{taxon_id}"
                    f"?locale={urllib.parse.quote(self.locale)}"
                )
                detail_req = urllib.request.Request(
                    detail_url,
                    headers={"User-Agent": "FishVitDetection/1.0 (macOS Apple Silicon)"},
                )
                with urllib.request.urlopen(detail_req, timeout=3.5) as detail_resp:
                    detail_data = json.loads(detail_resp.read().decode("utf-8"))
                detail_results = detail_data.get("results", [])
                if detail_results:
                    t_full = detail_results[0]
                    common_name = (
                        t_full.get("preferred_common_name")
                        or t_full.get("english_common_name")
                        or common_name
                    )
                    for anc in t_full.get("ancestors", []):
                        if anc.get("rank") == "family":
                            family = anc.get("name", fallback_family)
                            break

            self._cache[cache_key] = (common_name, family)
            return common_name, family

        except Exception:
            self._cache[cache_key] = (fallback_common, fallback_family)
            return fallback_common, fallback_family


# ==============================================================================
# Inference Engine (Local PyTorch MPS)
# ==============================================================================


class BaseClassifier:
    model_label: str = "Unknown"

    def predict(self, img: Image.Image, top_k: int = 5) -> List[SpeciesHypothesis]:
        raise NotImplementedError


class LocalTimmINatClassifier(BaseClassifier):
    """
    Local vision classifier using PyTorch + `timm` with Apple Silicon (`mps`)
    hardware acceleration and a checkpoint fine-tuned on iNaturalist 2021.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_LOCAL_MODEL,
        enrich_taxonomy: bool = True,
        locale: str = "en",
    ) -> None:
        try:
            import timm
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Libraries `torch` and `timm` are required for local mode.\n"
                "Install them with: pip install torch torchvision timm pillow"
            ) from exc

        self.torch = torch
        self.model_name = model_name

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Apple Silicon hardware acceleration enabled (backend: MPS).")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("CUDA hardware acceleration enabled.")
        else:
            self.device = torch.device("cpu")
            logger.warning("MPS not available; falling back to CPU.")

        self.model_label = f"Local PyTorch ({model_name} on `{self.device.type}`)"
        logger.info("Loading local model %s...", model_name)

        self.model = timm.create_model(model_name, pretrained=True)
        self.model.eval()
        self.model.to(self.device)

        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

        pretrained_cfg = getattr(self.model, "pretrained_cfg", {}) or {}
        self.label_names: List[str] = pretrained_cfg.get("label_names", [])
        self.label_descriptions: Dict[str, str] = pretrained_cfg.get("label_descriptions", {})

        self.taxonomy_resolver = INaturalistTaxonomyResolver(
            enabled=enrich_taxonomy,
            locale=locale,
        )

    def predict(self, img: Image.Image, top_k: int = 5) -> List[SpeciesHypothesis]:
        tensor = self.transforms(img).unsqueeze(0).to(self.device)

        with self.torch.inference_mode():
            logits = self.model(tensor)
            probs = self.torch.nn.functional.softmax(logits, dim=-1)[0]
            top_probs, top_indices = self.torch.topk(probs, k=top_k)

        hypotheses: List[SpeciesHypothesis] = []
        for idx, prob in zip(top_indices.cpu().tolist(), top_probs.cpu().tolist()):
            sci_name = (
                self.label_names[idx]
                if idx < len(self.label_names)
                else f"Taxon #{idx}"
            )
            raw_desc = self.label_descriptions.get(sci_name, "")
            common_name, family = self.taxonomy_resolver.resolve(sci_name, raw_desc)

            hypotheses.append(
                SpeciesHypothesis(
                    scientific_name=sci_name,
                    common_name=common_name,
                    family=family,
                    confidence=round(float(prob) * 100.0, 1),
                )
            )

        return hypotheses


# ==============================================================================
# Markdown Report Formatting
# ==============================================================================


def build_markdown_header(
    folder_path: Path,
    model_label: str,
    total_images: int,
) -> str:
    """Build the standardized header for `DETECTION.md`."""
    now_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    folder_display = folder_path.name or str(folder_path)
    return (
        "# 🐠 Underwater Species Identification Report — North Sulawesi (Lembeh & Bunaken)\n\n"
        f"- **Last analysis date:** {now_str}\n"
        f"- **Analyzed folder:** `{folder_display}`\n"
        f"- **Model used:** {model_label}\n"
        f"- **Total images detected:** {total_images}\n\n"
        "---\n\n"
    )


def format_prediction_entry(result: ImagePredictionResult) -> str:
    """Format a single image prediction result as a readable Markdown section."""
    lines = [f"### 📷 `{result.rel_path}`", ""]

    if result.error:
        lines.append(f"- ⚠️ **Analysis error:** `{result.error}`")
        lines.append("\n---\n")
        return "\n".join(lines)

    if not result.hypotheses:
        lines.append("- ⚠️ No hypotheses returned by the model.")
        lines.append("\n---\n")
        return "\n".join(lines)

    top1 = result.hypotheses[0]
    lines.append(
        f"- **Top 1:** **[{top1.scientific_name}]** (*{top1.common_name}*) "
        f"— **Confidence: {top1.confidence:.1f}%**"
    )
    lines.append(f"- **Taxonomic family:** `{top1.family}`")

    if len(result.hypotheses) > 1:
        alts = [
            f"*{h.scientific_name}* ({h.common_name}, {h.family} — {h.confidence:.1f}%)"
            for h in result.hypotheses[1:]
        ]
        lines.append(f"- **Alternatives:** {', '.join(alts)}")

    lines.append("")
    lines.append("| Rank | Scientific Name | Common Name | Family | Confidence |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")
    for idx, h in enumerate(result.hypotheses, start=1):
        lines.append(
            f"| #{idx} | **{h.scientific_name}** | {h.common_name} | `{h.family}` | {h.confidence:.1f}% |"
        )

    lines.append("\n---\n")
    return "\n".join(lines)


# ==============================================================================
# CLI Entrypoint
# ==============================================================================


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively analyze a folder of underwater photos (Lembeh & Bunaken) "
            "and generate a DETECTION.md report at the root of the target folder."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--folder",
        "-f",
        required=True,
        type=Path,
        help="Path to the folder containing underwater photos (.jpg, .jpeg, .png).",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["local"],
        default="local",
        help="Inference mode: 'local' (PyTorch MPS + iNaturalist 2021, zero cost).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the model identifier.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of hypotheses to return per image (3 to 5 recommended).",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1024,
        help="Maximum image dimension (px) when resizing in memory before inference.",
    )
    parser.add_argument(
        "--offline-taxonomy",
        action="store_true",
        help="Disable iNaturalist API requests for taxonomic family enrichment in local mode.",
    )
    return parser.parse_args(argv)


def create_classifier(args: argparse.Namespace) -> BaseClassifier:
    """Instantiate the selected inference engine."""
    if args.mode == "local":
        model_name = args.model or DEFAULT_LOCAL_MODEL
        return LocalTimmINatClassifier(
            model_name=model_name,
            enrich_taxonomy=not args.offline_taxonomy,
        )
    raise ValueError(f"Unknown mode: {args.mode}")


def main() -> None:
    args = parse_args()
    folder: Path = args.folder.expanduser().resolve()

    if not folder.exists() or not folder.is_dir():
        logger.error("The specified folder does not exist or is not a directory: %s", folder)
        sys.exit(1)

    output_md = folder / "DETECTION.md"

    all_images = discover_images(folder)
    total_images = len(all_images)

    if total_images == 0:
        logger.warning("No images (.jpg, .jpeg, .png) found in %s", folder)
        sys.exit(0)

    logger.info("%d image(s) found in %s", total_images, folder)

    try:
        classifier = create_classifier(args)
    except Exception as exc:
        logger.error("Failed to initialize classifier: %s", exc)
        sys.exit(1)

    header = build_markdown_header(folder, classifier.model_label, total_images)
    output_md.write_text(header, encoding="utf-8")

    success_count = 0
    error_count = 0

    for idx, img_path in enumerate(all_images, start=1):
        rel_path = img_path.relative_to(folder).as_posix()
        logger.info("[%d/%d] Analyzing %s...", idx, total_images, rel_path)

        try:
            img = load_and_resize_image(img_path, max_size=args.max_size)
        except (UnidentifiedImageError, OSError, SyntaxError) as exc:
            logger.warning("Skipping corrupted or unreadable image (%s): %s", rel_path, exc)
            error_count += 1
            continue

        try:
            hypotheses = classifier.predict(img, top_k=args.top_k)
            result = ImagePredictionResult(rel_path=rel_path, hypotheses=hypotheses)
            success_count += 1
        except Exception as exc:
            logger.error("Inference error on %s: %s", rel_path, exc)
            result = ImagePredictionResult(rel_path=rel_path, error=str(exc))
            error_count += 1

        entry_markdown = format_prediction_entry(result)
        with output_md.open("a", encoding="utf-8") as f:
            f.write(entry_markdown)

    logger.info(
        "Done! %d image(s) analyzed successfully (%d error(s)). Report written to: %s",
        success_count,
        error_count,
        output_md,
    )


if __name__ == "__main__":
    main()
