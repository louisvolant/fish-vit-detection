# Fish & Marine Life ViT Detection (North Sulawesi)

Standalone Python CLI tool optimized for **macOS Apple Silicon (M1/M2/M3/M4)** that recursively analyzes a folder of underwater photos and records species predictions in a `DETECTION.md` report at the root of the target folder.

Designed for Indo-Pacific marine biodiversity, including **North Sulawesi (Lembeh Strait & Bunaken)**: reef fishes, macro fauna, frogfishes, nudibranchs, and pleurobranchs.

---

## Features

- **Apple Silicon Hardware Acceleration (`mps`)**: Runs a Vision Transformer (`timm/eva02_large_patch14_clip_336.merged2b_ft_inat21`) fine-tuned on **iNaturalist 2021** (10,000 species) locally via PyTorch MPS at zero cost.
- **Taxonomic Enrichment**: Automatically queries the free public iNaturalist API (no key required, with in-memory caching) to retrieve exact taxonomic families and vernacular names.
- **Recursive & Safe Batch Processing**:
  - Recursively scans subfolders for `.jpg`, `.jpeg`, and `.png` images.
  - Automatically skips macOS resource forks (`._*`, `.DS_Store`), hidden directories, and corrupted images.
  - Resizes images proportionally in memory before inference to protect unified RAM.

---

## Installation

### Using `uv` (Recommended)

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Using `pip`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage

```bash
python scan.py --folder /path/to/underwater/photos
```

The report is generated at `/path/to/underwater/photos/DETECTION.md`.

### Running Tests

```bash
python3 -m unittest discover -s tests
```
