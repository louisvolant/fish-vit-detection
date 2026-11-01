# Fish & Marine Life ViT Detection (North Sulawesi)

Standalone Python CLI tool optimized for **macOS Apple Silicon (M1/M2/M3/M4)** that recursively analyzes a folder of underwater photos and records species predictions in a `DETECTION.md` report at the root of the target folder.

Designed for Indo-Pacific marine biodiversity, including **North Sulawesi (Lembeh Strait & Bunaken)**: reef fishes, macro fauna, frogfishes, nudibranchs, and pleurobranchs.

---

## Features

- **Dual Inference Engines (`--mode`)**:
  1. **`--mode local` (Default, Zero Cost)**: Runs a Vision Transformer (`timm/eva02_large_patch14_clip_336.merged2b_ft_inat21`) fine-tuned on **iNaturalist 2021** (10,000 species) locally via PyTorch `mps` on Apple Silicon, enriched with the public iNaturalist API.
  2. **`--mode gemini` / `--mode openai` (Geo-Contextual Vision API)**: Uses `gemini-2.5-flash` or `gpt-4o-mini` with a specialized system prompt tailored to North Sulawesi marine fauna (Lembeh Strait muck diving & Bunaken coral reefs).
- **Taxonomic Enrichment**: Automatically retrieves exact taxonomic families and vernacular names along with top-K confidence scores.
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
# 1. Local Mode (Apple Silicon MPS + iNaturalist 2021 model)
python scan.py --folder /path/to/underwater/photos --mode local

# 2. Gemini Flash Vision API Mode (North Sulawesi context)
export GEMINI_API_KEY="your_api_key"
python scan.py --folder /path/to/underwater/photos --mode gemini

# 3. OpenAI Vision API Mode (GPT-4o-mini / GPT-4o)
export OPENAI_API_KEY="your_api_key"
python scan.py --folder /path/to/underwater/photos --mode openai
```

The report is generated at `/path/to/underwater/photos/DETECTION.md`.

### Running Tests

```bash
python3 -m unittest discover -s tests
```
