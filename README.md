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
- **Crash Recovery & Existing Report Policies (`--existing`)**:
  - Writes entries incrementally to `DETECTION.md`.
  - Automatically resumes interrupted runs without re-analyzing already recorded images, or allows full overwrite (`--overwrite`), interactive confirmation (`--existing ask`), or immediate exit (`--existing skip`).
- **Multi-Language Output (`--lang` / `-l`)**:
  - Supports writing `DETECTION.md` headers, table columns, and species vernacular names (for fishes, nudibranchs, pleurobranchs, and macro fauna) in your chosen language (`en`, `fr`, `id`, `es`, `de`, etc.).
  - Passes the language locale to both the iNaturalist taxonomy resolver and Vision LLM system prompts.

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

`DETECTION.md` is written directly at the root of the directory passed to `--folder`.

### Handling an Existing `DETECTION.md`

- **Option 1 — Resume (`--existing resume`, enabled by default)**:
  Reads `DETECTION.md`, skips images already listed, and appends only newly discovered images.
  ```bash
  python scan.py --folder /path/to/underwater/photos
  ```
- **Option 2 — Overwrite (`--overwrite` or `--existing overwrite`)**:
  Re-creates `DETECTION.md` from scratch and re-analyzes every image.
  ```bash
  python scan.py --folder /path/to/underwater/photos --overwrite
  ```
- **Additional Options (`ask` / `skip`)**:
  - `--existing ask`: Prompts interactively in the terminal (`Resume`, `Overwrite`, or `Cancel`).
  - `--existing skip`: Exits immediately if `DETECTION.md` already exists.

### Examples by Mode & Language

```bash
# 1. Local Mode (Apple Silicon MPS + iNaturalist 2021 model, default English output)
python scan.py --folder /path/to/underwater/photos --mode local

# 2. Local Mode with French vernacular names and French report
python scan.py --folder /path/to/underwater/photos --mode local --lang fr

# 3. Gemini Flash Vision API Mode (North Sulawesi context, French output)
export GEMINI_API_KEY="your_api_key"
python scan.py --folder /path/to/underwater/photos --mode gemini --lang fr

# 4. OpenAI Vision API Mode (GPT-4o-mini / GPT-4o, interactive existing policy)
export OPENAI_API_KEY="your_api_key"
python scan.py --folder /path/to/underwater/photos --mode openai --existing ask --lang en
```

### CLI Options

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--folder`, `-f` | *(required)* | Target folder containing underwater photos (`.jpg`, `.jpeg`, `.png`). |
| `--mode`, `-m` | `local` | Inference engine: `local` (PyTorch MPS), `gemini`, or `openai`. |
| `--lang`, `-l` | `en` | Output language for `DETECTION.md` and species common names (`en`, `fr`, `id`, `es`, `de`, etc.). |
| `--existing` | `resume` | Policy when `DETECTION.md` exists: `resume`, `overwrite`, `ask`, or `skip`. |
| `--overwrite` | `False` | Shorthand for `--existing overwrite`. |
| `--top-k` | `5` | Number of species hypotheses returned per image. |
| `--max-size` | `1024` | Max dimension (px) for in-memory image resizing before inference. |
| `--offline-taxonomy` | `False` | Disable public iNaturalist API queries in `local` mode. |

The report is generated at `/path/to/underwater/photos/DETECTION.md`.

### Running Tests

```bash
python3 -m unittest discover -s tests
```
