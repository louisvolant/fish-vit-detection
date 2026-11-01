import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, UnidentifiedImageError

import scan


class TestScanCore(unittest.TestCase):
    def test_discover_images_ignores_hidden_and_macos_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "nudibranch.jpg").write_bytes(b"dummy")
            (root / "._nudibranch.jpg").write_bytes(b"dummy")
            (root / ".DS_Store").write_bytes(b"dummy")

            sub = root / "lembeh"
            sub.mkdir()
            (sub / "frogfish.PNG").write_bytes(b"dummy")

            hidden_dir = root / ".hidden"
            hidden_dir.mkdir()
            (hidden_dir / "secret.jpeg").write_bytes(b"dummy")

            discovered = scan.discover_images(root)
            rel_paths = [p.relative_to(root).as_posix() for p in discovered]
            self.assertEqual(rel_paths, ["nudibranch.jpg", "lembeh/frogfish.PNG"])

    def test_load_and_resize_image_preserves_aspect_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "large.jpg"
            img = Image.new("RGB", (2400, 1200), color=(10, 50, 120))
            img.save(img_path, format="JPEG")

            resized = scan.load_and_resize_image(img_path, max_size=600)
            self.assertEqual(resized.size, (600, 300))

    def test_load_and_resize_corrupted_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = Path(tmpdir) / "corrupted.jpg"
            bad_path.write_bytes(b"not-a-valid-image")

            with self.assertRaises((UnidentifiedImageError, OSError)):
                scan.load_and_resize_image(bad_path)

    def test_taxonomy_resolver_offline_fallback(self) -> None:
        resolver = scan.INaturalistTaxonomyResolver(enabled=False)
        common, family = resolver.resolve(
            "Chromodoris annae",
            fallback_desc="Anna's Chromodoris, Chromodorididae",
        )
        self.assertEqual(common, "Anna's Chromodoris")
        self.assertEqual(family, "Chromodorididae")

    def test_select_best_taxon_match_prefers_exact_species_over_complex(self) -> None:
        results = [
            {"id": 1599546, "name": "Amphiprion percula", "rank": "complex"},
            {"id": 132688, "name": "Amphiprion ocellaris", "rank": "species"},
        ]
        chosen = scan.INaturalistTaxonomyResolver._select_best_taxon_match(
            results, "Amphiprion ocellaris"
        )
        self.assertEqual(chosen["id"], 132688)

    def test_format_prediction_entry_filters_below_5_percent(self) -> None:
        result = scan.ImagePredictionResult(
            rel_path="bunaken/clownfish.jpg",
            hypotheses=[
                scan.SpeciesHypothesis(
                    scientific_name="Amphiprion ocellaris",
                    common_name="Ocellaris Clownfish",
                    family="Pomacentridae",
                    confidence=91.4,
                ),
                scan.SpeciesHypothesis(
                    scientific_name="Amphiprion percula",
                    common_name="Orange Clownfish",
                    family="Pomacentridae",
                    confidence=6.2,
                ),
                scan.SpeciesHypothesis(
                    scientific_name="Premnas biaculeatus",
                    common_name="Spinecheek Anemonefish",
                    family="Pomacentridae",
                    confidence=2.4,
                ),
            ],
        )
        md = scan.format_prediction_entry(result)
        self.assertIn("### 📷 `bunaken/clownfish.jpg`", md)
        self.assertIn("**[Amphiprion ocellaris]** (*Ocellaris Clownfish*)", md)
        self.assertIn("`Pomacentridae`", md)
        self.assertIn("*Amphiprion percula*", md)
        self.assertNotIn("Premnas biaculeatus", md)

        low_conf_result = scan.ImagePredictionResult(
            rel_path="lembeh/blurry_sand.jpg",
            hypotheses=[
                scan.SpeciesHypothesis(
                    scientific_name="Inimicus didactylus",
                    common_name="Spiny Devilfish",
                    family="Synanceiidae",
                    confidence=3.8,
                )
            ],
        )
        low_md = scan.format_prediction_entry(low_conf_result, lang="fr")
        self.assertIn("Aucune espèce n'a atteint le seuil minimal de confiance.", low_md)
        self.assertNotIn("Inimicus didactylus", low_md)

    def test_parse_json_predictions_handles_markdown_fences(self) -> None:
        raw = """```json
        {
          "predictions": [
            {
              "scientific_name": "Hippocampus bargibanti",
              "common_name": "Bargibant's Pygmy Seahorse",
              "family": "Syngnathidae",
              "confidence": 94.2
            },
            {
              "scientific_name": "Hippocampus denise",
              "common_name": "Denise's Pygmy Seahorse",
              "family": "Syngnathidae",
              "confidence": 5.1
            }
          ]
        }
        ```"""
        preds = scan._parse_json_predictions(raw, top_k=1)
        self.assertEqual(len(preds), 1)
        self.assertEqual(preds[0].scientific_name, "Hippocampus bargibanti")
        self.assertEqual(preds[0].family, "Syngnathidae")
        self.assertAlmostEqual(preds[0].confidence, 94.2)

    def test_api_classifiers_require_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                scan.GeminiVisionClassifier(api_key=None)
            with self.assertRaises(RuntimeError):
                scan.OpenAIVisionClassifier(api_key=None)

    def test_parse_existing_detection_md_and_resume_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            md_path = folder / "DETECTION.md"
            header = scan.build_markdown_header(folder, "TestModel", 2)
            entry = scan.format_prediction_entry(
                scan.ImagePredictionResult(
                    rel_path="lembeh/wonderpus.jpg",
                    hypotheses=[
                        scan.SpeciesHypothesis(
                            scientific_name="Wunderpus photogenicus",
                            common_name="Wonderpus Octopus",
                            family="Octopodidae",
                            confidence=98.0,
                        )
                    ],
                )
            )
            md_path.write_text(header + entry, encoding="utf-8")

            processed = scan.parse_existing_detection_md(md_path)
            self.assertEqual(processed, {"lembeh/wonderpus.jpg"})

            should_overwrite, existing = scan.resolve_existing_file_policy(
                md_path, policy="resume", force_overwrite=False
            )
            self.assertFalse(should_overwrite)
            self.assertEqual(existing, {"lembeh/wonderpus.jpg"})

            should_overwrite_2, existing_2 = scan.resolve_existing_file_policy(
                md_path, policy="resume", force_overwrite=True
            )
            self.assertTrue(should_overwrite_2)
            self.assertEqual(existing_2, set())

            scan.update_markdown_header_metadata(md_path, folder, "UpdatedModel", 5)
            updated_text = md_path.read_text(encoding="utf-8")
            self.assertIn("UpdatedModel", updated_text)
            self.assertIn("`lembeh/wonderpus.jpg`", updated_text)

    def test_multilingual_report_formatting_and_system_prompt(self) -> None:
        result = scan.ImagePredictionResult(
            rel_path="lembeh/pleurobranch.jpg",
            hypotheses=[
                scan.SpeciesHypothesis(
                    scientific_name="Pleurobranchus forskalii",
                    common_name="Pleurobranche de Forskal",
                    family="Pleurobranchidae",
                    confidence=95.0,
                )
            ],
        )
        fr_entry = scan.format_prediction_entry(result, lang="fr")
        self.assertIn("Famille taxonomique", fr_entry)
        self.assertIn("Nom vernaculaire", fr_entry)
        self.assertIn("Confiance: 95.0%", fr_entry)

        id_header = scan.build_markdown_header(
            Path("/tmp/dive_photos"),
            "TestModel",
            10,
            lang="id",
        )
        self.assertIn("Laporan Identifikasi Spesies Bawah Air", id_header)
        self.assertIn("Bahasa laporan:** Indonesian (`id`)", id_header)

        prompt_fr = scan.build_system_prompt_sulawesi("fr")
        self.assertIn("French", prompt_fr)

        args = scan.parse_args(["--folder", "/tmp/photos", "--lang", "fr"])
        self.assertEqual(args.lang, "fr")

    def test_resolve_local_model_presets_vit_and_cnn(self) -> None:
        self.assertEqual(scan.resolve_local_model_name(None), scan.DEFAULT_LOCAL_MODEL)
        self.assertEqual(scan.resolve_local_model_name("vit"), scan.DEFAULT_LOCAL_MODEL)
        self.assertEqual(scan.resolve_local_model_name("cnn"), scan.DEFAULT_CNN_MODEL)
        self.assertEqual(scan.resolve_local_model_name("convnext"), scan.DEFAULT_CNN_MODEL)
        self.assertEqual(
            scan.resolve_local_model_name("hf-hub:timm/custom_model"),
            "hf-hub:timm/custom_model",
        )


if __name__ == "__main__":
    unittest.main()
