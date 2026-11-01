import tempfile
import unittest
from pathlib import Path

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

    def test_format_prediction_entry(self) -> None:
        result = scan.ImagePredictionResult(
            rel_path="bunaken/clownfish.jpg",
            hypotheses=[
                scan.SpeciesHypothesis(
                    scientific_name="Amphiprion ocellaris",
                    common_name="Ocellaris Clownfish",
                    family="Pomacentridae",
                    confidence=96.4,
                ),
                scan.SpeciesHypothesis(
                    scientific_name="Amphiprion percula",
                    common_name="Orange Clownfish",
                    family="Pomacentridae",
                    confidence=3.1,
                ),
            ],
        )
        md = scan.format_prediction_entry(result)
        self.assertIn("### 📷 `bunaken/clownfish.jpg`", md)
        self.assertIn("**[Amphiprion ocellaris]** (*Ocellaris Clownfish*)", md)
        self.assertIn("`Pomacentridae`", md)
        self.assertIn("*Amphiprion percula*", md)


if __name__ == "__main__":
    unittest.main()
