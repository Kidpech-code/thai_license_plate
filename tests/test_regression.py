from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from plate_recognition.config import load_domain_config, resolve_image_path
from plate_recognition.normalization import normalize_plate_line


REPO_ROOT = Path(__file__).resolve().parents[1]


class ConfigPathResolutionTests(unittest.TestCase):
    def test_resolve_image_path_falls_back_between_jpg_and_jpeg(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            existing_image = temp_path / "sample.jpg"
            existing_image.write_bytes(b"test")

            resolved_path = resolve_image_path(temp_path / "sample.jpeg")

            self.assertEqual(resolved_path, existing_image)


class ThaiPlateNormalizationRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.domain_config = load_domain_config(REPO_ROOT / "plate_config.json")

    def test_config_loads_verified_two_letter_series(self):
        self.assertIn("ฒก", self.domain_config.valid_two_letter_series_by_vehicle_type["private_pickup"])

    def test_pickup_prefix_prefers_correct_confusion_resolution(self):
        plate_line = normalize_plate_line(["ผก 8534", "อริ3"], "private_pickup", self.domain_config)

        self.assertEqual(plate_line, "ฒก 8534")

    def test_trailing_letters_after_digits_are_ignored(self):
        plate_line = normalize_plate_line(["ผกอ 8534"], "private_pickup", self.domain_config)

        self.assertEqual(plate_line, "ฒก 8534")


if __name__ == "__main__":
    unittest.main()