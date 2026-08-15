"""Offline unit tests for replacement-holdout selection contracts."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import gzip
from hashlib import sha256
import tempfile
import sys
import unittest

from PIL import Image


sys.path.insert(0, str(Path(__file__).parent))

from build_historical_index import (  # noqa: E402
    canonical_group,
    deterministic_gzip,
    load_docci_clusters,
    recover_one,
)
from prepare import (  # noqa: E402
    PerceptualIndex,
    ROOT,
    download,
    hamming,
    image_facts,
    priority,
    qualify,
    safe_output_path,
    validate_data_root,
)


class ReplacementPreparationTests(unittest.TestCase):
    def test_priority_is_namespaced_and_deterministic(self) -> None:
        self.assertEqual(priority("fixed:", 7), priority("fixed:", 7))
        self.assertNotEqual(priority("fixed:", 7), priority("other:", 7))
        self.assertNotEqual(priority("fixed:", 7), priority("fixed:", 8))

    def test_dhash_uses_exif_oriented_nine_by_eight_pixels(self) -> None:
        image = Image.new("RGB", (9, 8))
        for y in range(8):
            for x in range(9):
                image.putpixel((x, y), (x * 20, x * 20, x * 20))
        encoded = BytesIO()
        image.save(encoded, format="PNG")
        width, height, value, extension = image_facts(encoded.getvalue())
        self.assertEqual((width, height, extension), (9, 8, ".png"))
        self.assertEqual(len(value), 16)
        self.assertEqual(hamming(value, value), 0)

    def test_perceptual_index_is_complete_at_eight_bits(self) -> None:
        # Two differing bits in each of four 16-bit blocks forces the
        # pigeonhole boundary exercised by the 4x16-bit index.
        base = "0000000000000000"
        boundary = "0003000300030003"
        index = PerceptualIndex([(base, "old")])
        self.assertEqual(hamming(base, boundary), 8)
        self.assertEqual(index.matches(boundary, 8)[0]["matchingId"], "old")
        self.assertEqual(index.matches(boundary, 7), [])

    def test_qualify_fails_closed_for_every_overlap_dimension(self) -> None:
        base = {
            "id": "new-id",
            "imageSha256": "a" * 64,
            "sourceGroupId": "new-group",
            "perceptualDhash64": "0123456789abcdef",
        }
        empty = PerceptualIndex()
        self.assertEqual(qualify(base, {"new-id"}, set(), set(), empty, 8)["reason"], "id")
        self.assertEqual(qualify(base, set(), {"a" * 64}, set(), empty, 8)["reason"], "imageSha256")
        self.assertEqual(qualify(base, set(), set(), {"new-group"}, empty, 8)["reason"], "sourceGroupId")
        near = PerceptualIndex([("0123456789abcdef", "old-id")])
        self.assertEqual(qualify(base, set(), set(), set(), near, 8)["reason"], "perceptualDhash64")
        far = PerceptualIndex([("fedcba9876543210", "far")])
        self.assertIsNone(qualify(base, set(), set(), set(), far, 8))

    def test_docci_groups_come_from_official_cluster_metadata(self) -> None:
        row = {"datasetRevision": "revision"}
        left = canonical_group("docci:revision:train:train_06024", row, {"train_06024": "42"})
        right = canonical_group("docci:revision:test:test_00080", row, {"test_00080": "42"})
        self.assertEqual(left, right)
        self.assertEqual(left, "docci:revision:cluster:42")

    def test_offline_source_verification_rejects_same_size_poison(self) -> None:
        data_parent = ROOT / "benchmark" / "data"
        with tempfile.TemporaryDirectory(dir=data_parent) as temporary:
            path = Path(temporary) / "source.parquet"
            path.write_bytes(b"wrong")
            with self.assertRaisesRegex(ValueError, "offline verification"):
                download(
                    "https://example.invalid/source",
                    path,
                    5,
                    "0" * 64,
                    allow_download=False,
                    allowed_root=Path(temporary),
                )

    def test_deterministic_gzip_fixes_the_complete_header(self) -> None:
        value = b"prooflens deterministic historical evidence\n" * 50
        first = deterministic_gzip(value)
        second = deterministic_gzip(value)
        self.assertEqual(first, second)
        self.assertEqual(first[:10], b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff")
        self.assertEqual(gzip.decompress(first), value)

    def test_child_output_symlink_is_rejected(self) -> None:
        data_parent = ROOT / "benchmark" / "data"
        with tempfile.TemporaryDirectory(dir=data_parent) as outside_name:
            outside = Path(outside_name)
            with tempfile.TemporaryDirectory(dir=data_parent) as root_name:
                root = Path(root_name)
                (root / "confirmatory").symlink_to(outside, target_is_directory=True)
                with self.assertRaisesRegex(ValueError, "Symlinked data path component"):
                    safe_output_path(root, "confirmatory/escape.jpg")

    def test_child_source_symlink_is_rejected_before_offline_read(self) -> None:
        data_parent = ROOT / "benchmark" / "data"
        with tempfile.TemporaryDirectory(dir=data_parent) as outside_name:
            outside = Path(outside_name)
            source = outside / "source.parquet"
            source.write_bytes(b"valid")
            with tempfile.TemporaryDirectory(dir=data_parent) as root_name:
                root = Path(root_name)
                (root / "source").symlink_to(outside, target_is_directory=True)
                with self.assertRaisesRegex(ValueError, "Symlinked data path component"):
                    download(
                        "https://example.invalid/source",
                        root / "source" / "source.parquet",
                        5,
                        sha256(b"valid").hexdigest(),
                        allow_download=False,
                        allowed_root=root,
                    )

    def test_docci_metadata_missing_file_fails_closed_offline(self) -> None:
        data_parent = ROOT / "benchmark" / "data"
        with tempfile.TemporaryDirectory(dir=data_parent) as temporary:
            missing = Path(temporary) / "docci.jsonlines"
            recipe = {
                "historicalExclusions": {
                    "docciMetadata": {
                        "path": str(missing.relative_to(ROOT)),
                        "url": "https://example.invalid/docci",
                        "bytes": 1,
                        "sha256": sha256(b"x").hexdigest(),
                        "items": 1,
                    }
                }
            }
            with self.assertRaisesRegex(ValueError, "offline verification"):
                load_docci_clusters(recipe, allow_download=False)

    def test_docci_metadata_same_size_poison_fails_closed_offline(self) -> None:
        data_parent = ROOT / "benchmark" / "data"
        good = b'{"example_id":"a","cluster_id":"1"}\n'
        poison = b"x" * len(good)
        with tempfile.TemporaryDirectory(dir=data_parent) as temporary:
            path = Path(temporary) / "docci.jsonlines"
            path.write_bytes(poison)
            recipe = {
                "historicalExclusions": {
                    "docciMetadata": {
                        "path": str(path.relative_to(ROOT)),
                        "url": "https://example.invalid/docci",
                        "bytes": len(good),
                        "sha256": sha256(good).hexdigest(),
                        "items": 1,
                    }
                }
            }
            with self.assertRaisesRegex(ValueError, "offline verification"):
                load_docci_clusters(recipe, allow_download=False)

    def test_legacy_pixel_missing_fails_closed_offline(self) -> None:
        data_parent = ROOT / "benchmark" / "data"
        with tempfile.TemporaryDirectory(dir=data_parent) as temporary:
            recipe = {
                "historicalExclusions": {
                    "legacyPixelSources": {
                        "openImagesBaseUrl": "https://example.invalid/open-images",
                        "qwenDataset": "example/qwen",
                        "qwenRevision": "revision",
                    }
                }
            }
            with self.assertRaisesRegex(ValueError, "unavailable during offline verification"):
                recover_one(
                    "open-images:missing",
                    sha256(b"expected").hexdigest(),
                    recipe,
                    Path(temporary),
                    allow_download=False,
                )

    def test_data_root_rejects_root_symlink_escape(self) -> None:
        data_parent = ROOT / "benchmark" / "data"
        with tempfile.TemporaryDirectory(dir=data_parent) as outside_name:
            outside = Path(outside_name)
            with tempfile.TemporaryDirectory(dir=data_parent) as container_name:
                container = Path(container_name)
                link = container / "linked-root"
                link.symlink_to(outside, target_is_directory=True)
                with self.assertRaisesRegex(ValueError, "Symlinked data-root component"):
                    validate_data_root(link)

    def test_data_root_rejects_parent_symlink_escape(self) -> None:
        data_parent = ROOT / "benchmark" / "data"
        with tempfile.TemporaryDirectory(dir=data_parent) as outside_name:
            outside = Path(outside_name)
            with tempfile.TemporaryDirectory(dir=data_parent) as container_name:
                container = Path(container_name)
                link = container / "linked-parent"
                link.symlink_to(outside, target_is_directory=True)
                with self.assertRaisesRegex(ValueError, "Symlinked data-root component"):
                    validate_data_root(link / "child")


if __name__ == "__main__":
    unittest.main()
