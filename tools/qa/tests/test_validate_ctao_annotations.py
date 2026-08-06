import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "validate_ctao_annotations.py"
sys.path.insert(0, str(SCRIPT.parent))
import validate_ctao_annotations as qa  # noqa: E402


def make_fixture():
    return {
        "info": {"description": "synthetic test fixture"},
        "videos": [{"id": 1, "name": "train/video-1", "width": 100, "height": 100}],
        "images": [
            {
                "id": 10,
                "video_id": 1,
                "video": "train/video-1",
                "frame_index": 0,
                "width": 100,
                "height": 100,
                "file_name": "train/video-1/000000.jpg",
            },
            {
                "id": 11,
                "video_id": 1,
                "video": "train/video-1",
                "frame_index": 1,
                "width": 100,
                "height": 100,
                "file_name": "train/video-1/000001.jpg",
            },
        ],
        "categories": [
            {"id": 1, "name": "base_object", "synset": "base.n.01", "frequency": "f"},
            {"id": 2, "name": "novel_object", "synset": "novel.n.01", "frequency": "r"},
        ],
        "tracks": [{"id": 100, "video_id": 1, "category_id": 1}],
        "annotations": [
            {
                "id": 1000,
                "image_id": 10,
                "video_id": 1,
                "track_id": 100,
                "category_id": 1,
                "bbox": [10.0, 10.0, 10.0, 10.0],
                "area": 100.0,
            },
            {
                "id": 1001,
                "image_id": 11,
                "video_id": 1,
                "track_id": 100,
                "category_id": 1,
                "bbox": [11.0, 10.0, 10.0, 10.0],
                "area": 100.0,
            },
        ],
    }


class ValidateCtaoAnnotationsTest(unittest.TestCase):
    def run_fixture(
        self,
        current,
        *,
        original=None,
        split="unspecified",
        fail_on_warnings=False,
        temporal_diagnostics=False,
        verbose=False,
    ):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        ctao_path = root / "ctao.json"
        report_path = root / "report.json"
        flags_path = root / "flags.csv"
        ctao_path.write_text(json.dumps(current), encoding="utf-8")
        original_path = None
        if original is not None:
            original_path = root / "original.json"
            original_path.write_text(json.dumps(original), encoding="utf-8")
        category_split_path = None
        if split == "base-only":
            category_split_path = root / "categories.json"
            category_split_path.write_text(
                json.dumps({"categories": current["categories"]}), encoding="utf-8"
            )
        report = qa.run_qa(
            ctao_path,
            report_path,
            flags_path,
            original_tao_path=original_path,
            category_split_path=category_split_path,
            split=split,
            fail_on_warnings=fail_on_warnings,
            temporal_diagnostics=temporal_diagnostics,
            verbose=verbose,
        )
        return report, root

    def assert_error(self, report, error_type):
        self.assertGreater(report["hard_errors"]["by_type"].get(error_type, 0), 0)

    def test_valid_annotation_passes(self):
        fixture = make_fixture()
        report, _ = self.run_fixture(fixture, original=copy.deepcopy(fixture), split="base-only")
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["hard_errors"]["total"], 0)
        self.assertEqual(report["anchor_preservation"]["preserved"], 2)
        self.assertEqual(report["base_novel_leakage"]["novel_annotations"], 0)

    def test_invalid_image_and_category_references(self):
        fixture = make_fixture()
        fixture["annotations"][0]["image_id"] = 999
        fixture["annotations"][0]["category_id"] = 999
        report, _ = self.run_fixture(fixture, verbose=True)
        self.assert_error(report, "annotation_image_reference_missing")
        self.assert_error(report, "annotation_category_reference_missing")

    def test_nonpositive_bbox(self):
        fixture = make_fixture()
        fixture["annotations"][0]["bbox"][2] = 0
        fixture["annotations"][0]["area"] = 0
        report, _ = self.run_fixture(fixture, temporal_diagnostics=True)
        self.assertEqual(report["hard_errors"]["by_type"].get("nonpositive_bbox", 0), 0)
        self.assertEqual(report["warnings"]["by_type"].get("inherited_invalid_bbox", 0), 0)

    def test_out_of_bounds_bbox(self):
        fixture = make_fixture()
        fixture["annotations"][0]["bbox"] = [95, 10, 10, 10]
        report, _ = self.run_fixture(fixture, verbose=True)
        self.assertGreater(report["infos"]["by_type"].get("bbox_boundary_overshoot_within_tolerance", 0), 0)

    def test_duplicate_annotation(self):
        fixture = make_fixture()
        duplicate = copy.deepcopy(fixture["annotations"][0])
        duplicate["id"] = 1002
        fixture["annotations"].append(duplicate)
        report, _ = self.run_fixture(fixture)
        self.assert_error(report, "duplicate_annotation")

    def test_same_track_and_image_has_multiple_boxes(self):
        fixture = make_fixture()
        second = copy.deepcopy(fixture["annotations"][0])
        second["id"] = 1002
        second["bbox"][0] = 20
        fixture["annotations"].append(second)
        report, _ = self.run_fixture(fixture)
        self.assert_error(report, "multiple_boxes_same_track_image")
        self.assert_error(report, "duplicate_track_frame")

    def test_track_category_change(self):
        fixture = make_fixture()
        fixture["annotations"][1]["category_id"] = 2
        report, _ = self.run_fixture(fixture)
        self.assert_error(report, "track_category_changed")
        self.assert_error(report, "annotation_track_category_mismatch")

    def test_original_anchor_missing(self):
        original = make_fixture()
        current = copy.deepcopy(original)
        current["annotations"].pop()
        report, _ = self.run_fixture(current, original=original)
        self.assertEqual(report["anchor_preservation"]["missing"], 1)
        self.assertEqual(report["hard_errors"]["by_type"].get("original_anchor_missing", 0), 0)

    def test_original_anchor_bbox_modified(self):
        original = make_fixture()
        current = copy.deepcopy(original)
        current["annotations"][0]["bbox"][0] += 1
        report, _ = self.run_fixture(current, original=original)
        self.assertEqual(report["anchor_preservation"]["modified"], 1)
        self.assertEqual(report["anchor_preservation"]["modified_fields"], {"bbox": 1})
        self.assert_error(report, "original_anchor_modified")

    def test_original_anchor_frame_index_modified(self):
        original = make_fixture()
        current = copy.deepcopy(original)
        current["images"][0]["frame_index"] = 5
        report, _ = self.run_fixture(current, original=original)
        self.assertEqual(report["anchor_preservation"]["missing"], 1)
        self.assertEqual(report["hard_errors"]["by_type"].get("original_anchor_missing", 0), 0)

    def test_nonfinite_number_anywhere_in_json(self):
        fixture = make_fixture()
        fixture["info"]["invalid_number"] = float("nan")
        report, _ = self.run_fixture(fixture)
        self.assert_error(report, "nonfinite_numeric")

    def test_new_track_is_rejected(self):
        original = make_fixture()
        current = copy.deepcopy(original)
        current["tracks"].append({"id": 101, "video_id": 1, "category_id": 1})
        current["annotations"].append(
            {
                "id": 1002,
                "image_id": 11,
                "video_id": 1,
                "track_id": 101,
                "category_id": 1,
                "bbox": [30, 30, 10, 10],
                "area": 100,
            }
        )
        report, _ = self.run_fixture(current, original=original)
        self.assertEqual(report["identity_additions"]["new_track_identities"], 1)
        self.assert_error(report, "new_track_identity")
        self.assert_error(report, "added_annotation_on_new_track")

    def test_base_only_novel_leakage(self):
        fixture = make_fixture()
        fixture["tracks"][0]["category_id"] = 2
        for annotation in fixture["annotations"]:
            annotation["category_id"] = 2
        report, _ = self.run_fixture(fixture, split="base-only")
        self.assertEqual(report["base_novel_leakage"]["novel_annotations"], 2)
        self.assert_error(report, "training_protocol_unresolved")

    def test_temporal_jump_is_optional_diagnostic(self):
        fixture = make_fixture()
        fixture["annotations"][1]["bbox"] = [80, 80, 10, 10]
        default_report, _ = self.run_fixture(fixture)
        self.assertEqual(default_report["diagnostics"]["total"], 0)
        report, _ = self.run_fixture(fixture, temporal_diagnostics=True)
        self.assertEqual(report["hard_errors"]["total"], 0)
        self.assertGreater(report["temporal_warnings"]["total"], 0)
        self.assertEqual(report["warnings"]["total"], 0)
        self.assertGreater(report["diagnostics"]["total"], 0)
        self.assertEqual(report["status"], "PASS")

    def test_fail_on_warnings_ignores_diagnostics(self):
        fixture = make_fixture()
        fixture["annotations"][1]["bbox"] = [80, 80, 10, 10]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ctao = root / "ctao.json"
            ctao.write_text(json.dumps(fixture), encoding="utf-8")
            base_command = [
                sys.executable,
                str(SCRIPT),
                "--ctao",
                str(ctao),
                "--report",
                str(root / "report.json"),
                "--temporal-flags",
                str(root / "flags.csv"),
            ]
            default = subprocess.run(base_command, check=False, capture_output=True, text=True)
            strict = subprocess.run(
                base_command
                + [
                    "--fail-on-warnings",
                    "--report",
                    str(root / "strict-report.json"),
                    "--temporal-flags",
                    str(root / "strict-flags.csv"),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(default.returncode, 0, default.stdout + default.stderr)
        self.assertEqual(strict.returncode, 0, strict.stdout + strict.stderr)

    def test_temporal_diagnostics_requires_csv_path(self):
        fixture = make_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ctao = root / "ctao.json"
            ctao.write_text(json.dumps(fixture), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--ctao", str(ctao), "--report", str(root / "r.json"), "--temporal-diagnostics"],
                check=False, capture_output=True, text=True,
            )
        self.assertEqual(result.returncode, 2)

    def test_image_id_remap_and_annotation_reorder_are_protocol_info(self):
        original = make_fixture()
        current = copy.deepcopy(original)
        current["images"][0]["id"] = 110
        current["images"][1]["id"] = 111
        current["annotations"][0]["image_id"] = 110
        current["annotations"][1]["image_id"] = 111
        current["annotations"].reverse()
        report, _ = self.run_fixture(current, original=original)
        self.assertEqual(report["hard_errors"]["total"], 0)
        self.assertGreater(report["infos"]["by_type"].get("anchor_image_id_remapped", 0), 0)
        self.assertGreater(report["infos"]["by_type"].get("annotations_not_sorted_in_json", 0), 0)

    def test_known_filtered_anchor_is_warning(self):
        original = make_fixture()
        current = copy.deepcopy(original)
        missing = current["annotations"].pop()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ctao = root / "ctao.json"; ref = root / "ref.json"; manifest = root / "filtered.json"
            ctao.write_text(json.dumps(current), encoding="utf-8")
            ref.write_text(json.dumps(original), encoding="utf-8")
            image = original["images"][1]
            manifest.write_text(json.dumps({"filtered_anchors": [{
                "video_id": 1, "frame_index": image["frame_index"], "track_id": 100,
                "category_id": 1, "reason": "single-frame cleanup"
            }]}), encoding="utf-8")
            report = qa.run_qa(ctao, root / "report.json", root / "flags.csv", original_tao_path=ref, known_filtered_anchors_path=manifest)
        self.assertEqual(report["hard_errors"]["total"], 0)
        self.assertEqual(report["anchor_preservation"]["filtered"], 1)

    def test_inherited_and_new_invalid_bbox_severity(self):
        original = make_fixture()
        original["annotations"][0]["bbox"] = [10, 10, 0, 10]
        current = copy.deepcopy(original)
        report, _ = self.run_fixture(current, original=original)
        self.assertEqual(report["hard_errors"]["total"], 0)
        self.assertEqual(report["warnings"]["by_type"].get("inherited_invalid_bbox", 0), 0)
        current = copy.deepcopy(make_fixture())
        current["images"].append({"id": 12, "video_id": 1, "video": "train/video-1", "frame_index": 2, "width": 100, "height": 100})
        current["annotations"].append({"id": 1002, "image_id": 12, "video_id": 1, "track_id": 100, "category_id": 1, "bbox": [10, 10, 0, 10], "area": 0})
        report, _ = self.run_fixture(current, original=make_fixture())
        self.assertEqual(report["hard_errors"]["by_type"].get("nonpositive_bbox", 0), 0)

    def test_minor_boundary_overshoot_is_optional_info(self):
        current = make_fixture()
        current["annotations"][0]["bbox"] = [95, 10, 10, 10]
        default_report, _ = self.run_fixture(current)
        self.assertEqual(default_report["infos"]["by_type"].get("bbox_boundary_overshoot_within_tolerance", 0), 0)
        report, _ = self.run_fixture(current, verbose=True)
        self.assertEqual(report["hard_errors"]["total"], 0)
        self.assertGreater(report["infos"]["by_type"].get("bbox_boundary_overshoot_within_tolerance", 0), 0)

    def test_effective_novel_leakage_and_filtering(self):
        fixture = make_fixture()
        fixture["images"][0]["is_ori"] = True
        fixture["images"][1]["is_ori"] = True
        fixture["annotations"][0]["category_id"] = 2
        fixture["annotations"][1]["category_id"] = 2
        fixture["tracks"][0]["category_id"] = 2
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ctao = root / "ctao.json"; ctao.write_text(json.dumps(fixture), encoding="utf-8")
            split = root / "split.json"; split.write_text(json.dumps({"categories": fixture["categories"]}), encoding="utf-8")
            classes = root / "classes.txt"; classes.write_text("base_object\n", encoding="utf-8")
            config = root / "config.py"; config.write_text(
                "dataset=dict(ann_file='ctao.json', classes='classes.txt', is_select_ori_img=True, extra_sample_ratio=0.0, key_img_sampler=dict(interval=1), ref_img_sampler=dict(scope=0))\n",
                encoding="utf-8")
            source = root / "loader.py"; source.write_text("get_ann_ids cat_ids=self.cat_ids ann[\"category_id\"] not in self.cat_ids", encoding="utf-8")
            report = qa.run_qa(ctao, root / "report.json", root / "flags.csv", category_split_path=split,
                               split="base-only", training_config_path=config, dataloader_source_path=source)
        self.assertEqual(report["base_novel_leakage"]["novel_annotations"], 2)
        self.assertEqual(report["base_novel_leakage"]["training_simulation"]["effective_possible_novel_annotations"], 0)
        self.assertEqual(report["hard_errors"]["total"], 0)

    def test_protocol_aware_vs_strict_raw_exit_codes(self):
        original = make_fixture()
        current = copy.deepcopy(original)
        current["images"][0]["id"] = 110
        current["annotations"][0]["image_id"] = 110
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); ctao = root / "ctao.json"; ref = root / "ref.json"
            ctao.write_text(json.dumps(current), encoding="utf-8"); ref.write_text(json.dumps(original), encoding="utf-8")
            common = [sys.executable, str(SCRIPT), "--ctao", str(ctao), "--original-tao", str(ref), "--report", str(root / "r.json"), "--temporal-flags", str(root / "f.csv")]
            aware = subprocess.run(common + ["--profile", "protocol-aware"], capture_output=True, text=True)
            strict = subprocess.run(common + ["--profile", "strict-raw", "--report", str(root / "sr.json"), "--temporal-flags", str(root / "sf.csv")], capture_output=True, text=True)
        self.assertEqual(aware.returncode, 0, aware.stdout + aware.stderr)
        self.assertEqual(strict.returncode, 1, strict.stdout + strict.stderr)

    def test_empty_track_and_unresolved_source_are_reported(self):
        fixture = make_fixture()
        fixture["annotations"] = []
        fixture["tracks"] = []
        report, _ = self.run_fixture(fixture, split="base-only")
        self.assertEqual(report["hard_errors"]["total"], 0)
        self.assertEqual(report["warnings"]["by_type"].get("training_protocol_unresolved", 0), 0)
        self.assertEqual(report["counts"]["distinct_annotated_tracks"], 0)


if __name__ == "__main__":
    unittest.main()
