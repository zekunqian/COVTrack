#!/usr/bin/env python3
"""Read-only quality assurance for C-TAO/TAO COCO-video annotations."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


TOOL_VERSION = "1.0.0"
REQUIRED_TOP_LEVEL = ("videos", "images", "annotations", "tracks", "categories")
ERROR_GROUPS = (
    "schema",
    "reference",
    "bbox",
    "duplicate",
    "track_category",
    "identity",
    "anchor",
    "leakage",
    "split_overlap",
)
CSV_FIELDS = (
    "video_id",
    "track_id",
    "frame_t",
    "frame_t1",
    "flag_type",
    "measured_value",
    "threshold",
)


class ErrorLog:
    """Count every hard error while retaining only bounded examples."""

    def __init__(self, max_examples: int = 20) -> None:
        self.max_examples = max_examples
        self.counts: Counter[str] = Counter()
        self.group_counts: Counter[str] = Counter()
        self.examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def add(self, kind: str, group: str, **details: Any) -> None:
        self.add_count(kind, group, 1)
        if len(self.examples[kind]) < self.max_examples:
            self.examples[kind].append(details)

    def add_count(
        self,
        kind: str,
        group: str,
        count: int,
        example: Optional[Dict[str, Any]] = None,
    ) -> None:
        if count <= 0:
            return
        self.counts[kind] += count
        self.group_counts[group] += count
        if example is not None and len(self.examples[kind]) < self.max_examples:
            self.examples[kind].append(example)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "by_group": {group: self.group_counts.get(group, 0) for group in ERROR_GROUPS},
            "by_type": dict(sorted(self.counts.items())),
            "examples": dict(sorted(self.examples.items())),
            "examples_are_capped_at": self.max_examples,
        }


class DiagnosticLog:
    """Bounded INFO/WARNING diagnostics which never affect hard status."""

    def __init__(self, max_examples: int = 20) -> None:
        self.max_examples = max_examples
        self.counts: Counter[str] = Counter()
        self.examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def add(self, kind: str, **details: Any) -> None:
        self.counts[kind] += 1
        if len(self.examples[kind]) < self.max_examples:
            self.examples[kind].append(details)

    def add_count(self, kind: str, count: int, example: Optional[Dict[str, Any]] = None) -> None:
        if count <= 0:
            return
        self.counts[kind] += count
        if example is not None and len(self.examples[kind]) < self.max_examples:
            self.examples[kind].append(example)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "by_type": dict(sorted(self.counts.items())),
            "examples": dict(sorted(self.examples.items())),
            "examples_are_capped_at": self.max_examples,
        }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return _is_number(value) and math.isfinite(value)


def _valid_id(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        hash(value)
    except TypeError:
        return False
    return isinstance(value, (int, str)) and value != ""


def _count_nonfinite(value: Any) -> int:
    if isinstance(value, float):
        return 0 if math.isfinite(value) else 1
    if isinstance(value, Mapping):
        return sum(_count_nonfinite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_count_nonfinite(item) for item in value)
    return 0


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("top-level JSON value must be an object")
    return data


def _semantic_key(annotation: Mapping[str, Any], images: Mapping[Any, Mapping[str, Any]]) -> Optional[Tuple[Any, ...]]:
    """Return the stable TAO identity used across C-TAO image-ID rebuilds."""
    image = images.get(annotation.get("image_id"))
    if image is None:
        return None
    frame = image.get("frame_index")
    if frame is None:
        frame = image.get("frame_id")
    if frame is None:
        return None
    return (
        annotation.get("video_id"),
        frame,
        annotation.get("track_id"),
        annotation.get("category_id"),
    )


def _semantic_index(
    annotations: Iterable[Mapping[str, Any]], images: Mapping[Any, Mapping[str, Any]], errors: ErrorLog
) -> Dict[Tuple[Any, ...], Mapping[str, Any]]:
    result: Dict[Tuple[Any, ...], Mapping[str, Any]] = {}
    for annotation in annotations:
        key = _semantic_key(annotation, images)
        if key is None:
            continue
        if key in result:
            errors.add("duplicate_semantic_anchor_key", "duplicate", semantic_key=list(key))
        else:
            result[key] = annotation
    return result


def _load_known_filtered_anchors(path: Optional[Path], errors: ErrorLog) -> Dict[Tuple[Any, ...], str]:
    if path is None:
        return {}
    data = _load_json(path)
    entries: Any = data
    if isinstance(data, dict):
        entries = data.get("filtered_anchors", data.get("anchors", []))
    if not isinstance(entries, list):
        errors.add("known_filter_manifest_invalid", "anchor", path=str(path))
        return {}
    result: Dict[Tuple[Any, ...], str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            errors.add("known_filter_manifest_entry_invalid", "anchor", entry=entry)
            continue
        frame = entry.get("frame_index", entry.get("frame_id"))
        key = (entry.get("video_id"), frame, entry.get("track_id"), entry.get("category_id"))
        if None in key:
            errors.add("known_filter_manifest_key_invalid", "anchor", entry=entry)
            continue
        result[key] = str(entry.get("reason", "declared construction filter"))
    return result


def _literal_ast(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict":
            value: Dict[str, Any] = {}
            for keyword in node.keywords:
                if keyword.arg is not None:
                    value[keyword.arg] = _literal_ast(keyword.value)
            return value
        return None


def _inspect_training_config(
    config_path: Optional[Path],
    source_path: Optional[Path],
    ctao_path: Path,
    data: Mapping[str, Any],
    errors: ErrorLog,
) -> Dict[str, Any]:
    """Statically reproduce the category/image filtering used by TaoDataset."""
    result: Dict[str, Any] = {
        "checked": False,
        "config_path": str(config_path.resolve()) if config_path else None,
        "source_path": str(source_path.resolve()) if source_path else None,
        "filter_logic_confirmed": False,
        "reason": None,
    }
    if config_path is None:
        result["reason"] = "--training-config was not supplied"
        result["unresolved"] = True
        return result
    if not config_path.is_file():
        result["reason"] = "training config does not exist"
        result["unresolved"] = True
        return result
    try:
        tree = ast.parse(config_path.read_text(encoding="utf-8"), filename=str(config_path))
    except (OSError, SyntaxError) as exc:
        result["reason"] = "cannot parse training config: {}".format(exc)
        result["unresolved"] = True
        return result
    matches: List[Dict[str, Any]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict"):
            continue
        values = {keyword.arg: _literal_ast(keyword.value) for keyword in node.keywords if keyword.arg}
        ann_file = values.get("ann_file")
        if isinstance(ann_file, str) and Path(ann_file).name == ctao_path.name:
            matches.append(values)
    if not matches:
        result["reason"] = "no dataset dict references the selected C-TAO filename"
        result["unresolved"] = True
        return result
    options = matches[-1]
    classes_file = options.get("classes")
    if isinstance(classes_file, str):
        class_path = Path(classes_file)
        candidates = [Path.cwd() / class_path, config_path.parent / class_path]
        class_path = next((candidate for candidate in candidates if candidate.is_file()), class_path)
        result["classes_path"] = str(class_path.resolve()) if class_path.is_file() else str(class_path)
        if class_path.is_file():
            names = {line.strip() for line in class_path.read_text(encoding="utf-8").splitlines() if line.strip()}
            categories = data.get("categories", [])
            result["allowed_category_ids"] = {
                c.get("id") for c in categories if isinstance(c, dict) and c.get("name") in names
            }
            result["allowed_category_count"] = len(result["allowed_category_ids"])
            result["class_name_count"] = len(names)
            result["novel_category_excluded"] = any(
                c.get("frequency") == "r" and c.get("name") not in names
                for c in categories if isinstance(c, dict)
            )
        else:
            result["reason"] = "classes file referenced by config does not exist"
            result["unresolved"] = True
            return result
    else:
        result["reason"] = "dataset classes option could not be resolved"
        result["unresolved"] = True
        return result
    source_text = ""
    if source_path is not None and source_path.is_file():
        source_text = source_path.read_text(encoding="utf-8")
        companion = source_path.with_name("coco_video_dataset.py")
        if companion.is_file():
            source_text += "\n" + companion.read_text(encoding="utf-8")
    result["filter_logic_confirmed"] = (
        "get_ann_ids" in source_text
        and "cat_ids=self.cat_ids" in source_text
        and 'ann["category_id"] not in self.cat_ids' in source_text
    )
    if source_path is not None and not result["filter_logic_confirmed"]:
        result["reason"] = "dataloader source does not match the verified category filter pattern"
        result["unresolved"] = True
        return result
    result.update(
        {
            "is_select_ori_img": options.get("is_select_ori_img"),
            "extra_sample_ratio": options.get("extra_sample_ratio", 0.0),
            "key_img_sampler": options.get("key_img_sampler"),
            "ref_img_sampler": options.get("ref_img_sampler"),
            "checked": True,
            "reason": "verified TaoDataset image/category filtering",
            "unresolved": False,
        }
    )
    return result


def _simulate_training_filter(
    data: Mapping[str, Any], protocol: Mapping[str, Any], novel_ids: Set[Any]
) -> Dict[str, Any]:
    images = {image.get("id"): image for image in data.get("images", []) if isinstance(image, dict)}
    annotations_by_image: Dict[Any, List[Mapping[str, Any]]] = defaultdict(list)
    for annotation in data.get("annotations", []):
        if isinstance(annotation, dict):
            annotations_by_image[annotation.get("image_id")].append(annotation)
    by_video: Dict[Any, List[Mapping[str, Any]]] = defaultdict(list)
    for image in images.values():
        by_video[image.get("video_id")].append(image)
    key_ids: Set[Any] = set()
    reference_ids: Set[Any] = set()
    ratio = protocol.get("extra_sample_ratio", 0.0) or 0.0
    sampler = protocol.get("key_img_sampler") or {}
    interval = sampler.get("interval", 1) if isinstance(sampler, dict) else 1
    ref_sampler = protocol.get("ref_img_sampler") or {}
    scope = ref_sampler.get("scope", 0) if isinstance(ref_sampler, dict) else 0
    for video_id, records in by_video.items():
        records = sorted(records, key=lambda item: (item.get("frame_id", 0), str(item.get("id"))))
        all_ids = [record.get("id") for record in records][:: max(1, int(interval))]
        if protocol.get("is_select_ori_img"):
            selected = [record.get("id") for record in records if record.get("is_ori") is True]
            available = list(set(all_ids) - set(selected))
            count = int(len(selected) * float(ratio))
            if len(available) <= count:
                extra = available
            else:
                state = random.getstate()
                random.seed(42)
                extra = random.sample(available, count)
                random.setstate(state)
            selected += extra
        else:
            selected = all_ids
        key_ids.update(selected)
        positions = {record.get("id"): index for index, record in enumerate(records)}
        for image_id in selected:
            position = positions.get(image_id)
            if position is None:
                continue
            left, right = max(0, position - int(scope)), min(len(records) - 1, position + int(scope))
            reference_ids.update(record.get("id") for record in records[left : right + 1] if record.get("id") != image_id)
    training_ids = key_ids | reference_ids
    allowed_ids = protocol.get("allowed_category_ids")
    if not isinstance(allowed_ids, set):
        allowed_ids = set(allowed_ids) if isinstance(allowed_ids, (list, tuple)) else None
    def effective(annotation: Mapping[str, Any]) -> bool:
        image = images.get(annotation.get("image_id"))
        if image is None or annotation.get("image_id") not in training_ids:
            return False
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            return False
        x, y, width, height = bbox
        inter_w = max(0, min(x + width, image.get("width", 0)) - max(x, 0))
        inter_h = max(0, min(y + height, image.get("height", 0)) - max(y, 0))
        return (
            not annotation.get("ignore", False)
            and inter_w * inter_h != 0
            and annotation.get("area", 0) > 0
            and width >= 1
            and height >= 1
            and (allowed_ids is None or annotation.get("category_id") in allowed_ids)
        )
    key_novel = sum(
        effective(annotation) and annotation.get("image_id") in key_ids and annotation.get("category_id") in novel_ids
        for annotation in data.get("annotations", [])
    )
    possible_novel = sum(
        effective(annotation) and annotation.get("category_id") in novel_ids
        for annotation in data.get("annotations", [])
    )
    return {
        "key_images": len(key_ids),
        "reference_candidate_images": len(reference_ids),
        "training_image_universe": len(training_ids),
        "effective_key_novel_annotations": key_novel,
        "effective_possible_novel_annotations": possible_novel,
    }


def _discover_reference_paths(
    original_full: Optional[Path],
    original_base: Optional[Path],
    reference_dir: Optional[Path],
    split: str,
) -> Dict[str, Any]:
    candidates: List[Path] = []
    if reference_dir and reference_dir.is_dir():
        candidates.extend(sorted(reference_dir.glob("*train*base*.json")))
        candidates.extend(sorted(reference_dir.glob("*train*.json")))
    if original_full:
        candidates.append(original_full.with_name(original_full.stem + "_base.json"))
    candidates = list(dict.fromkeys(candidates))
    selected = original_base if split == "base-only" and original_base else original_full
    if split == "base-only" and selected is None:
        selected = next((path for path in candidates if path.is_file() and "base" in path.stem), None)
    if selected is None:
        selected = original_full
    return {
        "selected": str(selected.resolve()) if selected else None,
        "full": str(original_full.resolve()) if original_full else None,
        "base": str(original_base.resolve()) if original_base else None,
        "discovered_candidates": [str(path.resolve()) for path in candidates if path.is_file()],
        "selection_reason": "base-only selects an existing base reference; otherwise full reference",
    }


def _file_metadata(path: Optional[Path], sha256: bool) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    result: Dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
    }
    if sha256:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        result["sha256"] = digest.hexdigest()
    return result


def _records(data: Mapping[str, Any], key: str, errors: ErrorLog) -> List[Dict[str, Any]]:
    value = data.get(key, [])
    if not isinstance(value, list):
        errors.add("top_level_not_array", "schema", field=key, actual_type=type(value).__name__)
        return []
    result: List[Dict[str, Any]] = []
    malformed = 0
    for index, item in enumerate(value):
        if isinstance(item, dict):
            result.append(item)
        else:
            malformed += 1
            errors.add("record_not_object", "schema", section=key, index=index)
    return result


def _make_index(
    records: Iterable[Mapping[str, Any]], section: str, errors: ErrorLog
) -> Dict[Any, Mapping[str, Any]]:
    result: Dict[Any, Mapping[str, Any]] = {}
    for position, record in enumerate(records):
        record_id = record.get("id")
        if not _valid_id(record_id):
            kind = "annotation_id_missing_or_invalid" if section == "annotations" else "id_missing_or_invalid"
            errors.add(kind, "schema", section=section, position=position, value=record_id)
            continue
        if record_id in result:
            errors.add("duplicate_id", "schema", section=section, id=record_id)
        else:
            result[record_id] = record
    return result


def _validate_dimensions(
    record: Mapping[str, Any], section: str, errors: ErrorLog
) -> None:
    for field in ("width", "height"):
        value = record.get(field)
        if not _is_finite_number(value) or value <= 0:
            errors.add(
                "invalid_dimension",
                "schema",
                section=section,
                id=record.get("id"),
                field=field,
                value=value,
            )


def _valid_bbox(bbox: Any) -> bool:
    return (
        isinstance(bbox, list)
        and len(bbox) == 4
        and all(_is_finite_number(value) for value in bbox)
    )


def _boxes_close(left: Any, right: Any, tolerance: float) -> bool:
    return _valid_bbox(left) and _valid_bbox(right) and all(
        math.isclose(a, b, rel_tol=0.0, abs_tol=tolerance) for a, b in zip(left, right)
    )


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx1, ly1, lw, lh = left
    rx1, ry1, rw, rh = right
    lx2, ly2 = lx1 + lw, ly1 + lh
    rx2, ry2 = rx1 + rw, ry1 + rh
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0 else 0.0


def _load_category_split(
    path: Path, errors: ErrorLog
) -> Tuple[Set[Any], Set[Any], Dict[str, Any]]:
    data = _load_json(path)
    categories: Any = data.get("categories")
    if categories is None and isinstance(data, dict):
        if "base_category_ids" in data and "novel_category_ids" in data:
            base = set(data["base_category_ids"])
            novel = set(data["novel_category_ids"])
            source = "explicit base_category_ids/novel_category_ids"
        elif "base" in data and "novel" in data:
            base = set(data["base"])
            novel = set(data["novel"])
            source = "explicit base/novel IDs"
        else:
            errors.add(
                "category_split_unrecognized",
                "leakage",
                path=str(path),
                expected="categories with frequency, or explicit base/novel ID arrays",
            )
            return set(), set(), {"source": None}
    else:
        if not isinstance(categories, list):
            errors.add("category_split_unrecognized", "leakage", path=str(path))
            return set(), set(), {"source": None}
        base, novel = set(), set()
        missing_frequency = 0
        for category in categories:
            if not isinstance(category, dict) or not _valid_id(category.get("id")):
                errors.add("category_split_invalid_category", "leakage", category=category)
                continue
            frequency = category.get("frequency")
            if frequency == "r":
                novel.add(category["id"])
            elif frequency in ("c", "f"):
                base.add(category["id"])
            else:
                missing_frequency += 1
        errors.add_count(
            "category_split_missing_frequency",
            "leakage",
            missing_frequency,
            {"path": str(path)},
        )
        source = "category frequency (r=novel; c/f=base)"
    overlap = base & novel
    if overlap:
        errors.add_count(
            "category_split_overlap",
            "leakage",
            len(overlap),
            {"category_ids": sorted(overlap, key=str)[:20]},
        )
    return base, novel, {
        "path": str(path.resolve()),
        "source": source,
        "base_categories": len(base),
        "novel_categories": len(novel),
    }


def _identity_checks(
    current: Mapping[str, Mapping[Any, Mapping[str, Any]]],
    original: Mapping[str, Mapping[Any, Mapping[str, Any]]],
    errors: ErrorLog,
) -> Dict[str, Any]:
    additions: Dict[str, int] = {}
    fields = {
        "videos": ("name",),
        "categories": ("name", "synset"),
        "tracks": ("video_id", "category_id"),
    }
    for section, identity_fields in fields.items():
        current_index = current[section]
        original_index = original[section]
        new_ids = set(current_index) - set(original_index)
        additions[section] = len(new_ids)
        if new_ids:
            errors.add_count(
                "new_{}_identity".format(section[:-1]),
                "identity",
                len(new_ids),
                {"ids": sorted(new_ids, key=str)[:20]},
            )
        for record_id in set(current_index) & set(original_index):
            changed = {
                field: {"original": original_index[record_id].get(field), "ctao": current_index[record_id].get(field)}
                for field in identity_fields
                if original_index[record_id].get(field) != current_index[record_id].get(field)
            }
            if changed:
                errors.add(
                    "{}_identity_modified".format(section[:-1]),
                    "identity",
                    id=record_id,
                    changed=changed,
                )
    return {
        "new_video_identities": additions["videos"],
        "new_category_identities": additions["categories"],
        "new_track_identities": additions["tracks"],
    }


def _anchor_checks(
    annotations: Mapping[Any, Mapping[str, Any]],
    original_annotations: Mapping[Any, Mapping[str, Any]],
    images: Mapping[Any, Mapping[str, Any]],
    original_images: Mapping[Any, Mapping[str, Any]],
    original_track_ids: Set[Any],
    tolerance: float,
    errors: ErrorLog,
) -> Tuple[Dict[str, Any], Set[Any]]:
    preserved = missing = modified = 0
    modified_fields: Counter[str] = Counter()
    anchor_fields = ("image_id", "video_id", "track_id", "category_id")
    for annotation_id, original in original_annotations.items():
        current = annotations.get(annotation_id)
        if current is None:
            missing += 1
            errors.add("original_anchor_missing", "anchor", annotation_id=annotation_id)
            continue
        changed = {
            field: {"original": original.get(field), "ctao": current.get(field)}
            for field in anchor_fields
            if original.get(field) != current.get(field)
        }
        original_image = original_images.get(original.get("image_id"), {})
        current_image = images.get(current.get("image_id"), {})
        if original_image.get("frame_index") != current_image.get("frame_index"):
            changed["frame_index"] = {
                "original": original_image.get("frame_index"),
                "ctao": current_image.get("frame_index"),
            }
        if not _boxes_close(original.get("bbox"), current.get("bbox"), tolerance):
            changed["bbox"] = {"original": original.get("bbox"), "ctao": current.get("bbox")}
        if changed:
            modified += 1
            modified_fields.update(changed.keys())
            errors.add(
                "original_anchor_modified",
                "anchor",
                annotation_id=annotation_id,
                changed=changed,
            )
        else:
            preserved += 1

    added_ids = set(annotations) - set(original_annotations)
    added_on_new_tracks = 0
    for annotation_id in added_ids:
        annotation = annotations[annotation_id]
        if annotation.get("track_id") not in original_track_ids:
            added_on_new_tracks += 1
            errors.add(
                "added_annotation_on_new_track",
                "anchor",
                annotation_id=annotation_id,
                track_id=annotation.get("track_id"),
            )
    return (
        {
            "original_annotations": len(original_annotations),
            "preserved": preserved,
            "missing": missing,
            "modified": modified,
            "modified_fields": dict(sorted(modified_fields.items())),
            "added_annotations": len(added_ids),
            "added_annotations_on_new_tracks": added_on_new_tracks,
        },
        added_ids,
    )


def _write_flag(
    writer: csv.DictWriter,
    counts: Counter[str],
    video_id: Any,
    track_id: Any,
    frame_t: Any,
    frame_t1: Any,
    flag_type: str,
    measured: float,
    threshold: float,
) -> None:
    counts[flag_type] += 1
    writer.writerow(
        {
            "video_id": video_id,
            "track_id": track_id,
            "frame_t": frame_t,
            "frame_t1": frame_t1,
            "flag_type": flag_type,
            "measured_value": "{:.10g}".format(measured),
            "threshold": "{:.10g}".format(threshold),
        }
    )


def _temporal_screen(
    observations: Mapping[Any, List[Dict[str, Any]]],
    csv_path: Path,
    thresholds: Mapping[str, float],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for track_id, track_observations in observations.items():
            ordered = sorted(track_observations, key=lambda item: (item["frame"], str(item["image_id"])))
            if len(ordered) < thresholds.get("min_track_length", 2):
                first = ordered[0] if ordered else {"video_id": "", "frame": "", "bbox": [0, 0, 0, 0]}
                _write_flag(writer, counts, first.get("video_id"), track_id, first.get("frame"), first.get("frame"), "track_length_anomaly", float(len(ordered)), float(thresholds.get("min_track_length", 2)))
            identical_transitions = 0
            identical_reported = False
            for left, right in zip(ordered, ordered[1:]):
                frame_t, frame_t1 = left["frame"], right["frame"]
                gap = frame_t1 - frame_t
                common = (left["video_id"], track_id, frame_t, frame_t1)
                if gap > thresholds["max_frame_gap"]:
                    _write_flag(writer, counts, *common, "temporal_gap", gap, thresholds["max_frame_gap"])
                if gap > thresholds["reappearance_gap"]:
                    _write_flag(
                        writer, counts, *common, "track_reappearance", gap, thresholds["reappearance_gap"]
                    )

                overlap = _iou(left["bbox"], right["bbox"])
                if overlap < thresholds["min_iou"]:
                    _write_flag(writer, counts, *common, "low_iou", overlap, thresholds["min_iou"])

                lbox, rbox = left["bbox"], right["bbox"]
                center_distance = math.hypot(
                    (rbox[0] + rbox[2] / 2.0) - (lbox[0] + lbox[2] / 2.0),
                    (rbox[1] + rbox[3] / 2.0) - (lbox[1] + lbox[3] / 2.0),
                )
                scale = math.sqrt((lbox[2] * lbox[3] + rbox[2] * rbox[3]) / 2.0)
                normalized_center = center_distance / scale if scale > 0 else math.inf
                if normalized_center > thresholds["max_center_displacement"]:
                    _write_flag(
                        writer,
                        counts,
                        *common,
                        "center_displacement",
                        normalized_center,
                        thresholds["max_center_displacement"],
                    )

                areas = (lbox[2] * lbox[3], rbox[2] * rbox[3])
                area_ratio = max(areas) / min(areas)
                if area_ratio > thresholds["max_area_ratio"]:
                    _write_flag(
                        writer, counts, *common, "box_size_jump", area_ratio, thresholds["max_area_ratio"]
                    )
                aspects = (lbox[2] / lbox[3], rbox[2] / rbox[3])
                aspect_ratio = max(aspects) / min(aspects)
                if aspect_ratio > thresholds["max_aspect_ratio_change"]:
                    _write_flag(
                        writer,
                        counts,
                        *common,
                        "aspect_ratio_jump",
                        aspect_ratio,
                        thresholds["max_aspect_ratio_change"],
                    )

                if left["bbox"] == right["bbox"]:
                    identical_transitions += 1
                    run_length = identical_transitions + 1
                    if run_length >= thresholds["identical_run_length"] and not identical_reported:
                        _write_flag(
                            writer,
                            counts,
                            *common,
                            "identical_bbox_run",
                            run_length,
                            thresholds["identical_run_length"],
                        )
                        identical_reported = True
                else:
                    identical_transitions = 0
                    identical_reported = False

                left_visibility = left.get("visibility")
                right_visibility = right.get("visibility")
                if _is_finite_number(left_visibility) and _is_finite_number(right_visibility):
                    visibility_jump = abs(right_visibility - left_visibility)
                    if visibility_jump > thresholds["max_visibility_jump"]:
                        _write_flag(
                            writer,
                            counts,
                            *common,
                            "visibility_jump",
                            visibility_jump,
                            thresholds["max_visibility_jump"],
                        )
    return counts


def run_qa(
    ctao_path: Path,
    report_path: Path,
    temporal_flags_path: Optional[Path] = None,
    *,
    original_tao_path: Optional[Path] = None,
    original_tao_base_path: Optional[Path] = None,
    reference_dir: Optional[Path] = None,
    category_split_path: Optional[Path] = None,
    validation_path: Optional[Path] = None,
    training_config_path: Optional[Path] = None,
    dataloader_source_path: Optional[Path] = None,
    known_filtered_anchors_path: Optional[Path] = None,
    profile: str = "protocol-aware",
    split: str = "unspecified",
    bbox_tolerance: float = 1e-4,
    area_tolerance: float = 1e-4,
    max_frame_gap: int = 1,
    reappearance_gap: int = 10,
    min_iou: float = 0.1,
    max_center_displacement: float = 2.0,
    max_area_ratio: float = 4.0,
    max_aspect_ratio_change: float = 3.0,
    identical_run_length: int = 5,
    max_visibility_jump: float = 0.75,
    min_track_length: int = 2,
    temporal_diagnostics: bool = False,
    verbose: bool = False,
    fail_on_warnings: bool = False,
    sha256: bool = False,
    max_error_examples: int = 20,
) -> Dict[str, Any]:
    """Validate inputs, write JSON/CSV outputs, and return the JSON report."""
    ctao_path = Path(ctao_path)
    report_path = Path(report_path)
    temporal_flags_path = Path(temporal_flags_path) if temporal_flags_path else None
    original_tao_path = Path(original_tao_path) if original_tao_path else None
    original_tao_base_path = Path(original_tao_base_path) if original_tao_base_path else None
    reference_dir = Path(reference_dir) if reference_dir else None
    category_split_path = Path(category_split_path) if category_split_path else None
    validation_path = Path(validation_path) if validation_path else None
    training_config_path = Path(training_config_path) if training_config_path else None
    dataloader_source_path = Path(dataloader_source_path) if dataloader_source_path else None
    known_filtered_anchors_path = Path(known_filtered_anchors_path) if known_filtered_anchors_path else None
    if profile not in ("protocol-aware", "strict-raw"):
        raise ValueError("profile must be protocol-aware or strict-raw")

    reference_selection = _discover_reference_paths(
        original_tao_path, original_tao_base_path, reference_dir, split
    )
    selected_original_path = Path(reference_selection["selected"]) if reference_selection["selected"] else None
    full_original_path = Path(reference_selection["full"]) if reference_selection["full"] else None
    if selected_original_path is not None:
        original_tao_path = selected_original_path
    if dataloader_source_path is None:
        candidate_source = Path.cwd() / "ovtrack/datasets/tao_dataset.py"
        dataloader_source_path = candidate_source if candidate_source.is_file() else None

    input_paths = [
        path
        for path in (
            ctao_path,
            original_tao_path,
            original_tao_base_path,
            full_original_path,
            category_split_path,
            validation_path,
            training_config_path,
            dataloader_source_path,
            known_filtered_anchors_path,
        )
        if path
    ]
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(str(path))
    input_resolved = {path.resolve() for path in input_paths}
    if temporal_diagnostics and temporal_flags_path is None:
        raise ValueError("temporal_flags_path is required when temporal diagnostics are enabled")
    for output in (report_path, temporal_flags_path):
        if output is None:
            continue
        if output.resolve() in input_resolved:
            raise ValueError("output path must not overwrite an input annotation: {}".format(output))
        output.parent.mkdir(parents=True, exist_ok=True)

    errors = ErrorLog(max_error_examples)
    warnings = DiagnosticLog(max_error_examples)
    infos = DiagnosticLog(max_error_examples)
    diagnostics = DiagnosticLog(max_error_examples)
    data = _load_json(ctao_path)
    for field in REQUIRED_TOP_LEVEL:
        if field not in data:
            errors.add("missing_top_level_field", "schema", field=field)

    sections = {field: _records(data, field, errors) for field in REQUIRED_TOP_LEVEL}
    indexes = {field: _make_index(sections[field], field, errors) for field in REQUIRED_TOP_LEVEL}

    nonfinite = _count_nonfinite(data)
    errors.add_count(
        "nonfinite_numeric",
        "schema",
        nonfinite,
        {"section": "entire JSON document"},
    )

    videos = indexes["videos"]
    images = indexes["images"]
    categories = indexes["categories"]
    tracks = indexes["tracks"]
    annotations = indexes["annotations"]

    original_data: Optional[Dict[str, Any]] = None
    original_sections: Optional[Dict[str, List[Dict[str, Any]]]] = None
    original_indexes: Optional[Dict[str, Dict[Any, Mapping[str, Any]]]] = None
    full_original_data: Optional[Dict[str, Any]] = None
    full_original_indexes: Optional[Dict[str, Dict[Any, Mapping[str, Any]]]] = None
    if original_tao_path is not None:
        original_data = _load_json(original_tao_path)
        original_sections = {
            field: _records(original_data, field, errors) for field in REQUIRED_TOP_LEVEL
        }
        original_indexes = {
            field: _make_index(original_sections[field], "original_" + field, errors)
            for field in REQUIRED_TOP_LEVEL
        }
        if full_original_path is not None and full_original_path.resolve() != original_tao_path.resolve():
            full_original_data = _load_json(full_original_path)
            full_sections = {
                field: _records(full_original_data, field, errors) for field in REQUIRED_TOP_LEVEL
            }
            full_original_indexes = {
                field: _make_index(full_sections[field], "full_original_" + field, errors)
                for field in REQUIRED_TOP_LEVEL
            }
        else:
            full_original_data = original_data
            full_original_indexes = original_indexes

    reference_semantic: Dict[Tuple[Any, ...], Mapping[str, Any]] = {}
    reference_images: Dict[Any, Mapping[str, Any]] = {}
    if original_indexes is not None:
        reference_images.update(original_indexes["images"])
    if full_original_indexes is not None:
        fallback_image_count = sum(
            annotation.get("image_id") not in reference_images
            for annotation in (original_indexes or {}).get("annotations", {}).values()
        )
        if fallback_image_count and verbose:
            infos.add("reference_image_fallback_to_full", count=fallback_image_count)
        fallback_images = dict(full_original_indexes["images"])
        fallback_images.update(reference_images)
        reference_images = fallback_images
    if original_indexes is not None:
        reference_semantic = _semantic_index(
            original_indexes["annotations"].values(), reference_images, errors
        )

    known_filtered = _load_known_filtered_anchors(known_filtered_anchors_path, errors)
    training_protocol: Dict[str, Any] = {
        "checked": False,
        "reason": "not requested for this split",
    }
    if split == "base-only":
        training_protocol = _inspect_training_config(
            training_config_path,
            dataloader_source_path,
            ctao_path,
            data,
            errors,
        )

    bbox_issues: List[Dict[str, Any]] = []

    for video in sections["videos"]:
        _validate_dimensions(video, "videos", errors)

    video_frame_keys: Set[Tuple[Any, Any]] = set()
    for image in sections["images"]:
        image_id = image.get("id")
        _validate_dimensions(image, "images", errors)
        video_id = image.get("video_id")
        if video_id not in videos:
            errors.add("image_video_reference_missing", "reference", image_id=image_id, video_id=video_id)
        elif image.get("video") is not None and videos[video_id].get("name") is not None:
            if image.get("video") != videos[video_id].get("name"):
                errors.add(
                    "image_video_name_mismatch",
                    "reference",
                    image_id=image_id,
                    video_id=video_id,
                )
        frame = image.get("frame_index")
        if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
            errors.add("invalid_frame_index", "schema", image_id=image_id, value=frame)
        elif video_id in videos:
            key = (video_id, frame)
            if key in video_frame_keys:
                errors.add("duplicate_video_frame", "duplicate", video_id=video_id, frame_index=frame)
            else:
                video_frame_keys.add(key)
            video = videos[video_id]
            length = video.get("length", video.get("num_frames"))
            if _is_finite_number(length) and (frame < 0 or frame >= length):
                errors.add(
                    "frame_outside_video_length",
                    "reference",
                    image_id=image_id,
                    frame_index=frame,
                    video_length=length,
                )
        if "frame_id" in image:
            frame_id = image.get("frame_id")
            if not isinstance(frame_id, int) or isinstance(frame_id, bool) or frame_id < 0:
                errors.add("invalid_frame_id", "schema", image_id=image_id, value=frame_id)

    for track in sections["tracks"]:
        track_id = track.get("id")
        if track.get("video_id") not in videos:
            errors.add(
                "track_video_reference_missing",
                "reference",
                track_id=track_id,
                video_id=track.get("video_id"),
            )
        if track.get("category_id") not in categories:
            errors.add(
                "track_category_reference_missing",
                "reference",
                track_id=track_id,
                category_id=track.get("category_id"),
            )

    seen_track_images: Set[Tuple[Any, Any]] = set()
    seen_track_frames: Set[Tuple[Any, Any]] = set()
    seen_annotation_signatures: Set[Tuple[Any, ...]] = set()
    previous_frame: Dict[Any, int] = {}
    observed_track_video: Dict[Any, Any] = {}
    observed_track_category: Dict[Any, Any] = {}
    observations: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)

    # Validate references and collect observations.  Bbox severity is assigned
    # after semantic anchor matching so inherited defects can be reported as
    # warnings while newly added defects remain hard errors.
    bbox_records: List[Dict[str, Any]] = []
    for position, annotation in enumerate(sections["annotations"]):
        annotation_id = annotation.get("id")
        image_id = annotation.get("image_id")
        video_id = annotation.get("video_id")
        category_id = annotation.get("category_id")
        track_id = annotation.get("track_id")
        image = images.get(image_id)
        track = tracks.get(track_id)

        if image is None:
            errors.add(
                "annotation_image_reference_missing",
                "reference",
                annotation_id=annotation_id,
                image_id=image_id,
            )
        if video_id not in videos:
            errors.add(
                "annotation_video_reference_missing",
                "reference",
                annotation_id=annotation_id,
                video_id=video_id,
            )
        if category_id not in categories:
            errors.add(
                "annotation_category_reference_missing",
                "reference",
                annotation_id=annotation_id,
                category_id=category_id,
            )
        if track is None:
            errors.add(
                "annotation_track_reference_missing",
                "reference",
                annotation_id=annotation_id,
                track_id=track_id,
            )
        if image is not None and video_id != image.get("video_id"):
            errors.add(
                "annotation_image_video_mismatch",
                "reference",
                annotation_id=annotation_id,
                annotation_video_id=video_id,
                image_video_id=image.get("video_id"),
            )
        if track is not None:
            if video_id != track.get("video_id"):
                errors.add(
                    "annotation_track_video_mismatch",
                    "track_category",
                    annotation_id=annotation_id,
                    track_id=track_id,
                )
            if category_id != track.get("category_id"):
                errors.add(
                    "annotation_track_category_mismatch",
                    "track_category",
                    annotation_id=annotation_id,
                    track_id=track_id,
                )

        if track_id in observed_track_video and observed_track_video[track_id] != video_id:
            errors.add("track_video_changed", "track_category", track_id=track_id, annotation_id=annotation_id)
        else:
            observed_track_video.setdefault(track_id, video_id)
        if track_id in observed_track_category and observed_track_category[track_id] != category_id:
            errors.add("track_category_changed", "track_category", track_id=track_id, annotation_id=annotation_id)
        else:
            observed_track_category.setdefault(track_id, category_id)

        bbox = annotation.get("bbox")
        bbox_ok = _valid_bbox(bbox) and bbox[2] > 0 and bbox[3] > 0
        bbox_records.append({"annotation": annotation, "image": image, "valid_format": _valid_bbox(bbox)})

        track_image_key = (track_id, image_id)
        if track_image_key in seen_track_images:
            errors.add(
                "multiple_boxes_same_track_image",
                "duplicate",
                track_id=track_id,
                image_id=image_id,
            )
        else:
            seen_track_images.add(track_image_key)

        frame = image.get("frame_index") if image is not None else None
        if isinstance(frame, int) and not isinstance(frame, bool):
            track_frame_key = (track_id, frame)
            if track_frame_key in seen_track_frames:
                errors.add(
                    "duplicate_track_frame",
                    "duplicate",
                    track_id=track_id,
                    frame_index=frame,
                )
            else:
                seen_track_frames.add(track_frame_key)
            if track_id in previous_frame and frame <= previous_frame[track_id]:
                details = {
                    "track_id": track_id,
                    "previous_frame": previous_frame[track_id],
                    "frame_index": frame,
                    "annotation_position": position,
                }
                if profile == "strict-raw":
                    errors.add("track_frames_not_strictly_sorted", "track_category", **details)
                else:
                    infos.add("annotations_not_sorted_in_json", **details)
            previous_frame[track_id] = frame

        if _valid_bbox(bbox):
            signature = (image_id, video_id, track_id, category_id, tuple(bbox))
            if signature in seen_annotation_signatures:
                errors.add(
                    "duplicate_annotation",
                    "duplicate",
                    annotation_id=annotation_id,
                    image_id=image_id,
                    track_id=track_id,
                )
            else:
                seen_annotation_signatures.add(signature)

        if bbox_ok and isinstance(frame, int) and not isinstance(frame, bool) and image is not None:
            visibility = annotation.get("visibility", annotation.get("visibility_ratio"))
            observations[track_id].append(
                {
                    "video_id": video_id,
                    "image_id": image_id,
                    "frame": frame,
                    "bbox": list(bbox),
                    "visibility": visibility,
                }
            )

    current_semantic = _semantic_index(annotations.values(), images, errors)
    anchor_summary: Dict[str, Any] = {
        "original_annotations": None,
        "preserved": None,
        "missing": None,
        "modified": None,
        "modified_fields": None,
        "added_annotations": None,
        "added_annotations_on_new_tracks": None,
    }
    identity_summary = {
        "new_video_identities": None,
        "new_category_identities": None,
        "new_track_identities": None,
    }
    if original_indexes is not None:
        identity_reference = full_original_indexes or original_indexes
        identity_summary = _identity_checks(indexes, identity_reference, errors)
        if profile == "strict-raw":
            anchor_summary, _ = _anchor_checks(
                annotations,
                original_indexes["annotations"],
                images,
                reference_images,
                set(original_indexes["tracks"]),
                bbox_tolerance,
                errors,
            )
        else:
            preserved = missing = modified = filtered = 0
            modified_fields: Counter[str] = Counter()
            missing_examples: List[Dict[str, Any]] = []
            for key, original in reference_semantic.items():
                current = current_semantic.get(key)
                if current is None:
                    reason = known_filtered.get(key)
                    if reason:
                        filtered += 1
                        if profile == "strict-raw":
                            warnings.add("known_filtered_anchor", semantic_key=list(key), reason=reason)
                        elif verbose:
                            infos.add("known_filtered_anchor", semantic_key=list(key), reason=reason)
                    else:
                        missing += 1
                        # Sparse anchors may be filtered when constructing the
                        # continuous C-TAO subset. Keep the count for audit
                        # provenance, but do not emit a diagnostic in the
                        # protocol-aware profile.
                        if len(missing_examples) < max_error_examples:
                            missing_examples.append({"semantic_key": list(key)})
                    continue
                changed = {}
                if not _boxes_close(original.get("bbox"), current.get("bbox"), bbox_tolerance):
                    changed["bbox"] = {"original": original.get("bbox"), "ctao": current.get("bbox")}
                if original.get("video_id") != current.get("video_id"):
                    changed["video_id"] = True
                if original.get("track_id") != current.get("track_id"):
                    changed["track_id"] = True
                if original.get("category_id") != current.get("category_id"):
                    changed["category_id"] = True
                if changed:
                    modified += 1
                    modified_fields.update(changed.keys())
                    errors.add("original_anchor_modified", "anchor", semantic_key=list(key), changed=changed)
                else:
                    preserved += 1
                    if original.get("image_id") != current.get("image_id"):
                        infos.add("anchor_image_id_remapped", semantic_key=list(key), original_image_id=original.get("image_id"), ctao_image_id=current.get("image_id"))
                    if original.get("id") != current.get("id"):
                        infos.add("anchor_annotation_id_remapped", semantic_key=list(key), original_id=original.get("id"), ctao_id=current.get("id"))
            added_keys = set(current_semantic) - set(reference_semantic)
            added_annotations = sum(1 for annotation in annotations.values() if _semantic_key(annotation, images) in added_keys)
            added_on_new_tracks = sum(1 for annotation in annotations.values() if _semantic_key(annotation, images) in added_keys and annotation.get("track_id") not in set(identity_reference.get("tracks", {})))
            if added_on_new_tracks:
                errors.add_count("added_annotation_on_new_track", "anchor", added_on_new_tracks)
            anchor_summary = {
                "original_annotations": len(reference_semantic),
                "preserved": preserved,
                "missing": missing,
                "filtered": filtered,
                "modified": modified,
                "modified_fields": dict(sorted(modified_fields.items())),
                "added_annotations": added_annotations,
                "added_annotations_on_new_tracks": added_on_new_tracks,
                "matching_key": "(video_id, frame_index/frame_id, track_id, category_id)",
                "missing_examples": missing_examples,
            }
    # Assign bbox diagnostics after anchor classification.
    for record in bbox_records:
        annotation = record["annotation"]
        image = record["image"]
        bbox = annotation.get("bbox")
        key = _semantic_key(annotation, images)
        inherited = key in reference_semantic
        severity_error = profile == "strict-raw" or not inherited
        if not record["valid_format"]:
            if severity_error:
                errors.add("invalid_bbox_format", "bbox", annotation_id=annotation.get("id"), bbox=bbox)
            elif verbose:
                infos.add("inherited_invalid_bbox", annotation_id=annotation.get("id"), bbox=bbox)
            continue
        x, y, width, height = bbox
        # Match CocoVideoDataset._parse_ann_info: records that the active
        # dataloader discards cannot affect training and are intentionally
        # omitted from protocol-aware bbox diagnostics.
        if profile == "protocol-aware" and image is not None:
            image_width, image_height = image.get("width"), image.get("height")
            intersection = 0.0
            if _is_finite_number(image_width) and _is_finite_number(image_height):
                intersection = max(0.0, min(x + width, image_width) - max(x, 0.0)) * max(
                    0.0, min(y + height, image_height) - max(y, 0.0)
                )
            area_value = annotation.get("area", width * height)
            if (
                annotation.get("ignore", False)
                or (_is_number(area_value) and area_value <= 0)
                or width < 1
                or height < 1
                or intersection == 0
            ):
                continue
        if width <= 0 or height <= 0:
            if severity_error:
                errors.add("nonpositive_bbox", "bbox", annotation_id=annotation.get("id"), bbox=bbox)
            elif verbose:
                infos.add("inherited_invalid_bbox", annotation_id=annotation.get("id"), bbox=bbox)
        if image is not None and _is_finite_number(image.get("width")) and _is_finite_number(image.get("height")):
            overshoot = max(-x, -y, x + width - image["width"], y + height - image["height"], 0.0)
            if overshoot > bbox_tolerance:
                limit = max(8.0, 0.01 * max(float(image["width"]), float(image["height"])))
                details = {"annotation_id": annotation.get("id"), "bbox": bbox, "overshoot": overshoot, "limit": limit}
                if overshoot <= limit:
                    if profile == "strict-raw":
                        warnings.add("bbox_boundary_overshoot", **details)
                    elif verbose:
                        infos.add("bbox_boundary_overshoot_within_tolerance", **details)
                elif inherited:
                    if profile == "strict-raw":
                        warnings.add("inherited_bbox_boundary_overshoot", **details)
                    elif verbose:
                        infos.add("inherited_bbox_boundary_overshoot", **details)
                else:
                    errors.add("bbox_out_of_bounds", "bbox", **details)
        if "area" in annotation:
            area = annotation.get("area")
            expected = width * height
            if not _is_finite_number(area) or not math.isclose(area, expected, rel_tol=area_tolerance, abs_tol=bbox_tolerance):
                if severity_error:
                    errors.add("area_mismatch", "bbox", annotation_id=annotation.get("id"), area=area, expected=expected)
                elif verbose:
                    infos.add("inherited_area_mismatch", annotation_id=annotation.get("id"), area=area, expected=expected)

    leakage_summary: Dict[str, Any] = {
        "checked": False,
        "split": split,
        "base_annotations": None,
        "novel_annotations": None,
        "unclassified_annotations": None,
        "category_split": None,
    }
    if split == "base-only":
        leakage_summary["checked"] = True
        if category_split_path is None:
            errors.add(
                "category_split_missing",
                "leakage",
                message="base-only leakage cannot be checked without an authoritative category split",
            )
        else:
            base_ids, novel_ids, split_metadata = _load_category_split(category_split_path, errors)
            leakage_summary["category_split"] = split_metadata
            category_counts = Counter(annotation.get("category_id") for annotation in sections["annotations"])
            base_count = sum(category_counts[category_id] for category_id in base_ids)
            novel_count = sum(category_counts[category_id] for category_id in novel_ids)
            classified_ids = base_ids | novel_ids
            unclassified_count = sum(
                count for category_id, count in category_counts.items() if category_id not in classified_ids
            )
            leakage_summary.update(
                {
                    "base_annotations": base_count,
                    "novel_annotations": novel_count,
                    "unclassified_annotations": unclassified_count,
                }
            )
            if unclassified_count:
                if profile == "protocol-aware":
                    errors.add_count("unclassified_annotation_category", "leakage", unclassified_count)
                else:
                    warnings.add_count("unclassified_annotation_category", unclassified_count)
            if profile == "strict-raw":
                errors.add_count("novel_annotation_in_base_only", "leakage", novel_count, {"novel_annotations": novel_count})
            elif training_protocol.get("checked") and training_protocol.get("filter_logic_confirmed"):
                simulated = _simulate_training_filter(data, training_protocol, novel_ids)
                leakage_summary["training_simulation"] = simulated
                effective_novel = simulated["effective_possible_novel_annotations"]
                if effective_novel:
                    errors.add_count("effective_novel_annotation_in_training", "leakage", effective_novel, {"effective_novel_annotations": effective_novel})
                elif novel_count and verbose:
                    infos.add_count("raw_novel_filtered_by_training_protocol", novel_count)
            else:
                if novel_count:
                    errors.add_count("training_protocol_unresolved", "leakage", novel_count, {"raw_novel_annotations": novel_count})
                elif verbose:
                    infos.add("training_protocol_not_required_raw_novel_zero", reason=training_protocol.get("reason"))

    overlap_summary: Dict[str, Any] = {"checked": False, "overlapping_videos": None}
    if validation_path is not None:
        validation_data = _load_json(validation_path)
        validation_videos = _make_index(_records(validation_data, "videos", errors), "validation_videos", errors)
        current_video_names = {record.get("name") for record in videos.values() if record.get("name")}
        validation_video_names = {
            record.get("name") for record in validation_videos.values() if record.get("name")
        }
        overlap_ids = set(videos) & set(validation_videos)
        overlap_names = current_video_names & validation_video_names
        overlap_count = len(overlap_ids | overlap_names)
        overlap_summary = {
            "checked": True,
            "overlapping_videos": overlap_count,
            "id_examples": sorted(overlap_ids, key=str)[:20],
            "name_examples": sorted(overlap_names)[:20],
        }
        errors.add_count(
            "train_validation_video_overlap",
            "split_overlap",
            overlap_count,
            {"id_examples": overlap_summary["id_examples"], "name_examples": overlap_summary["name_examples"]},
        )

    thresholds = {
        "max_frame_gap": max_frame_gap,
        "reappearance_gap": reappearance_gap,
        "min_iou": min_iou,
        "max_center_displacement": max_center_displacement,
        "max_area_ratio": max_area_ratio,
        "max_aspect_ratio_change": max_aspect_ratio_change,
        "identical_run_length": identical_run_length,
        "max_visibility_jump": max_visibility_jump,
        "min_track_length": min_track_length,
    }
    warning_counts: Counter[str] = Counter()
    if temporal_diagnostics or profile == "strict-raw":
        if temporal_flags_path is None:
            raise ValueError("temporal_flags_path is required for strict-raw temporal diagnostics")
        warning_counts = _temporal_screen(observations, temporal_flags_path, thresholds)
    for kind, count in warning_counts.items():
        diagnostics.add_count("temporal_" + kind, count)
    hard_status = "PASS" if errors.total == 0 else "FAIL"
    if profile == "protocol-aware":
        status = "FAIL" if errors.total else "PASS"
    else:
        status = "FAIL" if errors.total else ("PASS WITH WARNINGS" if warnings.total else "PASS")
    training_protocol_report = dict(training_protocol)
    if isinstance(training_protocol_report.get("allowed_category_ids"), set):
        training_protocol_report["allowed_category_ids"] = sorted(
            training_protocol_report["allowed_category_ids"], key=str
        )

    report: Dict[str, Any] = {
        "tool": {"name": "validate_ctao_annotations", "version": TOOL_VERSION},
        "inputs": {
            "ctao": _file_metadata(ctao_path, sha256),
            "original_tao": _file_metadata(original_tao_path, sha256),
            "original_tao_full": _file_metadata(full_original_path, sha256),
            "original_tao_base": _file_metadata(original_tao_base_path, sha256),
            "selected_reference": _file_metadata(selected_original_path, sha256),
            "category_split": _file_metadata(category_split_path, sha256),
            "validation": _file_metadata(validation_path, sha256),
            "training_config": _file_metadata(training_config_path, sha256),
            "dataloader_source": _file_metadata(dataloader_source_path, sha256),
            "known_filtered_anchors": _file_metadata(known_filtered_anchors_path, sha256),
        },
        "outputs": {
            "report": str(report_path.resolve()),
            "temporal_flags": str(temporal_flags_path.resolve()) if temporal_flags_path else None,
        },
        "parameters": {
            "split": split,
            "profile": profile,
            "bbox_tolerance": bbox_tolerance,
            "area_tolerance": area_tolerance,
            "fail_on_warnings": fail_on_warnings,
            "temporal_diagnostics": temporal_diagnostics,
            "verbose": verbose,
            "sha256": sha256,
            "temporal_thresholds": thresholds,
        },
        "counts": {
            "videos": len(sections["videos"]),
            "images": len(sections["images"]),
            "annotations": len(sections["annotations"]),
            "tracks": len(sections["tracks"]),
            "categories": len(sections["categories"]),
            "distinct_annotated_tracks": len(observations),
            "original_annotations": anchor_summary["original_annotations"],
            "added_annotations": anchor_summary["added_annotations"],
        },
        "hard_errors": errors.as_dict(),
        "warnings": warnings.as_dict(),
        "infos": infos.as_dict(),
        "diagnostics": diagnostics.as_dict(),
        "error_summary": {
            "schema_errors": errors.group_counts.get("schema", 0),
            "reference_errors": errors.group_counts.get("reference", 0),
            "bbox_errors": errors.group_counts.get("bbox", 0),
            "duplicate_errors": errors.group_counts.get("duplicate", 0),
            "track_category_errors": errors.group_counts.get("track_category", 0),
            "anchor_errors": errors.group_counts.get("anchor", 0),
            "identity_errors": errors.group_counts.get("identity", 0),
            "leakage_errors": errors.group_counts.get("leakage", 0),
            "split_overlap_errors": errors.group_counts.get("split_overlap", 0),
        },
        "anchor_preservation": anchor_summary,
        "identity_additions": identity_summary,
        "base_novel_leakage": leakage_summary,
        "reference_selection": reference_selection,
        "training_protocol": training_protocol_report,
        "train_validation_overlap": overlap_summary,
        "temporal_diagnostics": {
            "total": sum(warning_counts.values()),
            "by_type": dict(sorted(warning_counts.items())),
        },
        # Backward-compatible key; temporal events are diagnostics, not warnings.
        "temporal_warnings": {
            "total": sum(warning_counts.values()),
            "by_type": dict(sorted(warning_counts.items())),
        },
        "hard_validation_status": hard_status,
        "status": status,
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only validation and temporal diagnostic screening for C-TAO annotations."
    )
    parser.add_argument("--ctao", required=True, type=Path, help="C-TAO COCO-video/TAO JSON to validate")
    parser.add_argument(
        "--original-tao", type=Path, help="sparse TAO JSON in the same category-ID space, for anchor checks"
    )
    parser.add_argument("--original-tao-base", type=Path, help="explicit sparse original TAO base-only reference")
    parser.add_argument("--reference-dir", type=Path, help="directory searched for full/base sparse references")
    parser.add_argument(
        "--category-split",
        type=Path,
        help="authoritative JSON with category frequency or explicit base/novel ID arrays",
    )
    parser.add_argument(
        "--validation", type=Path, help="optional validation JSON used to detect train/validation video overlap"
    )
    parser.add_argument(
        "--split",
        choices=("unspecified", "base-only", "base+novel", "validation"),
        default="unspecified",
        help="declared role of --ctao; base-only enables strict leakage checking",
    )
    parser.add_argument("--profile", choices=("protocol-aware", "strict-raw"), default="protocol-aware")
    parser.add_argument("--training-config", type=Path, help="training config used to reproduce effective filtering")
    parser.add_argument("--dataloader-source", type=Path, help="dataloader source used to verify filtering")
    parser.add_argument("--known-filtered-anchors", type=Path, help="JSON manifest of anchors removed by a documented rule")
    parser.add_argument("--report", required=True, type=Path, help="output JSON QA report")
    parser.add_argument("--temporal-flags", type=Path, help="output temporal diagnostic CSV (used with --temporal-diagnostics)")
    parser.add_argument("--temporal-diagnostics", action="store_true", help="compute and write temporal diagnostic events")
    parser.add_argument("--verbose", action="store_true", help="retain protocol INFO details such as tolerated bbox/fallback events")
    parser.add_argument("--bbox-tolerance", type=float, default=1e-4, help="absolute bbox/edge tolerance")
    parser.add_argument("--area-tolerance", type=float, default=1e-4, help="relative area tolerance")
    parser.add_argument("--max-frame-gap", type=int, default=1)
    parser.add_argument("--reappearance-gap", type=int, default=10)
    parser.add_argument("--min-iou", type=float, default=0.1)
    parser.add_argument("--max-center-displacement", type=float, default=2.0)
    parser.add_argument("--max-area-ratio", type=float, default=4.0)
    parser.add_argument("--max-aspect-ratio-change", type=float, default=3.0)
    parser.add_argument("--identical-run-length", type=int, default=5)
    parser.add_argument("--max-visibility-jump", type=float, default=0.75)
    parser.add_argument("--min-track-length", type=int, default=2)
    parser.add_argument("--fail-on-warnings", action="store_true", help="return nonzero when warnings exist")
    parser.add_argument("--sha256", action="store_true", help="include SHA256 digests for every input")
    parser.add_argument("--max-error-examples", type=int, default=20)
    return parser


def _validate_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if (args.temporal_diagnostics or args.profile == "strict-raw") and args.temporal_flags is None:
        parser.error("--temporal-flags is required for temporal diagnostics")
    if args.bbox_tolerance < 0 or args.area_tolerance < 0:
        parser.error("bbox and area tolerances must be nonnegative")
    if args.max_frame_gap < 1 or args.reappearance_gap < 1:
        parser.error("frame-gap thresholds must be positive")
    if not 0 <= args.min_iou <= 1:
        parser.error("--min-iou must be in [0, 1]")
    if args.max_center_displacement <= 0 or args.max_area_ratio < 1:
        parser.error("center displacement must be positive and area ratio must be >= 1")
    if args.max_aspect_ratio_change < 1 or args.identical_run_length < 2:
        parser.error("aspect ratio must be >= 1 and identical run length must be >= 2")
    if args.max_visibility_jump < 0 or args.max_error_examples < 0:
        parser.error("visibility threshold and max examples must be nonnegative")
    if args.min_track_length < 1:
        parser.error("--min-track-length must be positive")


def _print_summary(report: Mapping[str, Any]) -> None:
    groups = report["hard_errors"]["by_group"]
    def result(*names: str) -> str:
        return "FAIL" if sum(groups.get(name, 0) for name in names) else "PASS"
    print("C-TAO annotation QA")
    print("Schema and references: {}".format(result("schema", "reference", "split_overlap")))
    print("Bounding-box validity: {}".format(result("bbox")))
    print("Track/category consistency: {}".format(result("duplicate", "track_category", "identity")))
    print("Sparse-anchor preservation: {}".format(result("anchor")))
    print("Base-only training protocol: {}".format(result("leakage")))
    if report.get("parameters", {}).get("temporal_diagnostics"):
        print("Diagnostic events: {}".format(report.get("diagnostics", {}).get("total", 0)))
    print("Final status: {}".format(report["status"]))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_arguments(args, parser)
    try:
        report = run_qa(
            args.ctao,
            args.report,
            args.temporal_flags,
            original_tao_path=args.original_tao,
            original_tao_base_path=args.original_tao_base,
            reference_dir=args.reference_dir,
            category_split_path=args.category_split,
            validation_path=args.validation,
            split=args.split,
            profile=args.profile,
            training_config_path=args.training_config,
            dataloader_source_path=args.dataloader_source,
            known_filtered_anchors_path=args.known_filtered_anchors,
            bbox_tolerance=args.bbox_tolerance,
            area_tolerance=args.area_tolerance,
            max_frame_gap=args.max_frame_gap,
            reappearance_gap=args.reappearance_gap,
            min_iou=args.min_iou,
            max_center_displacement=args.max_center_displacement,
            max_area_ratio=args.max_area_ratio,
            max_aspect_ratio_change=args.max_aspect_ratio_change,
            identical_run_length=args.identical_run_length,
            max_visibility_jump=args.max_visibility_jump,
            min_track_length=args.min_track_length,
            temporal_diagnostics=args.temporal_diagnostics,
            verbose=args.verbose,
            fail_on_warnings=args.fail_on_warnings,
            sha256=args.sha256,
            max_error_examples=args.max_error_examples,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print("QA could not run: {}".format(error), file=sys.stderr)
        return 2
    _print_summary(report)
    if report["hard_errors"]["total"]:
        return 1
    if args.fail_on_warnings and report.get("warnings", {}).get("total", 0):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
