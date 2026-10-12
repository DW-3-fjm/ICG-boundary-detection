"""Explicit LabelMe polygon conversion; source data and class semantics stay private."""

import base64
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from . import read_source_image, write_image

SPLITS = {"train", "val", "test"}


def _annotations(source_dir, exclude_labels=()):
    root = Path(source_dir).resolve()
    if not root.is_dir():
        raise ValueError("Source directory does not exist")
    sources = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".json")
    if not sources:
        raise ValueError("No LabelMe JSON annotations found")
    if exclude_labels:
        excluded = set(exclude_labels)
        sources = [p for p in sources if not any(
            shape.get('label') in excluded for shape in json.loads(p.read_text(encoding='utf-8-sig')).get('shapes', [])
        )]
        if not sources:
            raise ValueError("No annotations remain after excluding labels")
    return root, sources


def make_manifest_template(source_dir, output_csv, exclude_labels=()):
    """List annotations without guessing patient IDs or assigning data splits."""
    root, sources = _annotations(source_dir, exclude_labels)
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["annotation", "split", "group_id"])
        writer.writerows((p.relative_to(root).as_posix(), "", "") for p in sources)
    return len(sources)


def _read_sample(path):
    raw = path.read_bytes()
    annotation = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(annotation, dict) or not isinstance(annotation.get("shapes"), list):
        raise ValueError(f"Not a LabelMe annotation: {path.name}")
    stored_path = annotation.get("imagePath")
    external = path.parent / stored_path.replace("\\", "/") if stored_path else None
    if annotation.get("imageData"):
        data = base64.b64decode(annotation["imageData"], validate=True)
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Unreadable embedded image: {path.name}")
        if external is not None and external.is_file():
            reference = read_source_image(external)
            if not np.array_equal(image, reference):
                raise ValueError(f"Embedded and external images disagree: {path.name}")
    elif external is not None and external.is_file():
        image = read_source_image(external)
    else:
        raise ValueError(f"No embedded image and imagePath cannot be resolved: {path.name}")
    if image.dtype != np.uint8 or not (image.ndim == 2 or image.ndim == 3 and image.shape[2] == 3):
        raise ValueError(f"Expected an 8-bit grayscale or BGR image: {path.name}")
    height, width = image.shape[:2]
    if (annotation.get("imageHeight"), annotation.get("imageWidth")) != (height, width):
        raise ValueError(f"Image dimensions disagree with annotation: {path.name}")
    return annotation, image, hashlib.sha256(raw).hexdigest()


def _polygon_mask(annotation, foreground, background):
    height, width = annotation["imageHeight"], annotation["imageWidth"]
    mask = np.zeros((height, width), np.uint8)
    observed = set()
    for shape in annotation["shapes"]:
        label = shape.get("label")
        if label not in foreground | background:
            raise ValueError(f"Label {label!r} has no explicit foreground/background assignment")
        observed.add(label)
        if shape.get("shape_type", "polygon") != "polygon" or shape.get("flags", {}).get("hole", False):
            raise ValueError("Only filled polygons without holes are supported")
        points = np.asarray(shape.get("points"), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
            raise ValueError("Polygon must contain at least three finite (x,y) points")
        # LabelMe can place vertices on the outer image edge; tolerate floating roundoff.
        if (points < -1e-6).any() or (points > np.array([width, height]) + 1e-6).any():
            raise ValueError("Polygon coordinates are outside the image")
        pixels = np.rint(np.clip(points, 0, [width - 1, height - 1])).astype(np.int32)
        if len(np.unique(pixels, axis=0)) < 3 or cv2.contourArea(pixels) == 0:
            raise ValueError("Degenerate polygon after rasterization")
        if label in foreground:
            # Fill separately so overlapping foreground polygons form a union.
            cv2.fillPoly(mask, [pixels], 255)
    return mask, observed


def convert_labelme_dataset(source_dir, output_dir, manifest, foreground_labels, background_labels=(),
                            *, train_only=False, exclude_labels=()):
    """Convert all annotations using an explicit, complete patient-group split CSV.

    Foreground polygons are unioned; listed background polygons do not erase foreground.
    Every encountered label must be assigned explicitly. No random frame split is made.
    """
    root, sources = _annotations(source_dir, exclude_labels)
    output = Path(output_dir).resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("Output and source directories must be separate")
    if output.exists():
        raise FileExistsError("Output already exists; choose a new directory to preserve prior work")
    foreground, background = set(foreground_labels), set(background_labels)
    if not foreground or foreground & background or "" in foreground | background:
        raise ValueError("Foreground labels must be nonempty and disjoint from background labels")
    if set(exclude_labels) & (foreground | background):
        raise ValueError("Excluded labels cannot also be assigned to foreground/background")
    with Path(manifest).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"annotation", "split", "group_id"} <= set(reader.fieldnames or []):
            raise ValueError("Manifest columns must include annotation,split,group_id")
        rows = list(reader)
    mapping, groups = {}, {}
    for row in rows:
        name = row["annotation"].strip().replace("\\", "/")
        split, group = row["split"].strip(), row["group_id"].strip()
        if name in mapping or split not in SPLITS or not group:
            raise ValueError("Duplicate annotation, invalid split or empty group_id in manifest")
        if group in groups and groups[group] != split:
            raise ValueError("A patient group occurs in multiple splits")
        mapping[name] = split, group
        groups[group] = split
    expected = {p.relative_to(root).as_posix() for p in sources}
    if set(mapping) != expected:
        raise ValueError("Manifest must cover every source annotation exactly once")
    required_splits = {'train'} if train_only else SPLITS
    if {split for split, _ in mapping.values()} != required_splits:
        raise ValueError("Train-only conversion requires only train rows" if train_only else
                         "Provide nonempty train, val and test splits")

    plan, fingerprints, observed = [], {}, set()
    for index, source in enumerate(sources, 1):
        annotation, image, source_hash = _read_sample(source)
        mask, labels = _polygon_mask(annotation, foreground, background)
        observed.update(labels)
        name = source.relative_to(root).as_posix()
        split, group = mapping[name]
        digest = hashlib.sha256(str((image.shape, image.dtype)).encode() + image.tobytes()).hexdigest()
        if digest in fingerprints and fingerprints[digest] != split:
            raise ValueError("Exact duplicate image across splits")
        fingerprints[digest] = split
        plan.append({"annotation": name, "split": split, "group_id": group,
                     "filename": f"sample_{index:05d}.png", "source_sha256": source_hash,
                     "labels": sorted(labels), "foreground_pixels": int(np.count_nonzero(mask))})
    if not foreground <= observed:
        raise ValueError("Some requested foreground labels do not occur in the source annotations")

    # All data/label/group checks pass before creating output. Only one decoded image
    # and mask need to be live at once, including for full-resolution 4K frames.
    output.mkdir(parents=True)
    for source, row in zip(sources, plan):
        annotation, image, source_hash = _read_sample(source)
        if source_hash != row["source_sha256"]:
            raise RuntimeError("Source changed during conversion; discard the incomplete output")
        mask, _ = _polygon_mask(annotation, foreground, background)
        write_image(str(output / row["split"] / "images" / row["filename"]), image)
        write_image(str(output / row["split"] / "masks" / row["filename"]), mask)
    with (output / "groups.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "filename", "group_id"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(plan)
    report = {"samples": len(plan), "splits": {s: sum(row["split"] == s for row in plan) for s in sorted(SPLITS)},
              "foreground_labels": sorted(foreground), "background_labels": sorted(background),
              "excluded_labels": sorted(exclude_labels), "train_only": bool(train_only),
              "rasterization": "round/clamp vertices, fill polygon interiors and edges; union foreground",
              "source_mapping": plan}
    (output / "conversion.private.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {key: value for key, value in report.items() if key != "source_mapping"}
