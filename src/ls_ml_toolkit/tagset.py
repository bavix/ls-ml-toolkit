#!/usr/bin/env python3
"""Build a single-class YOLO detection dataset from mixed sources.

Two sources, one output layout (`images/` + `labels/` per split, YOLO txt, class 0):

- **A parquet detection dataset** (HuggingFace-style: image bytes + per-object boxes). Boxes are
  taken as they are — the dataset already marks the whole object.
- **An existing multi-class YOLO dataset**, whose per-part boxes are merged into one box per
  image: the union of all parts, padded. This is how a major/minor parts dataset becomes a
  whole-tag dataset: the union of the price parts, with margin, is where the tag's readable core
  sits — which is exactly the region a reader wants cropped.

The class list collapses to one on purpose. A parts detector needs parts annotation, which is the
expensive fine work; a single-class detector needs one rough box, and public datasets that mark
whole objects become directly usable as training data instead of being calibration-only.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Margin around the union of part boxes, as a fraction of the union's own size. The parts are the
# digits; the tag extends past them, and a detector trained on digit-tight boxes and asked for
# whole tags would be trained on a systematically different target than the parquet source's.
DEFAULT_UNION_MARGIN = 0.25


def _write_label(path: Path, boxes: list[tuple[float, float, float, float]]) -> None:
    lines = [
        f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        for cx, cy, w, h in boxes
        if w > 0 and h > 0
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def merge_yolo_parts(
    images_dir: str,
    labels_dir: str,
    out_images: str,
    out_labels: str,
    margin: float = DEFAULT_UNION_MARGIN,
) -> int:
    """One union box per image from a multi-class parts dataset. Returns images written.

    Images with no label file are skipped entirely rather than copied as backgrounds: in the
    source dataset an absent file may mean "not annotated yet", and a shelf full of tags written
    out as a background example teaches the detector to find nothing — the exact defect found in
    this project's own v1 split.
    """
    src_images = Path(images_dir)
    src_labels = Path(labels_dir)
    dst_images = Path(out_images)
    dst_labels = Path(out_labels)
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    written = 0
    for image in sorted(p for p in src_images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        label = src_labels / (image.stem + ".txt")
        if not label.is_file():
            continue
        boxes = []
        for line in label.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            _, cx, cy, w, h = (float(v) for v in parts)
            boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        if not boxes:
            continue
        left = min(b[0] for b in boxes)
        top = min(b[1] for b in boxes)
        right = max(b[2] for b in boxes)
        bottom = max(b[3] for b in boxes)
        pad_x = (right - left) * margin
        pad_y = (bottom - top) * margin
        left = max(0.0, left - pad_x)
        top = max(0.0, top - pad_y)
        right = min(1.0, right + pad_x)
        bottom = min(1.0, bottom + pad_y)

        shutil.copy2(image, dst_images / image.name)
        _write_label(
            dst_labels / (image.stem + ".txt"),
            [((left + right) / 2, (top + bottom) / 2, right - left, bottom - top)],
        )
        written += 1

    logger.info("parts->tag: %d images from %s", written, src_images)
    return written


def export_parquet(
    parquet_paths: Sequence[str],
    out_images: str,
    out_labels: str,
    category: Optional[str] = None,
    box_order: str = "yxyx",
    prefix: str = "",
) -> int:
    """Decode a parquet detection dataset into YOLO files. Returns images written."""
    import io

    from PIL import Image

    from .crops import _iter_rows, _to_xyxy

    dst_images = Path(out_images)
    dst_labels = Path(out_labels)
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    written = 0
    for parquet in parquet_paths:
        for row in _iter_rows(Path(parquet)):
            objects = row.get("objects") or {}
            raw_boxes = objects.get("bbox") or []
            names = objects.get("category_name") or []
            payload = row.get("image") or {}
            raw = payload.get("bytes") if isinstance(payload, dict) else None
            if raw is None:
                continue
            boxes = []
            for index, box in enumerate(raw_boxes):
                if category and index < len(names) and names[index] != category:
                    continue
                x1, y1, x2, y2 = _to_xyxy(box, box_order)
                x1, x2 = sorted((max(0.0, x1), min(1.0, x2)))
                y1, y2 = sorted((max(0.0, y1), min(1.0, y2)))
                boxes.append(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
            # Rows with no boxes are kept as genuine negatives: here the absence IS the
            # annotation — the dataset says "no price tag in this photograph" — unlike a missing
            # label file, which says nothing.
            name = f"{prefix}{row.get('image_id', written)}"
            with Image.open(io.BytesIO(raw)) as source:
                source.convert("RGB").save(dst_images / f"{name}.jpg", "JPEG", quality=92)
            _write_label(dst_labels / f"{name}.txt", boxes)
            written += 1

    logger.info("parquet->tag: %d images", written)
    return written


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Build a single-class YOLO dataset from parquet detections and/or a "
                    "multi-class YOLO parts dataset (parts are union-merged into one box)",
    )
    parser.add_argument("--out", required=True, help="Output dataset root (train/val created under it)")
    parser.add_argument("--name", default="price_tag", help="The single class name for data.yaml")
    parser.add_argument("--parquet-train", nargs="*", default=[], help="Parquet file(s) for train")
    parser.add_argument("--parquet-val", nargs="*", default=[], help="Parquet file(s) for val")
    parser.add_argument("--box-order", default="yxyx", help="bbox order in the parquet (see lsml-crops)")
    parser.add_argument("--category", help="Only use parquet objects with this category_name")
    parser.add_argument("--parts-train", nargs=2, metavar=("IMAGES", "LABELS"), help="YOLO parts train split")
    parser.add_argument("--parts-val", nargs=2, metavar=("IMAGES", "LABELS"), help="YOLO parts val split")
    parser.add_argument("--union-margin", type=float, default=DEFAULT_UNION_MARGIN)
    args = parser.parse_args()

    out = Path(args.out)
    totals = {"train": 0, "val": 0}
    for split, parquets, parts in (
        ("train", args.parquet_train, args.parts_train),
        ("val", args.parquet_val, args.parts_val),
    ):
        images = out / split / "images"
        labels = out / split / "labels"
        if parquets:
            totals[split] += export_parquet(
                parquets, str(images), str(labels),
                category=args.category, box_order=args.box_order, prefix="off_",
            )
        if parts:
            totals[split] += merge_yolo_parts(
                parts[0], parts[1], str(images), str(labels), margin=args.union_margin,
            )

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: train/images\nval: val/images\n\nnc: 1\nnames: ['{args.name}']\n",
    )
    logger.info("dataset at %s: train=%d val=%d", out, totals["train"], totals["val"])


if __name__ == "__main__":
    main()
