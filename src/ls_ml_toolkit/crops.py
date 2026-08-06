#!/usr/bin/env python3
"""Turn a coarse detection dataset into a fine annotation queue.

A public dataset usually marks the thing, not the parts of it. `openfoodfacts_price_tag_detection`
is 2,229 photographs with 1 category — `price-tag`, the whole label on the shelf. A pipeline that
reads the roubles and the kopecks separately cannot use those boxes as labels: they answer a
different question, and no crop can tell you where the decimal point is from a rectangle around the
whole tag.

What those boxes are worth is the crop itself. Annotating "which part of this is the major price"
inside a tight crop of one tag is a different job from finding that tag on a shelf photograph
first: nothing to search for, one obvious subject, two boxes that nearly fill the frame. The coarse
annotation somebody else already paid for buys the expensive half of the fine one.

So: read the parquet, cut one image per box, and — if a model is supplied — pre-label the parts
inside each crop so the work is correction rather than drawing. Thousands of tight crops out of a
dataset that, taken at face value, had nothing to offer a parts detector.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

# Room around the box, as a fraction of its own size. A tag cropped exactly to its annotation loses
# the edge the reader uses to tell a price from the text under it, and a box drawn tight by one
# annotator is drawn a few pixels loose by another.
DEFAULT_MARGIN = 0.08
DEFAULT_MIN_SIDE = 64

# Box conventions differ between datasets and none of them announce it in the data. Guessing is not
# a small error: reading `openfoodfacts_price_tag_detection` — which is (y_min, x_min, y_max, x_max)
# per its own README — as the more common x-first order produced tall narrow crops of product
# packaging instead of wide short crops of price tags, and every one of them looked like a
# plausible crop of something. Only the aspect ratios gave it away. So the caller states the order
# and the default is the one this toolkit was written for.
BOX_ORDERS = ("yxyx", "xyxy", "xywh", "cxcywh")
DEFAULT_BOX_ORDER = "yxyx"


def _iter_rows(parquet_path: Path, batch_size: int = 64):
    import pyarrow.parquet as pq

    reader = pq.ParquetFile(str(parquet_path))
    for batch in reader.iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def _to_xyxy(box, order: str):
    """Normalised box in [order] -> normalised (x1, y1, x2, y2)."""
    a, b, c, d = box
    if order == "yxyx":
        return b, a, d, c
    if order == "xyxy":
        return a, b, c, d
    if order == "xywh":
        return a, b, a + c, b + d
    if order == "cxcywh":
        return a - c / 2, b - d / 2, a + c / 2, b + d / 2
    raise ValueError(f"Unknown box order: {order}")


def _crop_box(size, box, margin: float, order: str = DEFAULT_BOX_ORDER):
    """Normalised box plus margin -> integer pixel box, clamped to the image."""
    width, height = size
    x1, y1, x2, y2 = _to_xyxy(box, order)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    pad_x = (x2 - x1) * margin
    pad_y = (y2 - y1) * margin
    return (
        int(max(0.0, x1 - pad_x) * width),
        int(max(0.0, y1 - pad_y) * height),
        int(min(1.0, x2 + pad_x) * width),
        int(min(1.0, y2 + pad_y) * height),
    )


def extract_crops(
    parquet_paths: Sequence[str],
    output_dir: str,
    category: Optional[str] = None,
    margin: float = DEFAULT_MARGIN,
    min_side: int = DEFAULT_MIN_SIDE,
    box_order: str = DEFAULT_BOX_ORDER,
) -> list[Path]:
    """Write one image per annotated object. Returns the paths written, in a stable order."""
    from PIL import Image

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    skipped_small = 0

    for parquet_path in parquet_paths:
        for row in _iter_rows(Path(parquet_path)):
            objects = row.get("objects") or {}
            boxes = objects.get("bbox") or []
            names = objects.get("category_name") or []
            payload = row.get("image") or {}
            raw = payload.get("bytes") if isinstance(payload, dict) else None
            if raw is None or not boxes:
                continue
            with Image.open(io.BytesIO(raw)) as source:
                image = source.convert("RGB")
                for index, box in enumerate(boxes):
                    if category and index < len(names) and names[index] != category:
                        continue
                    left, top, right, bottom = _crop_box(image.size, box, margin, box_order)
                    # Too small to annotate is too small to learn from: a thirty-pixel tag is a
                    # smudge, and a person asked to place two boxes inside one will place them
                    # somewhere, which is worse than not being asked.
                    if right - left < min_side or bottom - top < min_side:
                        skipped_small += 1
                        continue
                    crop = image.crop((left, top, right, bottom))
                    name = f"{row.get('image_id', 'img')}_{index}.jpg"
                    path = destination / name
                    crop.save(path, "JPEG", quality=92)
                    written.append(path)

    logger.info(
        "Wrote %d crops to %s (%d skipped as smaller than %dpx)",
        len(written), destination, skipped_small, min_side,
    )
    return written


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Cut per-object crops out of a HuggingFace-style detection parquet, and "
                    "optionally pre-label the parts inside them for Label Studio",
    )
    parser.add_argument("parquet", nargs="+", help="Parquet file(s) with image bytes and objects")
    parser.add_argument("--output", "-o", required=True, help="Directory to write crops into")
    parser.add_argument("--category", help="Only crop objects with this category_name")
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--min-side", type=int, default=DEFAULT_MIN_SIDE)
    parser.add_argument(
        "--box-order", choices=BOX_ORDERS, default=DEFAULT_BOX_ORDER,
        help="Coordinate order in the dataset's bbox field. Check the dataset card; reading one "
             "order as another yields crops that look plausible and are of the wrong thing.",
    )
    parser.add_argument(
        "--model", help="ONNX detector; supplied, the crops are pre-labelled for Label Studio",
    )
    parser.add_argument("--tasks", help="Where to write the Label Studio import file")
    parser.add_argument("--url-prefix", default="", help="Prepended to each crop name for data.image")
    args = parser.parse_args()

    crops = extract_crops(
        args.parquet, args.output, args.category, args.margin, args.min_side, args.box_order,
    )
    if args.model and crops:
        from .prelabel import prelabel

        prelabel(
            args.model,
            args.output,
            args.tasks or str(Path(args.output).parent / "crops_tasks.json"),
            image_url_prefix=args.url_prefix,
        )


if __name__ == "__main__":
    main()
