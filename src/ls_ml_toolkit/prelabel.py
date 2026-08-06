#!/usr/bin/env python3
"""Pre-label images with the current detector, for import into Label Studio.

The bottleneck on a small detection dataset is rarely the images — it is the hours. Drawing two
boxes on an image from scratch takes a person perhaps fifteen seconds of aiming; nudging two boxes
that are already roughly right takes two or three. The model does not have to be good for this to
pay: it only has to be closer than an empty canvas, and a detector that already finds tags at
around 0.6 confidence is far closer than that.

So this runs the model you have over the images you have not annotated, and writes Label Studio
tasks whose `predictions` are already filled in. The annotator corrects rather than draws, and the
corrections feed the next model, which pre-labels better. That loop is the only realistic route from
a few hundred annotations to a few thousand.

The one thing that must be exactly right is the coordinate transform. The model sees a letterboxed
square; Label Studio wants percentages of the original image. Getting that wrong produces boxes
that are plausibly placed and systematically shifted, which is worse than no pre-labels at all —
an annotator will accept a near-miss far more readily than they will accept an empty frame.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

# The label names this project's Label Studio config uses; index is the model's class index.
DEFAULT_LABELS = ("price_major", "price_minor")
DEFAULT_CONFIDENCE = 0.25
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _decode(raw, letterbox_size: int, labels: Sequence[str], threshold: float):
    """YOLO head -> [(label_index, cx, cy, w, h, score)] in letterbox pixel space.

    The export used here emits (1, 4 + num_classes, num_anchors) with boxes already in pixels and
    scores already through a Sigmoid, so there is nothing to scale and nothing to activate — only
    a transpose, a threshold and non-maximum suppression.
    """
    import numpy as np

    array = raw[0] if raw.ndim == 3 else raw
    if array.shape[0] < array.shape[1]:
        array = array.T
    boxes = array[:, :4]
    scores = array[:, 4 : 4 + len(labels)]
    best = scores.argmax(axis=1)
    confidence = scores.max(axis=1)
    keep = confidence >= threshold
    return _suppress(boxes[keep], best[keep], confidence[keep])


def _suppress(boxes, classes, scores, iou_threshold: float = 0.45):
    """Greedy per-class NMS. The head emits one prediction per anchor, so a tag arrives many times."""
    import numpy as np

    kept = []
    for label in np.unique(classes):
        mask = classes == label
        candidates = boxes[mask]
        confidence = scores[mask]
        order = confidence.argsort()[::-1]
        x1 = candidates[:, 0] - candidates[:, 2] / 2
        y1 = candidates[:, 1] - candidates[:, 3] / 2
        x2 = candidates[:, 0] + candidates[:, 2] / 2
        y2 = candidates[:, 1] + candidates[:, 3] / 2
        areas = (x2 - x1) * (y2 - y1)
        while order.size:
            i = order[0]
            kept.append((int(label), *(float(v) for v in candidates[i]), float(confidence[i])))
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])
            overlap = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = overlap / (areas[i] + areas[rest] - overlap + 1e-9)
            order = rest[iou <= iou_threshold]
    return kept


def _to_percent(cx, cy, w, h, source_size, letterbox_size: int):
    """Undo the letterbox, then express the box as Label Studio's percentages of the original.

    Label Studio's x/y are the box's top-left corner, not its centre, and every value is a
    percentage of the original image's own dimensions — not of the square the model saw.
    """
    source_w, source_h = source_size
    scale = min(letterbox_size / source_w, letterbox_size / source_h)
    pad_x = (letterbox_size - source_w * scale) / 2
    pad_y = (letterbox_size - source_h * scale) / 2
    left = (cx - w / 2 - pad_x) / scale
    top = (cy - h / 2 - pad_y) / scale
    # float(), not numpy's float32: the values go straight into json.dumps, which does not know
    # how to encode a numpy scalar and fails at the very end of an otherwise complete run.
    return (
        float(max(0.0, min(100.0, left / source_w * 100))),
        float(max(0.0, min(100.0, top / source_h * 100))),
        float(max(0.0, min(100.0, w / scale / source_w * 100))),
        float(max(0.0, min(100.0, h / scale / source_h * 100))),
    )


def prelabel(
    model_path: str,
    images_dir: str,
    output_path: str,
    image_url_prefix: str = "",
    labels: Sequence[str] = DEFAULT_LABELS,
    threshold: float = DEFAULT_CONFIDENCE,
    model_version: str = "prelabel",
) -> int:
    """Write a Label Studio import file whose tasks carry the detector's boxes as predictions.

    [image_url_prefix] is prepended to each file name to form `data.image`, so the same run works
    for a local server (`/data/local-files/?d=...`) or for the S3 layout this project already uses
    (`s3://bucket/prefix/`). Returns the number of tasks written.
    """
    import numpy as np
    import onnxruntime as ort
    from PIL import Image

    from .quantize import letterbox

    root = Path(images_dir)
    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        logger.warning("No images under %s", root)
        return 0

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    spec = session.get_inputs()[0]
    size = int(spec.shape[2]) if isinstance(spec.shape[2], int) else 640

    tasks = []
    predicted = 0
    for path in images:
        with Image.open(path) as raw:
            rgb = raw.convert("RGB")
            source_size = rgb.size
            tensor = np.expand_dims(
                np.asarray(letterbox(rgb, size, size), dtype=np.float32).transpose(2, 0, 1) / 255.0,
                0,
            )
        output = session.run(None, {spec.name: tensor})[0]
        results = []
        for label_index, cx, cy, w, h, score in _decode(output, size, labels, threshold):
            if label_index >= len(labels):
                continue
            x, y, bw, bh = _to_percent(cx, cy, w, h, source_size, size)
            results.append(
                {
                    "original_width": source_size[0],
                    "original_height": source_size[1],
                    "image_rotation": 0,
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "score": score,
                    "value": {
                        "x": x, "y": y, "width": bw, "height": bh, "rotation": 0,
                        "rectanglelabels": [labels[label_index]],
                    },
                }
            )
        if results:
            predicted += 1
        # A task is written even when nothing was found. An image the model cannot read is exactly
        # the one worth a person's attention, and dropping it would quietly bias the dataset
        # towards what the model already handles.
        tasks.append(
            {
                "data": {"image": f"{image_url_prefix}{path.name}"},
                "predictions": [{"model_version": model_version, "result": results}],
            }
        )

    Path(output_path).write_text(json.dumps(tasks, ensure_ascii=False, indent=2))
    logger.info(
        "Wrote %d tasks to %s; %d carry at least one box (%d had none and are flagged for a human "
        "by being empty)",
        len(tasks), output_path, predicted, len(tasks) - predicted,
    )
    return len(tasks)


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Pre-label images with an ONNX detector for Label Studio import",
    )
    parser.add_argument("model", help="Path to the ONNX detector")
    parser.add_argument("images", help="Directory of images to pre-label")
    parser.add_argument("--output", "-o", default="prelabelled.json", help="Label Studio import file")
    parser.add_argument("--url-prefix", default="", help="Prepended to each file name for data.image")
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS), help="Class names by index")
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIDENCE)
    args = parser.parse_args()

    prelabel(
        args.model, args.images, args.output,
        image_url_prefix=args.url_prefix,
        labels=args.labels,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
