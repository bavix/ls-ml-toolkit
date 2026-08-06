#!/usr/bin/env python3
"""INT8 quantization for the mobile detector.

Why this exists: `optimize_onnx.py` never made the model smaller or faster. It removed unused
initializers (an exported model has none) and "duplicate" nodes keyed on their input and output
names (which are unique in ONNX, so nothing ever matched). Its own comment said so — "for now,
just copy the model with basic cleanup" — and the model that shipped to phones was a full float32
graph the size of the original, named `optimized.onnx`.

Quantization is where the actual win is. On a convolutional detector the arithmetic is dominated by
Conv, and INT8 Conv on XNNPACK is typically two to three times faster than float32 while the weights
take a quarter of the space.

The important part is the calibration, not the API call. Static quantization picks the scale for
every activation tensor by watching real data flow through the network, so the data it watches has
to be the data the phone will actually send: the same letterbox colour, the same aspect-preserving
fit, the same channel order and the same 0..1 scaling that
`AndroidLayoutDetectionService.preprocess` performs. Calibrating on anything else sets the ranges
for a distribution the model will never see, and the accuracy loss that follows gets blamed on
INT8 rather than on the mismatch.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

# Straight from AndroidLayoutDetectionService: the grey YOLO letterboxes with, and the channel
# order and scaling its preprocess produces. Changing either here without changing it there is
# how a quantized model silently loses accuracy.
LETTERBOX_GREY = (114, 114, 114)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_CALIBRATION_SAMPLES = 128

# Per-channel weight scales are emitted as a `DequantizeLinear` with an `axis` attribute, and that
# attribute only exists from opset 13. Quantizing an opset-11 graph per-channel therefore produces
# a model that writes without complaint and fails to load: "Unrecognized attribute: axis for
# operator DequantizeLinear". Detector exports in the wild are frequently opset 11, so the
# quantizer raises the graph rather than making the caller find out at runtime.
MIN_PER_CHANNEL_OPSET = 13


def _find_images(root: Path, limit: int, seed: int = 0) -> list[Path]:
    """A deterministic sample of the images under [root], newest-first ordering ignored.

    Sampled rather than exhaustive: calibration reads every image through the full graph, and a
    few hundred is already past the point where more changes the ranges. Seeded so two runs of the
    toolkit produce the same model — an unreproducible artifact is not one you can bisect.
    """
    found = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if len(found) <= limit:
        return found
    return random.Random(seed).sample(found, limit)


def letterbox(image, width: int, height: int):
    """Fit an image into width x height keeping its aspect ratio, padding with [LETTERBOX_GREY].

    Mirrors the blit in `AndroidLayoutDetectionService.preprocess`: scale by the smaller ratio,
    centre the result, fill the rest with grey.
    """
    from PIL import Image

    scale = min(width / image.width, height / image.height)
    scaled = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    canvas = Image.new("RGB", (width, height), LETTERBOX_GREY)
    canvas.paste(image.resize(scaled, Image.BILINEAR),
                 ((width - scaled[0]) // 2, (height - scaled[1]) // 2))
    return canvas


def _tensor_from(path: Path, width: int, height: int):
    """One NCHW float32 batch in 0..1, RGB planar — the exact shape the Android side builds."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as raw:
        canvas = letterbox(raw.convert("RGB"), width, height)
    planar = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return np.expand_dims(planar, 0)


def model_input_shape(model_path: Path) -> tuple[str, int, int]:
    """The model's own input name and spatial size, rather than an assumed 640."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    spec = session.get_inputs()[0]
    _, _, h, w = spec.shape
    height = int(h) if isinstance(h, int) else 640
    width = int(w) if isinstance(w, int) else 640
    return spec.name, width, height


def _ensure_opset(model_path: Path, minimum: int) -> tuple[Path, bool]:
    """Raise the graph's opset to [minimum] if it is below it.

    Returns the path to use and whether the graph is now new enough. Conversion is attempted, not
    assumed: `convert_version` refuses graphs containing operators whose semantics changed in ways
    it cannot rewrite, and a failure here is not fatal — it only costs per-channel scales.
    """
    import onnx

    model = onnx.load(str(model_path))
    current = max((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), default=0)
    if current >= minimum:
        return model_path, True

    logger.info("Model is opset %d; raising to %d for per-channel quantization", current, minimum)
    try:
        from onnx import version_converter

        converted = version_converter.convert_version(model, minimum)
    except Exception as error:  # noqa: BLE001 - any conversion failure means the same fallback
        logger.warning(
            "Could not raise opset %d to %d (%s). Quantizing per-tensor instead: still INT8, but "
            "one scale per weight tensor rather than per output channel, so a filter bank is "
            "represented by its widest channel and accuracy suffers a little.",
            current, minimum, error,
        )
        return model_path, False

    raised = model_path.parent / f"{model_path.stem}.opset{minimum}.onnx"
    onnx.save(converted, str(raised))
    return raised, True


# Where the arithmetic stops being convolution and starts being decoding. Everything after the
# last one of these is the detection head.
FEATURE_OPS = ("Conv", "ConvTranspose", "Gemm", "MatMul")


def head_nodes(model_path: Path) -> list[str]:
    """Names of the decode nodes after the last convolution — the ones to leave in float.

    Quantizing them is what turns a working detector into one that finds nothing, and the failure
    is silent: the model loads, runs at full speed, and returns boxes whose coordinates are right
    and whose confidences are all exactly zero.

    The cause is one scale over two different quantities. A YOLO head ends by concatenating box
    coordinates, which run to the input size, with class scores, which a Sigmoid has put in 0..1.
    QDQ gives that concatenated tensor a single scale, so with a 640-pixel input the quantization
    step is about 2.5 — and every score, including a confident 0.91, rounds to zero. Observed
    exactly: box columns came back 2.63..638 against the float model's 3.69..635, while the score
    columns came back 0.0..0.0 against 0.0..0.91.

    Excluding them costs nothing. The head is a handful of elementwise operations over an 8400x6
    tensor; the convolutions that stay quantized are where every millisecond is. Defined by
    position rather than by name, so it holds for any export rather than for one model's layer
    numbering.
    """
    import onnx

    model = onnx.load(str(model_path))
    nodes = list(model.graph.node)  # ONNX guarantees topological order
    last_feature = max(
        (i for i, n in enumerate(nodes) if n.op_type in FEATURE_OPS),
        default=None,
    )
    if last_feature is None:
        return []
    return [n.name for n in nodes[last_feature + 1:] if n.name]


def _reader_class():
    from onnxruntime.quantization import CalibrationDataReader

    class LetterboxCalibrationReader(CalibrationDataReader):
        """Feeds real photographs through the production preprocess, one at a time."""

        def __init__(self, images: list[Path], input_name: str, width: int, height: int):
            self._input_name = input_name
            self._width = width
            self._height = height
            self._images = images
            self._iter: Optional[Iterator[Path]] = None

        def get_next(self):
            if self._iter is None:
                self._iter = iter(self._images)
            path = next(self._iter, None)
            if path is None:
                return None
            return {self._input_name: _tensor_from(path, self._width, self._height)}

        def rewind(self):
            self._iter = None

    return LetterboxCalibrationReader


def quantize_model(
    input_path: str,
    output_path: str,
    calibration_dir: Optional[str] = None,
    samples: int = DEFAULT_CALIBRATION_SAMPLES,
) -> bool:
    """Quantize to INT8, statically when there are images to calibrate on and dynamically when not.

    The distinction matters more than it looks. Dynamic quantization stores INT8 weights but works
    out activation scales at runtime, and ONNX Runtime applies it to MatMul-shaped operators rather
    than to Conv — on a convolutional detector that yields a smaller file and almost no speedup.
    Static quantization is the one that makes Conv run in INT8, and it can only do that if it has
    seen representative data. So dynamic is the fallback, announced as such, not the default.
    """
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_dynamic,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    source = Path(input_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    # Shape inference and graph cleanup first. Skipping this is the most common reason static
    # quantization produces a model that is slower than the float one: without inferred shapes the
    # quantizer leaves whole subgraphs in float and inserts conversions around each of them.
    prepared = target.parent / f"{target.stem}.preprocessed.onnx"
    try:
        quant_pre_process(str(source), str(prepared), skip_symbolic_shape=False)
    except Exception as error:  # noqa: BLE001 - preprocessing is best-effort by design
        logger.warning("quant_pre_process failed (%s); quantizing the graph as-is", error)
        prepared = source

    prepared, per_channel = _ensure_opset(prepared, MIN_PER_CHANNEL_OPSET)

    images: list[Path] = []
    if calibration_dir:
        root = Path(calibration_dir)
        if root.is_dir():
            images = _find_images(root, samples)
        else:
            logger.warning("Calibration directory not found: %s", root)

    try:
        if images:
            input_name, width, height = model_input_shape(prepared)
            logger.info(
                "Static INT8: calibrating on %d images at %dx%d (input '%s')",
                len(images), width, height, input_name,
            )
            reader = _reader_class()(images, input_name, width, height)
            excluded = head_nodes(prepared)
            if excluded:
                logger.info("Leaving %d decode nodes in float (detection head)", len(excluded))
            quantize_static(
                str(prepared),
                str(target),
                calibration_data_reader=reader,
                quant_format=QuantFormat.QDQ,
                # Unsigned activations and signed weights is what the CPU and XNNPACK kernels are
                # written for; per-channel weight scales matter on Conv, where one scale for a
                # whole filter bank is dominated by its widest channel.
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                per_channel=per_channel,
                calibrate_method=CalibrationMethod.MinMax,
                nodes_to_exclude=excluded,
            )
        else:
            logger.warning(
                "No calibration images — falling back to DYNAMIC quantization. Expect a smaller "
                "file and little speedup: dynamic quantization does not reach Conv, which is where "
                "a detector spends its time. Pass --calibration-dir to get the real win."
            )
            quantize_dynamic(str(prepared), str(target), weight_type=QuantType.QInt8)
    finally:
        for scratch in target.parent.glob(f"{target.stem}.*.onnx"):
            if scratch != target:
                scratch.unlink(missing_ok=True)
        if prepared != source and prepared.exists():
            prepared.unlink(missing_ok=True)

    before = source.stat().st_size
    after = target.stat().st_size
    logger.info(
        "Quantized: %.2f MB -> %.2f MB (%.1f%% smaller)",
        before / 1024 / 1024, after / 1024 / 1024, (1 - after / before) * 100,
    )
    return True
