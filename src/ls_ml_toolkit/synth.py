#!/usr/bin/env python3
"""Synthesizes labelled digit crops — printed and handwritten, mixed — for training a detector
whose job is reading digits inside an already-localized price field, not finding the field itself.

Two glyph sources, both unlimited in volume, neither requiring a human to draw a single box:

- **Printed**: rendered from real font outlines (DejaVu Sans, bundled with matplotlib — a
  dependency this toolkit already carries transitively for plotting, declared explicitly here so
  the font ships wherever this runs, including a bare Linux worker with no system fonts installed).
- **Handwritten**: real handwritten strokes from MNIST (60,000 digits, 10 writers' worth of
  variation is an undersell — thousands of writers). Not generated, not approximated — actual
  human handwriting, recoloured and composited.

Why this is worth building at all, stated plainly: the project's own past attempt at a handwritten
generator (`PriceTagImageGenerator`) never rendered a pixel — every "scenario" was a filename
string. This one produces real images with real per-digit YOLO labels, because the label is
generated alongside the pixels rather than guessed at afterward.

The backgrounds are synthetic on purpose, not real photographs. Pasting glyphs onto real tag crops
risks pasting them over or beside real, unlabelled digits already printed on the tag — exactly the
class of defect this project measured directly this session: eleven images in `dataset/v1`
silently taught a detector that a shelf full of price tags contains none, because their real
content went unboxed. A background whose only content is what this generator itself placed and
labelled cannot have that defect. Style (colour, a soft gradient, faint noise) is sampled from real
crops so the backgrounds still look like a price tag's own paper or plastic, not a blank studio card.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

# The classes a digit detector needs: ten digits plus the two separator shapes a price prints.
# Comma and period are visually near-identical at this scale and print interchangeably depending
# on locale, so they share one class rather than asking a detector to distinguish a distinction
# that carries no information at read time — PatternEngine normalises both to '.' regardless.
CLASS_NAMES = [str(d) for d in range(10)] + ["decimal_sep", "dash"]
DECIMAL_SEP_CLASS = 10
DASH_CLASS = 11

DEFAULT_CANVAS_SIZE = (384, 128)  # width, height — a price field's own aspect, not a square
DEFAULT_COUNT = 2000
DEFAULT_HANDWRITTEN_FRACTION = 0.35


@dataclass
class DigitPlacement:
    character: str
    cls: int
    left: int
    top: int
    width: int
    height: int


def _dejavu_font_path(bold: bool = False) -> str:
    """Locates the font bundled with matplotlib rather than trusting the host's own fonts.

    A Nomad worker container has no system fonts at all; a developer's laptop has hundreds of the
    wrong ones. matplotlib ships DejaVu Sans as its own fallback face specifically so its plots
    render identically everywhere — the same property this generator needs.
    """
    import matplotlib

    root = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(f"matplotlib's bundled {name} not found at {path}")
    return str(path)


def _background_stats(style_images: Sequence, rng: random.Random):
    """Mean colour and a rough texture amplitude sampled from real crops, not the crops themselves.

    Only two numbers survive from each source image — enough to make synthetic backgrounds land in
    the same colour family as real tags (white, cream, yellow sale-sticker, dark chalkboard)
    without ever compositing onto their real, unlabelled content.
    """
    from PIL import Image
    import numpy as np

    if not style_images:
        return (245, 245, 245), 6.0
    img = Image.open(rng.choice(style_images)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    mean = tuple(float(v) for v in arr.reshape(-1, 3).mean(axis=0))
    noise = float(arr.reshape(-1, 3).std(axis=0).mean() * 0.15)
    return mean, max(2.0, min(noise, 18.0))


def _synthetic_background(size: tuple[int, int], mean_rgb, noise_amplitude: float, rng: random.Random):
    from PIL import Image, ImageFilter
    import numpy as np

    width, height = size
    base = np.array(mean_rgb, dtype=np.float32)
    # A faint diagonal gradient, the way print-shop paper and backlit price-tag plastic actually
    # vary across a few centimetres — a flat fill reads as obviously synthetic in a way nothing
    # else about the image does, and a detector can learn to key on that seam.
    gradient = np.linspace(-noise_amplitude, noise_amplitude, width, dtype=np.float32)
    canvas = np.tile(base, (height, width, 1))
    canvas[:, :, :] += gradient[None, :, None]
    canvas += rng.gauss(0, noise_amplitude)
    noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0, noise_amplitude * 0.4, (height, width, 3))
    canvas = np.clip(canvas + noise, 0, 255).astype("uint8")
    img = Image.fromarray(canvas, "RGB")
    return img.filter(ImageFilter.GaussianBlur(0.4))


def _price_string(rng: random.Random) -> str:
    """One plausible price, in one of the layouts `TextAmountPatterns`/`AdjacentKopecks` handle."""
    major = rng.randint(1, 99999)
    minor = f"{rng.randint(0, 99):02d}"
    sep = rng.choice([".", ",", "-", " "])
    if rng.random() < 0.15:
        return str(major)  # a whole-number tag, no kopecks — must exist in the training mix too
    return f"{major}{sep}{minor}"


def _mnist_glyphs(rng: random.Random, cache_dir: str):
    from torchvision.datasets import MNIST

    dataset = MNIST(cache_dir, train=True, download=True)
    by_digit: dict[str, list] = {str(d): [] for d in range(10)}
    for image, label in dataset:
        by_digit[str(label)].append(image)
    return by_digit


def _paste_printed(
    canvas, draw, x: int, y: int, char: str, font, ink,
) -> tuple[int, int, int]:
    bbox = draw.textbbox((x, y), char, font=font)
    draw.text((x, y), char, font=font, fill=ink)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return bbox[0], bbox[1], width, height


@dataclass
class _CharPlan:
    """One character's random choices, decided once and reused for both measuring and drawing.

    The first version of this generator tried to measure a trial layout, then redraw at a
    corrected height by resetting the RNG and letting the real draw "replay" the same random
    sequence. That relied on the measurement pass and the draw pass calling the RNG the same
    number of times in the same order — true the day it was written, one `if` branch away from
    silently false, and it WAS false in practice (the real draw's `x`/`y` placement draws two
    extra random numbers the measurement pass never made, shifting every glyph/angle choice after
    it by two — three separate attempts at tuning this "worked" by producing a wrong answer that
    merely looked plausible, until compared against measured fit rates and found no better than
    guessing). A plan removes the assumption entirely: every random choice is made once, stored
    here, and both measuring and drawing are pure functions of the plan plus a height — no
    coincidence of call order to maintain.
    """

    character: str
    glyph: object | None  # a PIL Image, for a handwritten digit; None otherwise
    rotation: float
    spacing_after: int


def _build_plan(price: str, handwritten: bool, mnist_by_digit, rng: random.Random) -> list[_CharPlan]:
    plan = []
    for char in price:
        glyph = rng.choice(mnist_by_digit[char]) if handwritten and char.isdigit() else None
        rotation = rng.uniform(-12, 12) if glyph is not None else 0.0
        spacing = rng.randint(1, 8)  # scaled by the real digit_height at measure/draw time
        plan.append(_CharPlan(char, glyph, rotation, spacing))
    return plan


def _handwritten_size(glyph, target_height: int, angle: float) -> tuple[int, int]:
    """The (width, height) [_draw_handwritten] would produce for this glyph, without compositing.

    MNIST's raw 28x28 field pads every digit to roughly a square regardless of the stroke's own
    shape, so scaling the untouched image by height alone made a '1' occupy the same width as an
    '8'. Cropping to the actual ink first (`getbbox` finds it directly: MNIST is white strokes on a
    black field) makes width track the glyph's real shape, the way a font's advance width already
    does.
    """
    import math

    bbox = glyph.getbbox()
    ink_only = glyph.crop(bbox) if bbox else glyph
    scale = target_height / ink_only.height
    width = max(1, int(ink_only.width * scale))
    rad = math.radians(angle)
    rotated_w = int(abs(width * math.cos(rad)) + abs(target_height * math.sin(rad)))
    rotated_h = int(abs(width * math.sin(rad)) + abs(target_height * math.cos(rad)))
    return rotated_w, rotated_h


def _draw_handwritten(canvas, x: int, y: int, entry: _CharPlan, target_height: int, ink):
    from PIL import Image, ImageOps

    bbox = entry.glyph.getbbox()
    ink_only = entry.glyph.crop(bbox) if bbox else entry.glyph
    scale = target_height / ink_only.height
    resized = ink_only.resize((max(1, int(ink_only.width * scale)), target_height), Image.LANCZOS)
    rotated = resized.rotate(entry.rotation, expand=True, fillcolor=0)
    tinted = ImageOps.colorize(rotated, black=(0, 0, 0, 0), white=ink).convert("RGBA")
    tinted.putalpha(rotated.point(lambda p: p))
    canvas.paste(tinted, (x, y), tinted)
    return x, y, tinted.width, tinted.height


def _plan_width(plan: list[_CharPlan], digit_height: int, font) -> int:
    """Total width `plan` would occupy at `digit_height` — pure, no drawing, no randomness."""
    from PIL import ImageDraw, Image

    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    spacing_scale = digit_height / 10
    total = 0
    for entry in plan:
        if entry.character == " ":
            total += digit_height // 4
            continue
        if entry.glyph is not None:
            width, _ = _handwritten_size(entry.glyph, digit_height, entry.rotation)
        else:
            bbox = probe.textbbox((0, 0), entry.character, font=font)
            width = bbox[2] - bbox[0]
        total += width + max(1, int(entry.spacing_after * spacing_scale / 8))
    return total


def generate(
    output_dir: str,
    count: int = DEFAULT_COUNT,
    handwritten_fraction: float = DEFAULT_HANDWRITTEN_FRACTION,
    canvas_size: tuple[int, int] = DEFAULT_CANVAS_SIZE,
    style_source_dir: Optional[str] = None,
    mnist_cache_dir: str = "/tmp/lsml-mnist",
    seed: int = 0,
) -> int:
    """Writes `count` synthetic images with YOLO labels to `output_dir/images` and `.../labels`.

    Roughly `handwritten_fraction` of PRICES are rendered fully handwritten and the rest fully
    printed — not mixed within one price, because a real tag is written by one hand or one printer,
    never both. Returns the number of images written.
    """
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    out = Path(output_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    style_images = (
        [p for p in Path(style_source_dir).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if style_source_dir
        else []
    )
    mnist_by_digit = _mnist_glyphs(rng, mnist_cache_dir) if handwritten_fraction > 0 else {}
    printed_font = _dejavu_font_path(bold=False)
    printed_font_bold = _dejavu_font_path(bold=True)

    written = 0
    written_handwritten = 0
    for i in range(count):
        handwritten = rng.random() < handwritten_fraction
        mean_rgb, noise = _background_stats(style_images, rng)
        canvas = _synthetic_background(canvas_size, mean_rgb, noise, rng).convert("RGBA")
        draw = ImageDraw.Draw(canvas)
        # Ink darker than the background by a fixed margin reads as text at any background
        # brightness — a fixed dark grey would vanish against this generator's own dark-background
        # draws (chalkboard-style tags exist in the real distribution too).
        brightness = sum(mean_rgb) / 3
        ink = (20, 20, 20) if brightness > 128 else (235, 235, 235)

        price = _price_string(rng)
        from PIL import ImageFont

        use_bold = rng.random() < 0.3
        font_path = printed_font_bold if use_bold else printed_font
        plan = _build_plan(price, handwritten, mnist_by_digit, rng)

        # `x` decided before sizing, not after: the available width for the string is what's left
        # of the canvas past its own starting offset, and computing the budget as a fixed
        # `canvas_size[0] - margin` while a random 4-64px start offset was added on top of it
        # afterward is exactly why the first attempt at this still overflowed 87% of the time even
        # once the width measurement itself became exact — the budget and the thing consuming it
        # disagreed about where zero was.
        x = rng.randint(4, canvas_size[0] // 6)
        right_margin = 4
        usable_width = canvas_size[0] - x - right_margin

        # Sized to the string, not drawn from a height range independent of it. A real tag's
        # printer picks the type size that fits the field; a fixed height range sized for the
        # shortest prices this generator produces overflowed the canvas on most draws once prices
        # ran six to eight characters. `_plan_width` is pure — no randomness, no drawing — so it can
        # be called at a trial height and then again at the corrected height with no risk of the
        # two disagreeing about what they're measuring.
        max_height = int(canvas_size[1] * 0.7)
        min_height = canvas_size[1] // 4
        probe_font = ImageFont.truetype(font_path, max_height)
        width_at_max = _plan_width(plan, max_height, probe_font)
        digit_height = max_height if width_at_max <= usable_width else max_height * usable_width // width_at_max
        digit_height = max(min_height, min(max_height, digit_height))

        font = ImageFont.truetype(font_path, digit_height)

        y = (canvas_size[1] - digit_height) // 2 + rng.randint(-6, 6)
        spacing_scale = digit_height / 10
        placements: list[DigitPlacement] = []
        for entry in plan:
            if entry.character == " ":
                x += digit_height // 4
                continue
            cls = (
                int(entry.character)
                if entry.character.isdigit()
                else DECIMAL_SEP_CLASS
                if entry.character in ".,"
                else DASH_CLASS
            )
            if entry.glyph is not None:
                gx, gy, gw, gh = _draw_handwritten(canvas, x, y, entry, digit_height, ink)
            else:
                gx, gy, gw, gh = _paste_printed(canvas, draw, x, y, entry.character, font, ink)
            if gw > 0 and gh > 0:
                placements.append(DigitPlacement(entry.character, cls, gx, gy, gw, gh))
            x += gw + max(1, int(entry.spacing_after * spacing_scale / 8))

        if not placements or x > canvas_size[0]:
            continue  # ran off the canvas or produced nothing placeable; skip rather than mislabel

        name = f"synth_{'hw' if handwritten else 'pr'}_{i:06d}"
        canvas.convert("RGB").save(out / "images" / f"{name}.jpg", "JPEG", quality=95)
        lines = []
        for p in placements:
            cx = (p.left + p.width / 2) / canvas_size[0]
            cy = (p.top + p.height / 2) / canvas_size[1]
            w = p.width / canvas_size[0]
            h = p.height / canvas_size[1]
            if 0 < cx < 1 and 0 < cy < 1:
                lines.append(f"{p.cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (out / "labels" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        written += 1
        if handwritten:
            written_handwritten += 1

    names_yaml = ", ".join(f"'{n}'" for n in CLASS_NAMES)
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images\nval: images\n\n"
        f"nc: {len(CLASS_NAMES)}\nnames: [{names_yaml}]\n",
    )
    achieved_fraction = (written_handwritten / written * 100) if written else 0.0
    logger.info(
        "Wrote %d/%d synthetic digit images to %s (%.0f%% handwritten target, %.0f%% achieved — "
        "a handwritten price is more likely to overflow its own field than a printed one at the "
        "same character count, so the achieved share runs below target; request a larger --count "
        "rather than assume the target fraction was hit)",
        written, count, out, handwritten_fraction * 100, achieved_fraction,
    )
    return written


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Generate synthetic labelled digit images — printed and handwritten — "
                    "for a per-digit position detector",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--handwritten-fraction", type=float, default=DEFAULT_HANDWRITTEN_FRACTION)
    parser.add_argument(
        "--style-source-dir",
        help="Real price-tag images to sample background colour/texture statistics from "
             "(not pasted onto — see module docstring for why)",
    )
    parser.add_argument("--mnist-cache-dir", default="/tmp/lsml-mnist")
    parser.add_argument("--width", type=int, default=DEFAULT_CANVAS_SIZE[0])
    parser.add_argument("--height", type=int, default=DEFAULT_CANVAS_SIZE[1])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    generate(
        args.out, args.count, args.handwritten_fraction, (args.width, args.height),
        args.style_source_dir, args.mnist_cache_dir, args.seed,
    )


if __name__ == "__main__":
    main()
