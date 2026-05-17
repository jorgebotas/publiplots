"""abc panel-label sequencing + style resolution.

PR 2 introduces auto-letter labels (``abc='upper'``, ``'lower'``,
``'A.'``, ``'a.'``, or an explicit list), with per-canvas and per-panel
``label_style`` overrides. The ultraplot ``loc=`` vocabulary is
adopted verbatim: ``'ul'``, ``'ur'``, ``'ll'``, ``'lr'``, ``'uc'``,
``'lc'``, ``'cl'``, ``'cr'``.

Pure-Python except for the matplotlib ``ax.text`` call in
:func:`render_label` — which is gated to "render only" and used by
``canvas.add_row`` after the figure is built.
"""

from string import ascii_lowercase, ascii_uppercase
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union


VALID_LOCS = frozenset({"ul", "ur", "ll", "lr", "uc", "lc", "cl", "cr"})

# Default label_style — keys present on every Canvas; preset overrides
# `size` (and may override `loc` in future).
DEFAULT_LABEL_STYLE: Dict[str, Any] = {
    "weight": "bold",
    "size": 9,                # overridden by preset's label_size_pt
    "family": None,           # falls back to rcParams['font.family']
    "loc": "ul",
    "pad_mm": (0.0, 0.0),
    "border": False,
    "bbox": False,
}


def _seq_letter(idx: int, base: str) -> str:
    """0→A, 1→B, …, 25→Z, 26→AA, 27→AB, … (using `base` alphabet)."""
    if idx < 0:
        raise ValueError(f"sequence index must be non-negative, got {idx}")
    out = ""
    n = idx
    while True:
        out = base[n % 26] + out
        n = n // 26 - 1
        if n < 0:
            return out


def resolve_labels(
    *,
    panel_labels: Sequence[Union[str, None, bool]],
    abc: Union[str, bool, Sequence[str]],
) -> List[Union[str, None, bool]]:
    """Resolve a list of panel-input labels into a list of render-time labels.

    Each input slot is one of:
      - ``None`` — auto-letter slot, resolved per ``abc``.
      - ``False`` — explicitly suppressed; output is ``False``.
      - ``str`` — verbatim; output is the same str. Consumes a slot in
        the auto-letter sequence (so the next ``None`` skips ahead).

    ``abc`` is one of:
      - ``'upper'`` → 'A','B','C',...
      - ``'lower'`` → 'a','b','c',...
      - ``'A.'`` → 'A.','B.','C.',...
      - ``'a.'`` → 'a.','b.','c.',...
      - ``False`` → all auto slots resolve to ``None`` (no label)
      - ``Sequence[str]`` → explicit list, must match the count of
        non-``False`` panels.
    """
    if abc is False:
        # All auto slots become None; explicit strs pass through; False stays.
        return [v if v is not None else None for v in panel_labels]

    if isinstance(abc, str):
        # Detect template form (suffix after letter).
        if abc in ("upper", "lower"):
            base = ascii_uppercase if abc == "upper" else ascii_lowercase
            suffix = ""
        elif (
            len(abc) >= 2
            and abc[0].isalpha()
            and not any(c.isalpha() for c in abc[1:])
        ):
            base = ascii_uppercase if abc[0].isupper() else ascii_lowercase
            suffix = abc[1:]
        else:
            raise ValueError(
                f"abc must be 'upper', 'lower', a template like 'A.' or 'a.', "
                f"False, or an explicit list; got {abc!r}"
            )
        out: List[Any] = []
        slot = 0
        for v in panel_labels:
            if v is False:
                out.append(False)
                # False does NOT consume a slot.
            elif v is None:
                out.append(_seq_letter(slot, base) + suffix)
                slot += 1
            else:
                # str — pass through, consumes a slot.
                out.append(v)
                slot += 1
        return out

    # abc is Sequence[str]
    try:
        abc_list = list(abc)
    except TypeError:
        raise ValueError(
            f"abc must be 'upper', 'lower', a template, False, or a list; "
            f"got {type(abc).__name__}"
        )
    n_consuming = sum(1 for v in panel_labels if v is not False)
    if len(abc_list) < n_consuming:
        raise ValueError(
            f"abc list has {len(abc_list)} entries but {n_consuming} panels "
            f"need labels (panels with label=False are skipped)"
        )
    out2: List[Any] = []
    slot = 0
    for v in panel_labels:
        if v is False:
            out2.append(False)
        elif v is None:
            out2.append(abc_list[slot])
            slot += 1
        else:
            out2.append(v)
            slot += 1
    return out2


def merge_label_style(
    canvas_style: Mapping[str, Any],
    panel_override: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Merge canvas-wide style with per-panel override; panel wins."""
    merged = dict(canvas_style)
    if panel_override:
        merged.update(panel_override)
    return merged


def render_label(
    ax,
    label_text: str,
    *,
    style: Mapping[str, Any],
) -> None:
    """Place a text artist on ``ax`` at the resolved ``loc`` with style.

    Coordinates are axes-fraction; the loc-codes map to the 8 anchors
    defined by ultraplot: ul, ur, ll, lr (corners) + uc, lc (top/bottom
    center) + cl, cr (left/right middle).

    The pad_mm offset shifts the label INWARD from the chosen anchor,
    converted from mm to axes-fraction via the figure's dpi and the
    panel's pixel bbox. The conversion is correct for the panel's
    rendered size at the time render_label runs (after the figure
    has been laid out by SubplotsAutoLayout). If the bbox is
    degenerate (zero-sized — possible before any draw), pad_mm
    silently becomes a no-op.
    """
    loc = style["loc"]
    if loc not in VALID_LOCS:
        raise ValueError(f"loc must be one of {sorted(VALID_LOCS)}, got {loc!r}")

    # Map loc codes → (x_frac, y_frac, ha, va).
    # Outward-grow contract (PR 6c): at each anchor the text glyph grows
    # AWAY from the data into the canvas-reserved decoration margin —
    # 'ul' sits ON the top spine and grows UPWARD (va='bottom'), 'll'
    # sits ON the bottom spine and grows DOWNWARD (va='top'), etc.
    loc_table = {
        "ul": (0.0, 1.0, "left",   "bottom"),
        "ur": (1.0, 1.0, "right",  "bottom"),
        "ll": (0.0, 0.0, "left",   "top"),
        "lr": (1.0, 0.0, "right",  "top"),
        "uc": (0.5, 1.0, "center", "bottom"),
        "lc": (0.5, 0.0, "center", "top"),
        "cl": (0.0, 0.5, "right",  "center"),
        "cr": (1.0, 0.5, "left",   "center"),
    }
    x, y, ha, va = loc_table[loc]

    # Apply pad_mm as inward shift in axes fraction. Bbox can be
    # zero-sized before first draw, so guard against /0.
    pad_x_mm, pad_y_mm = style.get("pad_mm", (0.0, 0.0))
    bbox = ax.get_window_extent()
    fig = ax.figure
    dpi = fig.dpi if fig.dpi > 0 else 100.0
    if bbox.width > 0:
        x_frac_per_mm = (dpi / 25.4) / bbox.width
    else:
        x_frac_per_mm = 0.0
    if bbox.height > 0:
        y_frac_per_mm = (dpi / 25.4) / bbox.height
    else:
        y_frac_per_mm = 0.0
    if ha == "left":
        x += pad_x_mm * x_frac_per_mm
    elif ha == "right":
        x -= pad_x_mm * x_frac_per_mm
    if va == "top":
        y -= pad_y_mm * y_frac_per_mm
    elif va == "bottom":
        y += pad_y_mm * y_frac_per_mm

    text_kwargs = dict(
        x=x, y=y, s=label_text,
        ha=ha, va=va,
        transform=ax.transAxes,
        fontweight=style["weight"],
        fontsize=style["size"],
    )
    if style["family"]:
        text_kwargs["fontfamily"] = style["family"]

    bbox_kwargs = None
    if style["bbox"]:
        bbox_kwargs = {"boxstyle": "square,pad=0.2", "fc": "white", "ec": "none"}
    if style["border"]:
        # Border = white outline around the text glyphs.
        import matplotlib.patheffects as path_effects
        text = ax.text(**text_kwargs)
        text.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground="white"),
            path_effects.Normal(),
        ])
        if bbox_kwargs:
            text.set_bbox(bbox_kwargs)
        return

    text = ax.text(**text_kwargs)
    if bbox_kwargs:
        text.set_bbox(bbox_kwargs)
