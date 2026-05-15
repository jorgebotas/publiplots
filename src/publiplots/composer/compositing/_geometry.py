"""Pure-Python mm→pt + align×clip math for PDF + SVG compositing.

No matplotlib, no pypdf, no cairosvg imports. ``lxml`` is imported
lazily inside :func:`_resolve_svg_units` (the only consumer). The
compositing pipelines in :mod:`pdf` and :mod:`svg` call these helpers
to translate canvas-space mm rects into output-space transforms.

PDF user space:
- 1 pt = 1/72 inch
- y-axis is BOTTOM-UP (origin at lower-left)
- A page's mediabox is in pt

SVG user space:
- viewBox in user units (matplotlib emits pt; Inkscape emits mm;
  Illustrator + browsers emit px)
- y-axis is TOP-DOWN (origin at upper-left)
- mm-per-user-unit is deduced from the canvas SVG's width+viewBox via
  :func:`_resolve_svg_units`.

publiplots Canvas coordinate system:
- mm-based; ``bbox_mm = (x_mm, y_mm, w_mm, h_mm)`` is the slot in
  canvas mm coordinates with ``y_mm`` measured from the BOTTOM
  (bottom-up to match PDF). The SVG transform helper inverts to
  top-down internally.
"""
from __future__ import annotations

import re
import warnings
from typing import Any, Tuple


MM2PT: float = 72.0 / 25.4

# Per-unit mm-per-user-unit factors when width/height is "<value><unit>".
# The "user unit" denominator is supplied by the viewBox dimension.
# px → 25.4/96 mm (CSS reference DPI).
# pt → 25.4/72 mm.
# pc → 12 pt → 12 × 25.4/72 mm.
# in → 25.4 mm.
# cm → 10 mm.
# mm → 1 mm.
_UNIT_TO_MM: dict = {
    "px": 25.4 / 96.0,
    "pt": 25.4 / 72.0,
    "pc": 12.0 * 25.4 / 72.0,
    "in": 25.4,
    "cm": 10.0,
    "mm": 1.0,
}
_RELATIVE_UNITS = {"em", "ex", "%", "vh", "vw", "vmin", "vmax", "rem", "ch"}
# Default mm-per-user-unit when width/height absent or unitless: assume
# user units == px @ 96 DPI (CSS reference).
_DEFAULT_MM_PER_PX: float = 25.4 / 96.0

_LENGTH_RE = re.compile(
    r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z%]*)\s*$"
)


def _parse_svg_length(text: str) -> Tuple[float, str]:
    """Parse a length attribute like ``"40mm"`` → ``(40.0, "mm")``.

    Returns ``(value, unit)`` where ``unit`` is the lowercase suffix
    (``""`` if unitless). Raises ``ValueError`` on unparseable input.
    """
    m = _LENGTH_RE.match(text or "")
    if not m:
        raise ValueError(f"unparseable SVG length {text!r}")
    return float(m.group(1)), m.group(2).lower()


def _resolve_svg_units(
    svg_root: Any,
) -> Tuple[float, float, float, float, float]:
    """Inspect viewBox + width/height to deduce SVG user-unit → mm.

    Parameters
    ----------
    svg_root
        An ``lxml.etree._Element`` representing the SVG root tag (or
        any element-like object exposing ``.get(name)``).

    Returns
    -------
    (vb_x, vb_y, vb_w, vb_h, mm_per_user_unit) : 5-tuple of float
        ``vb_x``/``vb_y`` are the viewBox origin (often 0/0 but
        non-zero for Inkscape/Illustrator-authored SVGs).
        ``vb_w``/``vb_h`` are viewBox dimensions in user units.
        ``mm_per_user_unit`` is the multiplier converting user units
        into mm. So ``user_units * mm_per_user_unit == mm``.

    Raises
    ------
    ComposerVectorError
        ``viewBox`` is missing or malformed; OR ``width``/``height``
        carries a relative unit (``em``/``%``/etc) that requires
        font/parent context to resolve.
    """
    # Avoid the import-time dep: ComposerVectorError is in this package.
    from publiplots.composer.exceptions import ComposerVectorError

    viewbox = svg_root.get("viewBox")
    if not viewbox:
        raise ComposerVectorError(
            "SVG has no viewBox attribute; the composer cannot deduce "
            "user-unit → mm mapping without one. Add a viewBox "
            "(`viewBox=\"min-x min-y width height\"`) to the root <svg> "
            "tag, or convert to PDF/PNG.",
        )
    parts = viewbox.replace(",", " ").split()
    if len(parts) != 4:
        raise ComposerVectorError(
            f"SVG viewBox is malformed ({viewbox!r}); expected "
            f"\"min-x min-y width height\".",
        )
    try:
        vb_x = float(parts[0])
        vb_y = float(parts[1])
        vb_w = float(parts[2])
        vb_h = float(parts[3])
    except ValueError as e:
        raise ComposerVectorError(
            f"SVG viewBox values are not all numeric ({viewbox!r}): {e}",
        ) from e
    if vb_w <= 0 or vb_h <= 0:
        raise ComposerVectorError(
            f"SVG viewBox has zero or negative width/height ({viewbox!r}).",
        )

    width_attr = svg_root.get("width")
    height_attr = svg_root.get("height")

    def _mm_per_uu_from(attr: str, vb_dim: float) -> "float | None":
        """Resolve mm-per-uu from a width/height attribute. Returns None
        if the attribute is absent. Raises ComposerVectorError on
        relative-unit input."""
        if attr is None:
            return None
        try:
            value, unit = _parse_svg_length(attr)
        except ValueError:
            return None
        if unit in _RELATIVE_UNITS:
            raise ComposerVectorError(
                f"width/height with relative units {unit!r} cannot be "
                f"resolved without font/parent context; use absolute "
                f"units (mm/pt/px/cm/in) or remove the attributes to "
                f"fall back to px @ 96 DPI.",
            )
        if unit == "":
            unit_mm = _DEFAULT_MM_PER_PX  # px @ 96
        else:
            unit_mm = _UNIT_TO_MM.get(unit)
            if unit_mm is None:
                # Unknown absolute unit — be conservative + treat as px.
                unit_mm = _DEFAULT_MM_PER_PX
        # (value * unit_mm) is the attribute's mm length; the viewBox
        # dimension supplies the user-unit count. mm-per-uu = mm / uu.
        return (value * unit_mm) / vb_dim

    w_mm_per_uu = _mm_per_uu_from(width_attr, vb_w)
    h_mm_per_uu = _mm_per_uu_from(height_attr, vb_h)

    if w_mm_per_uu is None and h_mm_per_uu is None:
        return vb_x, vb_y, vb_w, vb_h, _DEFAULT_MM_PER_PX

    if w_mm_per_uu is not None and h_mm_per_uu is not None:
        # Disagreement check: > 1% drift → warn, prefer width.
        if w_mm_per_uu > 0 and abs(w_mm_per_uu - h_mm_per_uu) / w_mm_per_uu > 0.01:
            warnings.warn(
                f"SVG width/height units imply different mm-per-user-unit "
                f"({w_mm_per_uu:.6g} vs {h_mm_per_uu:.6g}); preferring "
                f"width.",
                UserWarning,
                stacklevel=2,
            )
        return vb_x, vb_y, vb_w, vb_h, w_mm_per_uu

    # Exactly one of the two resolved.
    return vb_x, vb_y, vb_w, vb_h, (
        w_mm_per_uu if w_mm_per_uu is not None else h_mm_per_uu
    )


# Shared align validation set, reused by both PDF and SVG transforms.
_VALID_ALIGN = {
    "top-left", "top", "top-right",
    "left", "center", "right",
    "bottom-left", "bottom", "bottom-right",
}


def compute_svg_transform(
    slot_bbox_mm: Tuple[float, float, float, float],
    schematic_size_mm: Tuple[float, float],
    *,
    canvas_mm_per_user_unit: float,
    canvas_vb_origin: Tuple[float, float],
    align: str,
    clip: str,
) -> Tuple[float, float, float, float]:
    """Compute (sx, sy, tx_user, ty_user) for an SVG ``<g transform>``.

    Parallel to :func:`compute_pdf_transform` but in canvas SVG user-unit
    space and with **SVG y-axis TOP-DOWN** (vs PDF bottom-up).

    Parameters
    ----------
    slot_bbox_mm : (x_mm, y_mm_top, w_mm, h_mm)
        Slot in canvas mm coordinates with ``y_mm_top`` from the TOP
        of the SVG canvas (the orchestrator pre-converts the
        composer's bottom-up ``bbox_mm`` into top-down before
        calling). This keeps the geometry helper ignorant of figure
        dims.
    schematic_size_mm : (w_mm, h_mm)
        Schematic intrinsic mm size (from
        :func:`_resolve_svg_units` × viewBox dims, or pixels × dpi for
        rasters).
    canvas_mm_per_user_unit
        mm-per-user-unit factor for the canvas SVG (returned by
        :func:`_resolve_svg_units`).
    canvas_vb_origin : (vb_x, vb_y)
        Canvas viewBox origin (user units). Added to the translation
        so non-zero-origin canvas SVGs are handled correctly.
    align
        One of nine CSS-``object-position`` values. Ignored when
        ``clip='stretch'``.
    clip : {'fit', 'fill', 'stretch'}
        Aspect-ratio policy.

    Returns
    -------
    sx, sy : float
        Scale factors for the wrapper ``<g transform="scale(...)">``.
    tx_user, ty_user : float
        Translation in canvas SVG user units. The wrapper is
        ``<g transform="translate(tx, ty) scale(sx, sy)">``.

    Raises
    ------
    ValueError
        Invalid align/clip, or schematic with zero/negative dimension.
    """
    sch_w_mm, sch_h_mm = schematic_size_mm
    if sch_w_mm <= 0 or sch_h_mm <= 0:
        raise ValueError(
            f"schematic has zero or negative dimension: "
            f"({sch_w_mm}, {sch_h_mm}). Cannot compute transform."
        )
    if clip not in ("fit", "fill", "stretch"):
        raise ValueError(
            f"clip={clip!r} invalid. Expected 'fit', 'fill', or 'stretch'."
        )
    if clip != "stretch" and align not in _VALID_ALIGN:
        raise ValueError(
            f"align={align!r} invalid. Expected one of {sorted(_VALID_ALIGN)}."
        )

    if canvas_mm_per_user_unit <= 0:
        raise ValueError(
            f"canvas_mm_per_user_unit={canvas_mm_per_user_unit!r} must be "
            f"> 0."
        )

    # Convert slot mm rect → user-unit rect. The slot's user-unit
    # position uses canvas_vb_origin as its reference (the canvas
    # viewBox y-down anchor).
    slot_x_mm, slot_y_mm_top, slot_w_mm, slot_h_mm = slot_bbox_mm
    slot_x_uu = slot_x_mm / canvas_mm_per_user_unit
    slot_w_uu = slot_w_mm / canvas_mm_per_user_unit
    slot_h_uu = slot_h_mm / canvas_mm_per_user_unit

    # The schematic, scaled by `scale`, occupies (sch_w_uu, sch_h_uu) in
    # the canvas user-unit space.
    sch_w_uu_unscaled = sch_w_mm / canvas_mm_per_user_unit
    sch_h_uu_unscaled = sch_h_mm / canvas_mm_per_user_unit

    # Compute scale factor.
    if clip == "stretch":
        sx = slot_w_uu / sch_w_uu_unscaled
        sy = slot_h_uu / sch_h_uu_unscaled
        scaled_w_uu = slot_w_uu
        scaled_h_uu = slot_h_uu
    else:
        if clip == "fit":
            scale = min(slot_w_uu / sch_w_uu_unscaled,
                        slot_h_uu / sch_h_uu_unscaled)
        else:  # fill
            scale = max(slot_w_uu / sch_w_uu_unscaled,
                        slot_h_uu / sch_h_uu_unscaled)
        sx = sy = scale
        scaled_w_uu = sch_w_uu_unscaled * scale
        scaled_h_uu = sch_h_uu_unscaled * scale

    # IMPORTANT: this helper expects ``slot_y_mm`` to be the slot's
    # TOP-y in the SVG canvas (mm-from-top), NOT bottom-up like the
    # PDF helper. The orchestrator in :mod:`svg` does the
    # bottom-up→top-down conversion (``fig_h_mm - slot_y_bottom_mm -
    # slot_h_mm``) before calling us, since only the orchestrator
    # has the figure height in scope. This keeps the geometry helper
    # ignorant of figure dims.
    slot_y_uu_top = slot_y_mm_top / canvas_mm_per_user_unit

    # Compute alignment offsets in user units.
    slack_w_uu = slot_w_uu - scaled_w_uu
    slack_h_uu = slot_h_uu - scaled_h_uu

    # Horizontal: 'left'-words → 0, 'right'-words → full slack, else half.
    if clip == "stretch":
        ox = 0.0
        oy = 0.0
    else:
        if align in ("top-left", "left", "bottom-left"):
            ox = 0.0
        elif align in ("top-right", "right", "bottom-right"):
            ox = slack_w_uu
        else:  # 'top', 'bottom', 'center'
            ox = slack_w_uu / 2.0

        # Vertical: SVG y-axis is TOP-DOWN. 'top'-words → 0; 'bottom'-words
        # → full slack. (Opposite of PDF.)
        if align in ("top-left", "top", "top-right"):
            oy = 0.0
        elif align in ("bottom-left", "bottom", "bottom-right"):
            oy = slack_h_uu
        else:  # 'left', 'right', 'center'
            oy = slack_h_uu / 2.0

    vb_x_origin, vb_y_origin = canvas_vb_origin
    return sx, sy, slot_x_uu + ox + vb_x_origin, slot_y_uu_top + oy + vb_y_origin


def compute_pdf_transform(
    slot_bbox_mm: Tuple[float, float, float, float],
    schematic_size_pt: Tuple[float, float],
    *,
    align: str,
    clip: str,
) -> Tuple[float, float, float, float]:
    """Compute (sx, sy, tx_pt, ty_pt) for stamping a schematic into a slot.

    Parameters
    ----------
    slot_bbox_mm : (x_mm, y_mm, w_mm, h_mm)
        Slot in canvas mm coordinates. ``(x_mm, y_mm)`` is the BOTTOM-LEFT
        corner (matching PDF's bottom-up y).
    schematic_size_pt : (w_pt, h_pt)
        Schematic's intrinsic mediabox dimensions in pt.
    align : str
        One of nine CSS-``object-position`` values. Ignored when
        ``clip='stretch'``.
    clip : {'fit', 'fill', 'stretch'}
        Aspect-ratio policy.

    Returns
    -------
    sx, sy, tx_pt, ty_pt : float
        Scale factors and translation in PDF user pt. Apply via
        ``pypdf.Transformation().scale(sx, sy).translate(tx_pt, ty_pt)``.

    Raises
    ------
    ValueError
        If ``schematic_size_pt`` has any zero or negative dimension,
        ``clip`` is invalid, or ``align`` is invalid (when ``clip``
        != 'stretch'; for 'stretch' align is unused and unvalidated).
    """
    sch_w_pt, sch_h_pt = schematic_size_pt
    if sch_w_pt <= 0 or sch_h_pt <= 0:
        raise ValueError(
            f"schematic mediabox has zero or negative dimension: "
            f"({sch_w_pt}, {sch_h_pt}). Cannot compute transform."
        )
    if clip not in ("fit", "fill", "stretch"):
        raise ValueError(
            f"clip={clip!r} invalid. Expected 'fit', 'fill', or 'stretch'."
        )
    # For non-stretch clips, validate align defensively. PanelImage's
    # __post_init__ already validates, but this helper is callable
    # directly; reject typos rather than silently centering.
    _VALID_ALIGN = {
        "top-left", "top", "top-right",
        "left", "center", "right",
        "bottom-left", "bottom", "bottom-right",
    }
    if clip != "stretch" and align not in _VALID_ALIGN:
        raise ValueError(
            f"align={align!r} invalid. Expected one of {sorted(_VALID_ALIGN)}."
        )

    slot_x_mm, slot_y_mm, slot_w_mm, slot_h_mm = slot_bbox_mm
    slot_x_pt = slot_x_mm * MM2PT
    slot_y_pt = slot_y_mm * MM2PT
    slot_w_pt = slot_w_mm * MM2PT
    slot_h_pt = slot_h_mm * MM2PT

    if clip == "stretch":
        sx = slot_w_pt / sch_w_pt
        sy = slot_h_pt / sch_h_pt
        return sx, sy, slot_x_pt, slot_y_pt

    if clip == "fit":
        scale = min(slot_w_pt / sch_w_pt, slot_h_pt / sch_h_pt)
    else:  # fill
        scale = max(slot_w_pt / sch_w_pt, slot_h_pt / sch_h_pt)
    sx = sy = scale
    scaled_w_pt = sch_w_pt * scale
    scaled_h_pt = sch_h_pt * scale

    # Compute alignment offsets (slack = slot dim − scaled schematic dim).
    # For 'fit' slack is ≥ 0 in both axes; for 'fill' slack is ≤ 0 in one
    # axis (the schematic overflows on that axis).
    slack_w_pt = slot_w_pt - scaled_w_pt
    slack_h_pt = slot_h_pt - scaled_h_pt

    # Horizontal: 'left'-words → 0, 'right'-words → full slack, else half.
    if align in ("top-left", "left", "bottom-left"):
        ox = 0.0
    elif align in ("top-right", "right", "bottom-right"):
        ox = slack_w_pt
    else:  # 'top', 'bottom', 'center'
        ox = slack_w_pt / 2.0

    # Vertical: PDF y is BOTTOM-UP. 'bottom'-words → 0; 'top'-words → full slack.
    if align in ("bottom-left", "bottom", "bottom-right"):
        oy = 0.0
    elif align in ("top-left", "top", "top-right"):
        oy = slack_h_pt
    else:  # 'left', 'right', 'center'
        oy = slack_h_pt / 2.0

    return sx, sy, slot_x_pt + ox, slot_y_pt + oy
