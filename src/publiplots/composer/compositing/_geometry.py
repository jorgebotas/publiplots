"""Pure-Python mm→pt + align×clip math for PDF compositing.

No matplotlib, no pypdf, no cairosvg imports. The compositing pipeline
in :mod:`pdf` calls these helpers to translate canvas-space mm rects
into PDF user-space pt transforms.

PDF user space:
- 1 pt = 1/72 inch
- y-axis is BOTTOM-UP (origin at lower-left)
- A page's mediabox is in pt

publiplots Canvas coordinate system:
- mm-based, but the canvas's matplotlib Figure is rendered to a PDF
  whose mediabox is the canvas's ``figure_size_mm * MM2PT``. So a
  panel slot at ``bbox_mm = (x_mm, y_mm, w_mm, h_mm)`` maps directly
  to the PDF page's coordinates by multiplying through MM2PT.
- ``y_mm`` is the BOTTOM of the slot in the canvas (bottom-up to match PDF).
"""
from __future__ import annotations

from typing import Tuple


MM2PT: float = 72.0 / 25.4


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
