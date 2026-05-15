"""canvas.align() resolver — pure-Python edge/mode/anchor logic.

Each align call records an :class:`_AlignmentRequest` on the canvas;
at finalization time, the resolver iterates the requests and computes
the per-panel mm shifts. Slot mm-rects are inviolate — if a shift would
push axes outside their slot, raise ComposerAlignmentError.

PR 3 ships ``mode='axes'`` (axes-bbox edges) primarily. ``mode='tight'``
is accepted but uses the same path as 'axes' until PR 4.5/PR 5 add
proper tightbbox measurement.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple


VALID_EDGES = frozenset({
    "left", "right", "top", "bottom",
    "center_x", "center_y", "baseline",
})
VALID_MODES = frozenset({"axes", "tight"})


@dataclass
class _AlignmentRequest:
    panels: Tuple[str, ...]
    edge: str
    mode: str
    anchor: Optional[str]


def _compute_target_mm(
    *,
    edge: str,
    panel_rects_mm: Dict[str, Tuple[float, float, float, float]],
    panels: Sequence[str],
    anchor: Optional[str],
) -> float:
    """Determine the target mm coord for the shared edge."""
    if anchor is not None:
        rect = panel_rects_mm[anchor]
    else:
        # Default convention: leftmost wins for left/center_x;
        # rightmost wins for right; topmost for top/center_y;
        # bottommost for bottom/baseline.
        if edge == "right":
            rect = max((panel_rects_mm[p] for p in panels),
                       key=lambda r: r[0] + r[2])
        elif edge in ("left", "center_x"):
            rect = min((panel_rects_mm[p] for p in panels),
                       key=lambda r: r[0])
        elif edge in ("top", "center_y"):
            rect = max((panel_rects_mm[p] for p in panels),
                       key=lambda r: r[1] + r[3])
        else:  # bottom, baseline
            rect = min((panel_rects_mm[p] for p in panels),
                       key=lambda r: r[1])
    x, y, w, h = rect
    if edge == "left":      return x
    if edge == "right":     return x + w
    if edge == "top":       return y + h
    if edge == "bottom":    return y
    if edge == "baseline":  return y      # PR 3 alias for bottom; refined in PR 4.5
    if edge == "center_x":  return x + w / 2
    if edge == "center_y":  return y + h / 2
    raise ValueError(f"unhandled edge {edge!r}")


def apply_alignments(
    *,
    requests: Sequence[_AlignmentRequest],
    panel_rects_mm: Dict[str, Tuple[float, float, float, float]],
    slot_rects_mm: Dict[str, Tuple[float, float, float, float]],
) -> Dict[str, Tuple[float, float, float, float]]:
    """Apply each alignment request, returning the updated panel rects.

    `slot_rects_mm` are the inviolate mm bounds within which each panel
    can be shifted. If a request would push a panel outside its slot,
    raises :class:`publiplots.composer.exceptions.ComposerAlignmentError`.
    """
    from publiplots.composer.exceptions import ComposerAlignmentError

    rects = dict(panel_rects_mm)
    for req in requests:
        target = _compute_target_mm(
            edge=req.edge,
            panel_rects_mm=rects,
            panels=req.panels,
            anchor=req.anchor,
        )
        for p in req.panels:
            x, y, w, h = rects[p]
            sx, sy, sw, sh = slot_rects_mm[p]
            if req.edge in ("left", "right", "center_x"):
                if req.edge == "left":
                    new_x = target
                elif req.edge == "right":
                    new_x = target - w
                else:  # center_x
                    new_x = target - w / 2
                new_y = y
                if new_x < sx - 1e-6 or new_x + w > sx + sw + 1e-6:
                    raise ComposerAlignmentError(
                        f"alignment shift would push panel {p!r} outside its slot "
                        f"(slot x=[{sx:.2f},{sx+sw:.2f}], requested x=[{new_x:.2f},{new_x+w:.2f}])",
                        panels=tuple(req.panels),
                        edge=req.edge,
                    )
            else:  # top/bottom/center_y/baseline
                new_x = x
                if req.edge == "top":
                    new_y = target - h
                elif req.edge in ("bottom", "baseline"):
                    new_y = target
                else:  # center_y
                    new_y = target - h / 2
                if new_y < sy - 1e-6 or new_y + h > sy + sh + 1e-6:
                    raise ComposerAlignmentError(
                        f"alignment shift would push panel {p!r} outside its slot "
                        f"(slot y=[{sy:.2f},{sy+sh:.2f}], requested y=[{new_y:.2f},{new_y+h:.2f}])",
                        panels=tuple(req.panels),
                        edge=req.edge,
                    )
            rects[p] = (new_x, new_y, w, h)

    return rects
