"""Figure→bytes helpers for ``canvas.embed_figure``.

The two pure helpers in this module render a matplotlib
:class:`~matplotlib.figure.Figure` to a PDF or SVG byte buffer with
the same byte-determinism contract that PR 5 / PR 6a's
``savefig_pdf`` and ``savefig_svg`` use for the canvas figure itself.

They are pure: no filesystem I/O, no global rcParam mutation outside
their localized ``plt.rc_context`` (SVG path), no mutation of the
input Figure beyond a single ``figure.canvas.draw()`` to settle the
``SubplotsAutoLayout`` reactor (architect-flagged determinism gotcha
for ``pp.subplots``-built figures).

Determinism contract:

- PDF: ``metadata={"CreationDate": None}`` to suppress matplotlib's
  auto-timestamp; pypdf re-stamps with the pinned ``_DEFAULT_CREATION_DATE``
  + ``_PRODUCER`` literals.
- SVG: ``plt.rc_context({"svg.hashsalt": _SVG_HASHSALT})`` to pin the
  ``<defs>`` element id suffixes; ``metadata={"Date": _DEFAULT_DATE}``
  to write the deterministic ``<dc:date>``.

PR 6c additions:

- :func:`extract_side_axes_bbox` returns the side figure's axes-data
  bbox in pt (BOTTOM-UP). Used by the ``embed_figure(anchor='axes')``
  compositing path to align the side figure's axes-data box with the
  slot rect rather than its outer mediabox.
- :func:`check_decoration_overflow` raises
  :class:`ComposerVectorError` if the side figure's decoration
  extents (mediabox − axes-data bbox per side) exceed the canvas's
  surrounding margin reservation, scaled by the post-transform
  scale factor.
"""
from __future__ import annotations

import io
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from publiplots.composer.compositing._constants import (
    _DEFAULT_CREATION_DATE,
    _DEFAULT_DATE,
    _PRODUCER,
    _SVG_HASHSALT,
)


# Conversion factor used in PR 6c overflow checks.
_PT2MM: float = 25.4 / 72.0


def _settle_subplots_auto_layout(figure: Figure) -> None:
    """Run the SubplotsAutoLayout reactor at most once.

    PR 6b architect-flagged determinism gotcha: ``pp.subplots`` attaches
    a reactor (``fig._publiplots_auto_layout``) that measures decoration
    tightbboxes at draw-time and resizes the figure mid-render. If the
    reactor runs in the middle of ``figure.savefig`` the output bytes can
    drift (font cache state etc). Settling the reactor BEFORE savefig
    pins the figure's geometry; subsequent renders are byte-deterministic.

    Idempotent: matplotlib's draw-cache makes repeated ``draw()`` calls
    cheap when nothing has changed.
    """
    figure.canvas.draw()


def render_figure_to_pdf_bytes(
    figure: Figure,
    *,
    metadata_creation_date: Optional[str] = None,
    **savefig_kwargs: Any,
) -> bytes:
    """Render ``figure`` to a deterministic PDF byte buffer.

    Parameters
    ----------
    figure
        The matplotlib Figure to render.
    metadata_creation_date
        PDF ``/CreationDate`` value. ``None`` (default) → use the
        pinned :data:`_constants._DEFAULT_CREATION_DATE` literal so the
        bytes are reproducible. Pass an explicit string to override.
    **savefig_kwargs
        Forwarded to ``figure.savefig`` (``dpi=`` etc).

    Returns
    -------
    bytes
        The PDF bytes — starts with ``%PDF-`` magic; metadata pinned.

    Notes
    -----
    Internally this calls ``figure.savefig(buf, format='pdf',
    metadata={"CreationDate": None})`` to suppress matplotlib's
    auto-timestamp, then opens the buffer with pypdf and re-stamps the
    pinned ``/CreationDate`` + ``/Producer`` metadata. Mirror the same
    pattern that :func:`savefig_pdf` uses for the canvas figure itself
    (so the embedded figure's bytes match the canvas's contract).
    """
    # Lazy import — pypdf is an optional [composer] dep; mirrors the
    # PR 5 install-hint pattern in compositing.pdf.
    try:
        import pypdf
        from pypdf.generic import NameObject, TextStringObject
    except ImportError as e:  # pragma: no cover — install hint
        from publiplots.composer.exceptions import ComposerVectorError
        raise ComposerVectorError(
            "pypdf is required for embed_figure → PDF compositing. "
            "Install with `pip install publiplots[composer]`.",
            source_error=str(e),
        ) from e

    # Settle the SubplotsAutoLayout reactor BEFORE savefig (architect
    # determinism gotcha).
    _settle_subplots_auto_layout(figure)

    # Stage 1: render to a PDF buffer with auto-timestamp suppressed.
    raw_buf = io.BytesIO()
    figure.savefig(
        raw_buf,
        format="pdf",
        metadata={"CreationDate": None},
        **savefig_kwargs,
    )
    raw_buf.seek(0)

    # Stage 2: re-stamp pinned metadata via pypdf clone_from (same
    # idiom as savefig_pdf).
    reader = pypdf.PdfReader(raw_buf)
    writer = pypdf.PdfWriter(clone_from=reader)
    cd = (metadata_creation_date if metadata_creation_date is not None
          else _DEFAULT_CREATION_DATE)
    writer.add_metadata({
        NameObject("/Producer"): TextStringObject(_PRODUCER),
        NameObject("/CreationDate"): TextStringObject(cd),
    })
    out_buf = io.BytesIO()
    writer.write(out_buf)
    return out_buf.getvalue()


def render_figure_to_svg_bytes(
    figure: Figure,
    *,
    metadata_date: Optional[str] = None,
    **savefig_kwargs: Any,
) -> bytes:
    """Render ``figure`` to a deterministic SVG byte buffer.

    Parameters
    ----------
    figure
        The matplotlib Figure to render.
    metadata_date
        SVG ``<dc:date>`` value. Three-state semantics (matches
        :func:`savefig_svg`):

        - ``None`` (default) → the pinned :data:`_constants._DEFAULT_DATE`
          literal so the byte stream is reproducible.
        - ``"omit"`` → matplotlib strips ``<dc:date>`` entirely.
        - any other string → write that literal value.
    **savefig_kwargs
        Forwarded to ``figure.savefig`` (``dpi=`` etc).

    Returns
    -------
    bytes
        SVG bytes — parseable by :func:`lxml.etree.fromstring`.
    """
    # Settle the SubplotsAutoLayout reactor BEFORE savefig.
    _settle_subplots_auto_layout(figure)

    if metadata_date is None:
        date_value: Optional[str] = _DEFAULT_DATE
    elif metadata_date == "omit":
        date_value = None
    else:
        date_value = metadata_date

    buf = io.BytesIO()
    with plt.rc_context({"svg.hashsalt": _SVG_HASHSALT}):
        figure.savefig(
            buf,
            format="svg",
            metadata={"Date": date_value},
            **savefig_kwargs,
        )
    return buf.getvalue()


# ---------------------------------------------------------------------------
# PR 6c: axes-data bbox extraction + decoration-overflow check
# ---------------------------------------------------------------------------

# Decoration-overflow tolerance for ``check_decoration_overflow``.
# PR 6c addendum (Task 12): the cap is now multiplicative — decorations
# may consume up to 80% of the corresponding canvas margin budget per
# side. The remaining 20% is breathing room: visually the side fig's
# decoration must NOT touch the next panel's axes-data box. The
# kitchen-sink 13.26 mm side-fig left decoration vs cell-2col's 15 mm
# gap (88%) crowded Panel A even though it geometrically fit; 80% caps
# that. Tighter than 80% risks false positives on otherwise reasonable
# layouts; looser admits visible crowding.
_DECORATION_OVERFLOW_BUDGET_FRACTION: float = 0.80


def extract_side_axes_bbox(
    figure: Figure,
) -> Tuple[float, float, float, float]:
    """Return the side figure's axes-data bbox in pt (BOTTOM-UP coords).

    Used by the ``canvas.embed_figure(anchor='axes')`` compositing
    path. The returned rect is the union of all axes-position rects
    in the figure (with twin-axes deduped), expressed in
    side-figure-mediabox-pt coordinates.

    Parameters
    ----------
    figure
        A matplotlib :class:`Figure`. CALLER is responsible for
        running :func:`_settle_subplots_auto_layout` (or equivalent
        ``figure.canvas.draw()``) BEFORE calling this helper —
        ``ax.get_position()`` returns pre-settle figure-fraction
        coords for ``pp.subplots``-built figures, which would yield
        the wrong bbox.

    Returns
    -------
    (left_pt, bottom_pt, w_pt, h_pt)
        The union axes-data rect in pt (PDF bottom-up convention).
        For an empty figure (``len(fig.axes) == 0``) the helper
        returns the full mediabox ``(0, 0, fig_w_pt, fig_h_pt)`` as a
        degenerate fallback.

    Notes
    -----
    Twin-axes deduplication uses **position-rect equality**, not
    ``ax.get_shared_x_axes()``. The latter incorrectly conflates
    ``subplots(nrows=2, sharex=True)`` (disjoint sibling rects, two
    distinct rows) with ``ax.twinx()`` (overlay; same position rect).
    Position-rect equality preserves the disjoint case (different
    rounded bounds → kept separately) AND collapses the overlay case
    (identical bounds → de-duped to one entry).
    """
    fig_w_in, fig_h_in = figure.get_size_inches()
    fig_w_pt = fig_w_in * 72.0
    fig_h_pt = fig_h_in * 72.0

    if not figure.axes:
        return (0.0, 0.0, fig_w_pt, fig_h_pt)

    # Position-rect dedup: round to 6 dp to absorb float noise; keys
    # in a set; iterate fig.axes in insertion order (matplotlib's
    # Z-order). Twin overlay → identical key → second axes ignored.
    seen_rects: set = set()
    rects: list = []
    for ax in figure.axes:
        bounds = ax.get_position().bounds  # (x0, y0, w, h) in fig-fraction
        rect_key = tuple(round(c, 6) for c in bounds)
        if rect_key in seen_rects:
            continue
        seen_rects.add(rect_key)
        rects.append(bounds)

    # Union over the de-duped rects.
    x0s = [r[0] for r in rects]
    y0s = [r[1] for r in rects]
    x1s = [r[0] + r[2] for r in rects]
    y1s = [r[1] + r[3] for r in rects]
    union_x0 = min(x0s)
    union_y0 = min(y0s)
    union_x1 = max(x1s)
    union_y1 = max(y1s)

    # fig-fraction → pt.
    left_pt = union_x0 * fig_w_pt
    bottom_pt = union_y0 * fig_h_pt
    w_pt = (union_x1 - union_x0) * fig_w_pt
    h_pt = (union_y1 - union_y0) * fig_h_pt
    return (left_pt, bottom_pt, w_pt, h_pt)


def check_decoration_overflow(
    figure: Figure,
    axes_bbox_pt: Tuple[float, float, float, float],
    decoration_budget_mm: Dict[str, float],
    panel_label: Any,
    *,
    slot_size_mm: Tuple[float, float],
) -> None:
    """Raise :class:`ComposerVectorError` if side-fig decorations
    overflow canvas reservation.

    Used by the ``canvas.embed_figure(anchor='axes')`` compositing
    path BEFORE the main render-and-stamp step, so users see the
    error at the failing panel rather than as a clipped output.

    Parameters
    ----------
    figure
        The side figure (already settled via
        :func:`_settle_subplots_auto_layout`).
    axes_bbox_pt
        ``(left_pt, bottom_pt, w_pt, h_pt)`` from
        :func:`extract_side_axes_bbox`.
    decoration_budget_mm
        ``{'left', 'right', 'top', 'bottom'}`` mm budgets returned by
        :meth:`Canvas._panel_decoration_budget_mm`.
    panel_label
        Panel label for the error message; ``None`` / ``False`` →
        ``'<unlabeled>'``.
    slot_size_mm
        ``(slot_w_mm, slot_h_mm)`` of the canvas slot the side fig is
        being mapped into. Used to compute the post-transform scale
        factor that scales side-fig decorations.

    Raises
    ------
    ComposerVectorError
        If any side's decoration extent exceeds
        ``_DECORATION_OVERFLOW_BUDGET_FRACTION`` (= 80%) of that
        side's canvas margin reservation. Message names the panel,
        side, mm overflow, and points at three remediation paths
        (``anchor='figure'``, shrink decoration, increase canvas
        margin) plus the PR 6d auto-expand hint.
    """
    from publiplots.composer.exceptions import ComposerVectorError

    fig_w_in, fig_h_in = figure.get_size_inches()
    fig_w_pt = fig_w_in * 72.0
    fig_h_pt = fig_h_in * 72.0
    axes_left_pt, axes_bottom_pt, axes_w_pt, axes_h_pt = axes_bbox_pt
    axes_right_pt = axes_left_pt + axes_w_pt
    axes_top_pt = axes_bottom_pt + axes_h_pt

    # Decoration extent on each side (in pt) = mediabox extent − axes
    # extent on that side. These are pre-scale; they are the "raw"
    # decoration thickness in side-figure mediabox coords.
    dec_left_pt = axes_left_pt
    dec_right_pt = fig_w_pt - axes_right_pt
    dec_bottom_pt = axes_bottom_pt
    dec_top_pt = fig_h_pt - axes_top_pt

    # Post-transform scale factor: the side fig's axes-data box
    # (axes_w_pt × axes_h_pt) is scaled to map to the slot
    # (slot_size_mm); decorations scale by the same factor (uniform
    # within each axis).
    slot_w_mm, slot_h_mm = slot_size_mm
    if axes_w_pt <= 0 or axes_h_pt <= 0:
        # Degenerate side fig: no axes — skip the overflow check.
        # Caller's compose path will produce a vacuous output but not
        # an overflow error.
        return
    slot_w_pt = slot_w_mm / _PT2MM
    slot_h_pt = slot_h_mm / _PT2MM
    scale_x = slot_w_pt / axes_w_pt  # pt/pt → unitless
    scale_y = slot_h_pt / axes_h_pt

    # Scaled decoration extents in mm. Horizontal decorations use
    # scale_x; vertical use scale_y. (The transform applied at compose
    # time is anisotropic when the side fig's axes aspect differs
    # from the slot — non-uniform scale is fine for axes-data-anchored
    # mapping; decorations land at proportionally scaled positions.)
    scaled_left_mm = dec_left_pt * scale_x * _PT2MM
    scaled_right_mm = dec_right_pt * scale_x * _PT2MM
    scaled_top_mm = dec_top_pt * scale_y * _PT2MM
    scaled_bottom_mm = dec_bottom_pt * scale_y * _PT2MM

    label_str = panel_label
    if label_str is None or label_str is False:
        label_str = "<unlabeled>"

    sides = [
        ("left", scaled_left_mm, decoration_budget_mm.get("left", 0.0),
         "ylabel"),
        ("right", scaled_right_mm, decoration_budget_mm.get("right", 0.0),
         "right-side legend / right axis"),
        ("top", scaled_top_mm, decoration_budget_mm.get("top", 0.0),
         "title / suptitle"),
        ("bottom", scaled_bottom_mm, decoration_budget_mm.get("bottom", 0.0),
         "xlabel"),
    ]

    for side, decoration_mm, budget_mm, hint_kind in sides:
        # PR 6c addendum (Task 12): cap decoration at 80% of budget so
        # the side fig's xlabel/ylabel doesn't visually crowd the
        # neighbouring panel's axes-data box.
        allowed_mm = _DECORATION_OVERFLOW_BUDGET_FRACTION * budget_mm
        if decoration_mm > allowed_mm:
            raise ComposerVectorError(
                f"PanelImage {label_str!r} embed_figure(anchor='axes'): "
                f"side figure decoration overflow on the {side!r} side — "
                f"{decoration_mm:.2f} mm exceeds "
                f"{int(_DECORATION_OVERFLOW_BUDGET_FRACTION * 100)}% of "
                f"the canvas's {side} margin reservation "
                f"({budget_mm:.2f} mm; allowed: {allowed_mm:.2f} mm). "
                f"The remaining 20% is breathing room so the side fig's "
                f"decoration doesn't visually crowd the neighbouring "
                f"panel's axes-data box. Either (a) shrink the side "
                f"figure's {hint_kind}, (b) use anchor='figure' to fit "
                f"the entire side figure inside the slot rect, or "
                f"(c) increase the canvas's {side} margin reservation. "
                f"PR 6d will auto-expand canvas margins to "
                f"accommodate; until then, this raise is the contract.",
                panel_label=label_str if isinstance(label_str, str)
                and label_str != "<unlabeled>" else None,
            )
