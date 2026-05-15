"""Vector-PDF compositing pipeline.

The single public entry point is :func:`savefig_pdf`. ``Canvas.savefig``
dispatches `.pdf` paths here. The pipeline is:

1. Render the matplotlib Figure (with empty PanelImage slots) to a
   PDF byte buffer with deterministic-friendly settings.
2. Open the canvas PDF as a :class:`pypdf.PdfReader`.
3. Build a :class:`pypdf.PdfWriter` via ``clone_from=canvas_reader``
   (spike Finding 3 — the modern idiom that survives pypdf 7.0's
   deprecation of ``merge_transformed_page`` on reader pages).
4. For each PanelImage panel, convert the schematic to a one-page PDF
   via :mod:`._resources`, compute the mm→pt transform via
   :mod:`._geometry`, and stamp the schematic onto the writer's page.
5. Pin ``/CreationDate`` + ``/Producer`` for byte-determinism, then
   write to disk.

The ``strict_vectors`` flag controls behavior on schematic-load
failure: ``True`` re-raises :class:`ComposerVectorError`; ``False``
attempts a raster fallback (re-render the schematic via Pillow) and
emits a ``UserWarning``.
"""
from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

from matplotlib.figure import Figure

from publiplots.composer.exceptions import ComposerVectorError
from publiplots.composer.compositing._geometry import (
    MM2PT,
    compute_pdf_transform,
)
from publiplots.composer.compositing._resources import (
    load_schematic_as_pdf_bytes,
)


_DEFAULT_CREATION_DATE = "D:20260101000000Z"
_PRODUCER = "publiplots-composer"


def _filter_image_panels(panels: Iterable[Any]) -> list[Any]:
    """Return only the panels that need vector compositing."""
    return [p for p in panels if getattr(p, "kind", None) == "image"]


def savefig_pdf(
    figure: Figure,
    path: Union[str, Path],
    *,
    panels: Sequence[Any],
    strict_vectors: bool = False,
    metadata_creation_date: Optional[str] = None,
    **savefig_kwargs: Any,
) -> None:
    """Vector-compose the canvas Figure to a PDF at ``path``.

    Parameters
    ----------
    figure
        The matplotlib Figure for the canvas (with empty image slots).
    path
        Output PDF path.
    panels
        The full panel list from ``canvas._panels_list``. Only entries
        with ``panel.kind == 'image'`` trigger compositing.
    strict_vectors
        On any schematic-load failure: ``True`` raises
        :class:`ComposerVectorError`; ``False`` falls back to a raster
        re-render + emits ``UserWarning``.
    metadata_creation_date
        PDF ``/CreationDate`` value. Defaults to a fixed string for
        byte-determinism. Pass ``None`` to use pypdf's auto-generated
        timestamp (NOT recommended — breaks goldens).

    Raises
    ------
    ComposerVectorError
        ``strict_vectors=True`` AND a schematic failed; OR pypdf is not
        installed.
    """
    try:
        import pypdf
        from pypdf.generic import NameObject, TextStringObject
    except ImportError as e:
        raise ComposerVectorError(
            "pypdf is required for `canvas.savefig('*.pdf')`. "
            "Install with `pip install publiplots[composer]`.",
            source_error=str(e),
        ) from e

    output_path = Path(path)
    image_panels = _filter_image_panels(panels)

    # Step 1: render the empty-slot canvas to a PDF buffer with
    # deterministic-friendly metadata. matplotlib's PDF writer accepts a
    # `metadata` dict; passing CreationDate=None suppresses the
    # auto-generated timestamp.
    canvas_buf = io.BytesIO()
    figure.savefig(
        canvas_buf,
        format="pdf",
        metadata={"CreationDate": None},
        **savefig_kwargs,
    )
    canvas_buf.seek(0)

    # Step 2 + 3: open as reader; build writer via clone_from (spike
    # Finding 3).
    canvas_reader = pypdf.PdfReader(canvas_buf)
    writer = pypdf.PdfWriter(clone_from=canvas_reader)
    target_page = writer.pages[0]

    # Step 4: stamp each PanelImage.
    for panel in image_panels:
        _compose_panel_onto(
            writer=writer,
            target_page=target_page,
            panel=panel,
            strict_vectors=strict_vectors,
            pypdf=pypdf,
        )

    # Step 5: deterministic metadata + write to disk.
    cd = (metadata_creation_date if metadata_creation_date is not None
          else _DEFAULT_CREATION_DATE)
    writer.add_metadata({
        NameObject("/Producer"): TextStringObject(_PRODUCER),
        NameObject("/CreationDate"): TextStringObject(cd),
    })
    with open(output_path, "wb") as f:
        writer.write(f)


def _compose_panel_onto(
    *,
    writer: Any,
    target_page: Any,
    panel: Any,
    strict_vectors: bool,
    pypdf: Any,
) -> None:
    """Stamp ``panel``'s schematic onto ``target_page``.

    Reads ``panel.path``, ``panel.bbox_mm``, ``panel.align``,
    ``panel.clip``, and ``panel.label``.
    """
    label = getattr(panel, "label", None) or "<unlabeled>"
    path = getattr(panel, "image_path", None) or getattr(panel, "path", None)
    align = getattr(panel, "image_align", None) or getattr(panel, "align", "center")
    clip = getattr(panel, "image_clip", None) or getattr(panel, "clip", "fit")
    bbox_mm = getattr(panel, "bbox_mm")

    try:
        pdf_bytes, _kind = load_schematic_as_pdf_bytes(path)
    except ComposerVectorError as e:
        if strict_vectors:
            raise
        warnings.warn(
            f"PanelImage {label!r}: vector load of {path!r} failed "
            f"({e}); falling back to raster.",
            UserWarning,
            stacklevel=3,
        )
        pdf_bytes = _raster_fallback(path)

    schematic_reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    schematic_page = schematic_reader.pages[0]
    sch_mb = schematic_page.mediabox
    sch_w_pt = float(sch_mb.width)
    sch_h_pt = float(sch_mb.height)

    sx, sy, tx_pt, ty_pt = compute_pdf_transform(
        bbox_mm, (sch_w_pt, sch_h_pt), align=align, clip=clip,
    )
    transformation = (
        pypdf.Transformation()
        .scale(sx=sx, sy=sy)
        .translate(tx=tx_pt, ty=ty_pt)
    )
    target_page.merge_transformed_page(schematic_page, transformation)


def _raster_fallback(path: Union[str, Path]) -> bytes:
    """Last-ditch raster render when vector load failed.

    Tries to open the file via Pillow regardless of extension; if
    Pillow can't open it, re-raises :class:`ComposerVectorError`.
    """
    from PIL import Image
    try:
        img = Image.open(path)
        img.load()
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PDF", resolution=300.0)
        return buf.getvalue()
    except Exception as e:
        raise ComposerVectorError(
            f"raster fallback also failed for {path!r}: {e}",
            path=str(path),
            source_error=str(e),
        ) from e
