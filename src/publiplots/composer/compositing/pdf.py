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
from publiplots.composer.compositing._constants import (
    _DEFAULT_CREATION_DATE,
    _PRODUCER,
)
from publiplots.composer.compositing._geometry import (
    MM2PT,
    compute_pdf_transform,
)
from publiplots.composer.compositing._resources import (
    _pillow_to_pdf_bytes,
    load_schematic_as_pdf_bytes,
)


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

    Reads ``panel.image_path``, ``panel.image_align``, ``panel.image_clip``,
    ``panel.bbox_mm``, and ``panel.label`` from the post-finalize Panel
    result record (NOT the PanelImage input dataclass — Canvas finalize
    translates the input to a Panel record carrying the image_* fields).

    PR 6b: when ``panel.embedded_figure`` is set (via
    ``canvas.embed_figure``), the embedded figure wins over any path —
    the figure is rendered to deterministic PDF bytes and stamped just
    like a vector schematic.
    """
    label = getattr(panel, "label", None) or "<unlabeled>"
    path = panel.image_path
    embedded_figure = getattr(panel, "embedded_figure", None)
    align = panel.image_align if panel.image_align is not None else "center"
    clip = panel.image_clip if panel.image_clip is not None else "fit"
    bbox_mm = panel.bbox_mm

    if embedded_figure is not None:
        # PR 6b embedded-figure branch — render the Figure to a
        # deterministic PDF buffer and treat like a vector schematic.
        from publiplots.composer.compositing._embed import (
            render_figure_to_pdf_bytes,
        )
        pdf_bytes = render_figure_to_pdf_bytes(embedded_figure)
    elif path is None:
        # Unfilled PanelImage with no embedded figure — actionable error.
        raise ComposerVectorError(
            f"PanelImage {label!r} has no path and no embedded figure; "
            f"either pass path= at construction or call "
            f"canvas.embed_figure(label, fig) before savefig.",
            panel_label=label if isinstance(label, str) else None,
        )
    else:
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

    Delegates to :func:`._resources._pillow_to_pdf_bytes`. If Pillow
    can't open the file, the helper raises :class:`ComposerVectorError`
    (with `path`/`source_error` set), which propagates from here.
    """
    return _pillow_to_pdf_bytes(path)
