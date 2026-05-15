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
"""
from __future__ import annotations

import io
from typing import Any, Optional

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from publiplots.composer.compositing._constants import (
    _DEFAULT_CREATION_DATE,
    _DEFAULT_DATE,
    _PRODUCER,
    _SVG_HASHSALT,
)


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
