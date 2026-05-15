"""Save dispatch for the Composer.

PR 1 ships only the raster pipeline; PR 5 lands the vector PDF
pipeline; PR 6a lands the vector SVG pipeline. The dispatch is by
file extension — same shape that the production design uses, just
with multi-format save deferred to PR 6b.
"""

from pathlib import Path
from typing import Any, Optional, Sequence, Union

from matplotlib.figure import Figure

from publiplots.utils.io import savefig as _pp_savefig


_RASTER_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
_VECTOR_PDF_EXTS = {".pdf"}
_VECTOR_SVG_EXTS = {".svg"}


def dispatch_savefig(
    figure: Figure,
    path: Union[str, Path],
    *,
    panels: Sequence[Any] = (),
    strict_vectors: bool = False,
    metadata_creation_date: Optional[str] = None,
    metadata_date: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Dispatch ``canvas.savefig`` by file extension.

    Raster paths (PNG/JPG/TIFF) delegate to :func:`pp.savefig`, which
    inherits publiplots' rcParams defaults (transparent background,
    600 dpi, ``bbox_inches=None``).

    PDF paths dispatch to the PR 5 vector compositing pipeline at
    :func:`publiplots.composer.compositing.pdf.savefig_pdf`. SVG paths
    dispatch to the PR 6a in-tree SVG composer at
    :func:`publiplots.composer.compositing.svg.savefig_svg`.

    The ``panels``, ``strict_vectors``, ``metadata_creation_date``,
    and ``metadata_date`` parameters are CONSUMED here and not
    forwarded into ``**kwargs``; the raster branch's
    :func:`pp.savefig` doesn't know about them. ``metadata_creation_date``
    is the PDF ``/CreationDate`` value; ``metadata_date`` is the SVG
    ``Date`` metadata key (matplotlib's ``<dc:date>``). They have
    distinct kwargs because the underlying writers use different
    metadata names.
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext in _RASTER_EXTS:
        _pp_savefig(str(p), **kwargs)
        return

    if ext in _VECTOR_PDF_EXTS:
        # PR 5: dispatch to compositing.pdf.savefig_pdf
        from publiplots.composer.compositing.pdf import savefig_pdf
        savefig_pdf(
            figure,
            p,
            panels=list(panels),
            strict_vectors=strict_vectors,
            metadata_creation_date=metadata_creation_date,
            **kwargs,
        )
        return
    if ext in _VECTOR_SVG_EXTS:
        # PR 6a: dispatch to compositing.svg.savefig_svg
        from publiplots.composer.compositing.svg import savefig_svg
        savefig_svg(
            figure,
            p,
            panels=list(panels),
            strict_vectors=strict_vectors,
            metadata_date=metadata_date,
            **kwargs,
        )
        return

    raise ValueError(
        f"unknown savefig extension {ext!r}; "
        f"supported: {sorted(_RASTER_EXTS | _VECTOR_PDF_EXTS | _VECTOR_SVG_EXTS)}"
    )
