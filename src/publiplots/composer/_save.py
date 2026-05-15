"""Save dispatch for the Composer.

PR 1 ships only the raster pipeline; PR 5 lands the vector PDF
pipeline. SVG still raises :class:`NotImplementedError` until PR 6.
The dispatch is by file extension — same shape that the production
design uses, just with most branches deferred.
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
    **kwargs: Any,
) -> None:
    """Dispatch ``canvas.savefig`` by file extension.

    Raster paths (PNG/JPG/TIFF) delegate to :func:`pp.savefig`, which
    inherits publiplots' rcParams defaults (transparent background,
    600 dpi, ``bbox_inches=None``).

    PDF paths dispatch to the PR 5 vector compositing pipeline at
    :func:`publiplots.composer.compositing.pdf.savefig_pdf`. SVG paths
    raise :class:`NotImplementedError` until PR 6 lands the in-tree SVG
    composer.

    The ``panels``, ``strict_vectors``, and ``metadata_creation_date``
    parameters are CONSUMED here and not forwarded into ``**kwargs``;
    the raster branch's :func:`pp.savefig` doesn't know about them.
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
        raise NotImplementedError(
            "Canvas.savefig('*.svg') requires the in-tree SVG composer, "
            "which lands in PR 6. For now, save to PNG/JPG/TIFF/PDF, or call "
            "fig.savefig(...) directly on canvas.figure to bypass compositing."
        )

    raise ValueError(
        f"unknown savefig extension {ext!r}; "
        f"supported: {sorted(_RASTER_EXTS | _VECTOR_PDF_EXTS)}; "
        f"SVG lands in PR 6"
    )
