"""Save dispatch for the Composer.

PR 1 ships only the raster pipeline; PDF and SVG raise
:class:`NotImplementedError` with pointers to the PRs that will add
those backends. The dispatch is by file extension — same shape that
the production design uses, just with most branches deferred.
"""

from pathlib import Path
from typing import Any, Union

from matplotlib.figure import Figure

from publiplots.utils.io import savefig as _pp_savefig


_RASTER_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
_VECTOR_PDF_EXTS = {".pdf"}
_VECTOR_SVG_EXTS = {".svg"}


def dispatch_savefig(
    figure: Figure,
    path: Union[str, Path],
    **kwargs: Any,
) -> None:
    """Dispatch ``canvas.savefig`` by file extension.

    PR 1: raster paths (PNG/JPG/TIFF) delegate to :func:`pp.savefig`,
    which inherits publiplots' rcParams defaults (transparent
    background, 600 dpi, ``bbox_inches=None``).

    PR 5/6: PDF and SVG dispatch to vector compositing pipelines. In
    PR 1, those raise :class:`NotImplementedError` with a pointer.
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext in _RASTER_EXTS:
        _pp_savefig(str(p), **kwargs)
        return

    if ext in _VECTOR_PDF_EXTS:
        raise NotImplementedError(
            "Canvas.savefig('*.pdf') requires the vector compositing pipeline, "
            "which lands in PR 5. For now, save to PNG/JPG/TIFF, or call "
            "fig.savefig(...) directly on canvas.figure to bypass compositing."
        )
    if ext in _VECTOR_SVG_EXTS:
        raise NotImplementedError(
            "Canvas.savefig('*.svg') requires the in-tree SVG composer, "
            "which lands in PR 6. For now, save to PNG/JPG/TIFF, or call "
            "fig.savefig(...) directly on canvas.figure to bypass compositing."
        )

    raise ValueError(
        f"unknown savefig extension {ext!r}; "
        f"supported in PR 1: {sorted(_RASTER_EXTS)}; "
        f"PDF lands in PR 5, SVG in PR 6"
    )
