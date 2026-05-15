"""Save dispatch for the Composer.

PR 1 ships only the raster pipeline; PR 5 lands the vector PDF
pipeline; PR 6a lands the vector SVG pipeline. PR 6b adds CMYK,
TIFF compression knob, and external-raster sidecar PNGs for SVG.
The dispatch is by file extension — same shape that the production
design uses.
"""

import io
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from matplotlib.figure import Figure

from publiplots.utils.io import savefig as _pp_savefig


_RASTER_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
_VECTOR_PDF_EXTS = {".pdf"}
_VECTOR_SVG_EXTS = {".svg"}

_DEFAULT_TIFF_COMPRESSION = "tiff_lzw"

# PR 6b: ext → Pillow ``Image.save(format=...)`` map. ``ext.upper()``
# is INCORRECT for ``.tif`` / ``.tiff`` (Pillow expects ``"TIFF"`` for
# both). Architect-found bug.
_PILLOW_FORMAT = {
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
}
_CMYK_RASTER_EXTS = {".tif", ".tiff", ".jpg", ".jpeg"}


def _render_via_pp_savefig_to_png_buffer(
    figure: Figure,
    **kwargs: Any,
) -> io.BytesIO:
    """Render the canvas to a PNG BytesIO via pp.savefig.

    Used by the CMYK + custom-TIFF-compression branches: Pillow
    re-opens the buffer and re-saves to the actual target format.
    Mirrors the existing PR 5/6a "render to PDF buffer, restamp via
    pypdf" pattern.

    pp.savefig accepts a file-like object as the first arg.
    """
    buf = io.BytesIO()
    # pp.savefig writes via figure.savefig under the hood; passing buf
    # avoids tempfile creation. Force the raster-PNG backend by
    # appending the extension via matplotlib's `format` kwarg.
    figure.savefig(buf, format="png", **kwargs)
    buf.seek(0)
    return buf


def _save_raster_via_pillow(
    figure: Figure,
    path: Path,
    *,
    cmyk: bool,
    tiff_compression: str,
    **kwargs: Any,
) -> None:
    """Render via pp.savefig → PNG buffer; re-save via Pillow with
    the requested mode + compression knobs.

    Used when ``cmyk=True`` OR ``tiff_compression`` differs from the
    default. The PNG round-trip is required because matplotlib's TIFF
    writer doesn't expose ``compression`` via savefig kwargs.
    """
    from PIL import Image

    ext = path.suffix.lower()
    pil_format = _PILLOW_FORMAT[ext]
    buf = _render_via_pp_savefig_to_png_buffer(figure, **kwargs)
    img = Image.open(buf)
    img.load()  # decode now so we can close the underlying BytesIO

    if cmyk:
        # Architect-flagged: PR 6b uses Pillow's built-in sRGB→CMYK
        # conversion (no ICC profile bundled). Emit a UserWarning so
        # journal QA users can debug rejected submissions.
        import warnings as _warn
        _warn.warn(
            "cmyk=True: using Pillow's basic RGB→CMYK conversion "
            "(no ICC profile). Journal-grade FOGRA39 / SWOP v2 profile "
            "bundling is deferred to PR 7. Verify the output "
            "with your target journal's pre-flight tooling.",
            UserWarning,
            stacklevel=4,
        )
        img = img.convert("CMYK")

    save_kwargs: dict = {}
    if pil_format == "TIFF":
        save_kwargs["compression"] = tiff_compression
    img.save(path, format=pil_format, **save_kwargs)


def dispatch_savefig(
    figure: Figure,
    path: Union[str, Path],
    *,
    panels: Sequence[Any] = (),
    strict_vectors: bool = False,
    metadata_creation_date: Optional[str] = None,
    metadata_date: Optional[str] = None,
    cmyk: bool = False,
    tiff_compression: str = _DEFAULT_TIFF_COMPRESSION,
    external_raster: bool = False,
    **kwargs: Any,
) -> None:
    """Dispatch ``canvas.savefig`` by file extension.

    Raster paths (PNG/JPG/TIFF) delegate to :func:`pp.savefig` for the
    base render; CMYK or non-default ``tiff_compression`` requests
    re-render through Pillow.

    PDF paths dispatch to the PR 5 vector compositing pipeline at
    :func:`publiplots.composer.compositing.pdf.savefig_pdf`. SVG paths
    dispatch to the PR 6a in-tree SVG composer at
    :func:`publiplots.composer.compositing.svg.savefig_svg`.

    The ``panels``, ``strict_vectors``, ``metadata_creation_date``,
    ``metadata_date``, ``cmyk``, ``tiff_compression``, and
    ``external_raster`` parameters are CONSUMED here and not forwarded
    into ``**kwargs`` — leaving them in ``**kwargs`` would leak them
    to the underlying ``pp.savefig`` / matplotlib backends, which raise
    ``AttributeError`` on unknown kwargs. (Architect-found leak hazard.)
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext in _RASTER_EXTS:
        # CMYK validation FIRST — PNG doesn't support CMYK at all.
        if cmyk and ext == ".png":
            raise ValueError(
                "PNG does not support CMYK; use .tif/.tiff/.jpg/.jpeg instead."
            )
        # CMYK → Pillow re-render path (only library that can do RGB→CMYK
        # on raster output; matplotlib's TIFF/JPEG backend is RGB-only).
        if cmyk:
            _save_raster_via_pillow(
                figure, p,
                cmyk=cmyk,
                tiff_compression=tiff_compression,
                **kwargs,
            )
            return
        # TIFF: thread `tiff_compression` through to matplotlib's PIL
        # backend via `pil_kwargs={"compression": ...}`. matplotlib's
        # default TIFF is RAW (uncompressed); without this thread-through,
        # the documented `tiff_compression='tiff_lzw'` default would be
        # silently ignored, producing 10–20× larger files than expected.
        if ext in {".tif", ".tiff"}:
            tiff_pil_kwargs = dict(kwargs.pop("pil_kwargs", {}))
            tiff_pil_kwargs.setdefault("compression", tiff_compression)
            _pp_savefig(str(p), pil_kwargs=tiff_pil_kwargs, **kwargs)
            return
        # PNG / JPEG default path — matplotlib backend handles directly.
        _pp_savefig(str(p), **kwargs)
        return

    if ext in _VECTOR_PDF_EXTS:
        # Belt-and-braces: ``Canvas.savefig`` already pre-validates this,
        # but ``dispatch_savefig`` is a semi-public dispatch boundary
        # (importable from compositing/) — guard against direct callers
        # that bypass Canvas.
        if cmyk:
            raise ValueError(
                "cmyk=True is only valid for raster outputs "
                "(.tif/.tiff/.jpg/.jpeg); matplotlib's PDF backend "
                "emits RGB, and there is no in-tree library that can "
                "produce CMYK PDFs. Convert the matching raster output "
                "instead."
            )
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
        if cmyk:
            raise ValueError(
                "cmyk=True is only valid for raster outputs "
                "(.tif/.tiff/.jpg/.jpeg); cairosvg cannot produce "
                "CMYK SVG. Convert the matching raster output instead."
            )
        # PR 6a: dispatch to compositing.svg.savefig_svg
        # PR 6b: thread external_raster through.
        from publiplots.composer.compositing.svg import savefig_svg
        savefig_svg(
            figure,
            p,
            panels=list(panels),
            strict_vectors=strict_vectors,
            metadata_date=metadata_date,
            external_raster=external_raster,
            **kwargs,
        )
        return

    raise ValueError(
        f"unknown savefig extension {ext!r}; "
        f"supported: {sorted(_RASTER_EXTS | _VECTOR_PDF_EXTS | _VECTOR_SVG_EXTS)}"
    )
