"""Schematic input → in-memory loaders for the PDF + SVG composers.

Two public dispatchers:

- :func:`load_schematic_as_pdf_bytes` — returns ``(pdf_bytes, kind)``
  for the PDF composer.
- :func:`load_schematic_as_svg_element` — returns ``(element,
  (w_mm, h_mm), kind)`` for the SVG composer.

Both share the :func:`_pillow_to_pdf_bytes` and
:func:`_pillow_to_data_uri` helpers for raster→PDF and raster→SVG-image
conversion respectively.

Optional deps (``pypdf``, ``cairosvg``, ``lxml``, ``Pillow``) are
imported lazily inside each branch so the ``[composer]`` extra is not
a hard dep at module-import time.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Optional, Tuple, Union

from publiplots.composer.exceptions import ComposerVectorError


_VECTOR_EXTS = {".pdf", ".svg"}
_RASTER_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
_DEFAULT_RASTER_DPI = 300.0


def _pillow_to_pdf_bytes(
    path: Union[str, Path],
    dpi: float = _DEFAULT_RASTER_DPI,
) -> bytes:
    """Render a raster file (PNG/JPG/TIFF/...) to single-page PDF bytes.

    Centralizes the Pillow→PDF conversion that was previously duplicated
    between :func:`load_schematic_as_pdf_bytes` (raster branch) and
    :func:`publiplots.composer.compositing.pdf._raster_fallback`. Both
    callers now delegate here.

    The image is opened via Pillow, mode-converted to ``RGB``/``L`` if
    needed (Pillow's PDF encoder rejects other modes), and saved via
    ``Image.save(..., format='PDF', resolution=...)``. Resolution
    defaults to the PNG/JPG/TIFF embedded ``dpi`` info if present, else
    falls back to ``dpi``.

    Parameters
    ----------
    path
        Path to a raster file readable by Pillow.
    dpi
        Default resolution in dots-per-inch when the file lacks
        embedded dpi metadata. Used for the PDF mediabox (1 pt = 1/72 in).

    Returns
    -------
    bytes
        A complete single-page PDF starting with ``b"%PDF-"``.

    Raises
    ------
    ComposerVectorError
        Pillow couldn't open or convert the file. The error carries
        ``path`` and ``source_error``.
    """
    p = Path(path)
    try:
        from PIL import Image
    except ImportError as e:
        raise ComposerVectorError(
            "Pillow is required for raster schematics. "
            "Install with `pip install publiplots[composer]`.",
            path=str(p),
            source_error=str(e),
        ) from e
    try:
        img = Image.open(p)
        img.load()
        # Pillow's PDF save needs RGB or L mode.
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        # Prefer the file's embedded dpi if present (Pillow stores it as a
        # 2-tuple (xdpi, ydpi)); else fall back to caller-supplied dpi.
        embedded = img.info.get("dpi")
        if embedded:
            resolution = float(embedded[0])
        else:
            resolution = float(dpi)
        img.save(buf, format="PDF", resolution=resolution)
        return buf.getvalue()
    except Exception as e:
        raise ComposerVectorError(
            f"Pillow failed to convert {p.name!r}: {e}",
            path=str(p),
            source_error=str(e),
        ) from e


def load_schematic_as_pdf_bytes(
    path: Union[str, Path],
) -> Tuple[bytes, str]:
    """Convert a schematic file into a single-page PDF byte string.

    Returns
    -------
    pdf_bytes : bytes
        A complete, valid PDF starting with ``b"%PDF-"``.
    source_kind : {'vector', 'raster'}
        Whether the source was vector (PDF/SVG) or raster (PNG/JPG/TIFF).

    Raises
    ------
    FileNotFoundError
        Path doesn't exist (also caught by PanelImage's __post_init__,
        but rechecked here for direct callers).
    ComposerVectorError
        A vector source failed to load/convert (corrupt SVG, malformed
        PDF, missing optional dep). The error carries ``path`` and
        ``source_error``.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"schematic not found: {p}")
    ext = p.suffix.lower()

    if ext == ".pdf":
        try:
            import pypdf
        except ImportError as e:
            raise ComposerVectorError(
                "pypdf is required for PDF schematics. "
                "Install with `pip install publiplots[composer]`.",
                path=str(p),
                source_error=str(e),
            ) from e
        try:
            reader = pypdf.PdfReader(str(p))
            n_pages = len(reader.pages)
            if n_pages == 1:
                # Single-page input: pass through bytes directly. Avoids
                # pypdf round-trip which can renumber objects + add a
                # Producer marker that conflicts with the composer's.
                return p.read_bytes(), "vector"
            # Multi-page: take page 0 + warn (single warn per path).
            import warnings
            warnings.warn(
                f"PanelImage schematic {p.name!r} has "
                f"{n_pages} pages; using page 0.",
                UserWarning,
                stacklevel=3,
            )
            writer = pypdf.PdfWriter()
            writer.add_page(reader.pages[0])
            buf = io.BytesIO()
            writer.write(buf)
            return buf.getvalue(), "vector"
        except Exception as e:
            raise ComposerVectorError(
                f"pypdf failed to read {p.name!r}: {e}",
                path=str(p),
                source_error=str(e),
            ) from e

    if ext == ".svg":
        try:
            import cairosvg
        except ImportError as e:
            raise ComposerVectorError(
                "cairosvg is required for SVG schematics. "
                "Install with `pip install publiplots[composer]`.",
                path=str(p),
                source_error=str(e),
            ) from e
        try:
            pdf_bytes = cairosvg.svg2pdf(url=str(p))
            return pdf_bytes, "vector"
        except Exception as e:
            raise ComposerVectorError(
                f"cairosvg failed to convert {p.name!r}: {e}",
                path=str(p),
                source_error=str(e),
            ) from e

    if ext in _RASTER_EXTS:
        return _pillow_to_pdf_bytes(p), "raster"

    raise ComposerVectorError(
        f"unsupported schematic extension {ext!r} for {p.name!r}",
        path=str(p),
        source_error=None,
    )


def _pillow_to_data_uri(
    path: Union[str, Path],
    dpi: float = _DEFAULT_RASTER_DPI,
) -> str:
    """Render a raster file via Pillow → ``"data:image/png;base64,..."``.

    Re-encoding through Pillow normalises the byte stream (Illustrator-
    exported PNGs sometimes carry non-sRGB color profiles that some
    browsers refuse to render); the returned URI is a fresh PNG.

    Parameters
    ----------
    path
        Path to a raster file readable by Pillow.
    dpi
        Default resolution if the source has no embedded ``dpi`` info.
        Used by callers to compute the corresponding mm size.

    Returns
    -------
    str
        ``"data:image/png;base64,..."``-prefixed URI.

    Raises
    ------
    ComposerVectorError
        Pillow couldn't open or convert the file.
    """
    p = Path(path)
    try:
        from PIL import Image
    except ImportError as e:
        raise ComposerVectorError(
            "Pillow is required for raster schematics. "
            "Install with `pip install publiplots[composer]`.",
            path=str(p),
            source_error=str(e),
        ) from e
    try:
        img = Image.open(p)
        img.load()
        # PNG accepts most modes, but normalise palette/CMYK to RGB(A).
        if img.mode not in ("RGB", "RGBA", "L", "LA"):
            img = img.convert("RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception as e:
        raise ComposerVectorError(
            f"Pillow failed to convert {p.name!r} to PNG data URI: {e}",
            path=str(p),
            source_error=str(e),
        ) from e


def load_schematic_as_svg_element(
    path: Union[str, Path],
    *,
    label: Optional[Any] = None,
) -> Tuple[Any, Tuple[float, float], str]:
    """Load a schematic file as an lxml element for SVG-canvas embedding.

    Three branches:

    - ``.svg``: lxml-parse the file; return the root element and its
      intrinsic mm size (computed via :func:`_resolve_svg_units`).
    - ``.png``/``.jpg``/etc: build a synthetic ``<image>`` element with
      an ``xlink:href`` data URI and intrinsic mm size derived from
      pixel dims × ``25.4 / dpi``.
    - ``.pdf``: explicitly raise — no in-tree library round-trips
      PDF → SVG with vector preservation.

    Parameters
    ----------
    path
        Path to the schematic file.
    label
        Optional panel label, used in the PDF-rejection error message
        to make the failure mode unambiguous.

    Returns
    -------
    (element, (w_mm, h_mm), source_kind) : 3-tuple
        ``element`` is an ``lxml.etree._Element`` ready to be wrapped
        in a ``<g transform=...>`` and appended to the canvas tree.
        ``source_kind`` is ``'vector'`` (SVG) or ``'raster'`` (image).

    Raises
    ------
    FileNotFoundError
        The path doesn't exist.
    ComposerVectorError
        Unsupported extension; corrupt SVG; PDF input; missing viewBox
        in an SVG schematic; missing ``lxml``/``Pillow``.
    """
    from publiplots.composer.compositing._geometry import _resolve_svg_units

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"schematic not found: {p}")
    ext = p.suffix.lower()

    if ext == ".svg":
        try:
            from lxml import etree as _lxml_etree
        except ImportError as e:
            raise ComposerVectorError(
                "lxml is required for SVG schematic embedding. "
                "Install with `pip install publiplots[composer]`.",
                path=str(p),
                source_error=str(e),
            ) from e
        try:
            tree = _lxml_etree.parse(str(p))
        except Exception as e:
            raise ComposerVectorError(
                f"lxml failed to parse SVG schematic {p.name!r}: {e}",
                path=str(p),
                source_error=str(e),
            ) from e
        root = tree.getroot()
        # _resolve_svg_units raises ComposerVectorError for missing
        # viewBox / relative-unit width — propagate as-is.
        _, _, vb_w, vb_h, mm_per_uu = _resolve_svg_units(root)
        # Intrinsic mm = viewBox dims × mm-per-user-unit.
        sch_w_mm = vb_w * mm_per_uu
        sch_h_mm = vb_h * mm_per_uu
        return root, (sch_w_mm, sch_h_mm), "vector"

    if ext == ".pdf":
        label_str = repr(label) if label not in (None, False) else "<unlabeled>"
        raise ComposerVectorError(
            f"PanelImage {label_str}: PDF schematic {p.name!r} cannot be "
            f"embedded in SVG output. PDF→SVG vector round-trip is not "
            f"supported by cairosvg or any in-tree library. Either "
            f"(a) convert {p.name!r} to SVG first, or (b) use "
            f"canvas.savefig('fig.pdf') for PDF output where this "
            f"PanelImage works as expected.",
            path=str(p),
            panel_label=str(label) if label not in (None, False) else None,
        )

    if ext in _RASTER_EXTS:
        try:
            from lxml import etree as _lxml_etree
        except ImportError as e:
            raise ComposerVectorError(
                "lxml is required for SVG schematic embedding. "
                "Install with `pip install publiplots[composer]`.",
                path=str(p),
                source_error=str(e),
            ) from e
        try:
            from PIL import Image
        except ImportError as e:
            raise ComposerVectorError(
                "Pillow is required for raster schematics. "
                "Install with `pip install publiplots[composer]`.",
                path=str(p),
                source_error=str(e),
            ) from e
        # Probe pixel dims + dpi without re-encoding.
        with Image.open(p) as probe:
            probe.load()
            px_w, px_h = probe.size
            embedded = probe.info.get("dpi")
            dpi = float(embedded[0]) if embedded else _DEFAULT_RASTER_DPI
        sch_w_mm = px_w * 25.4 / dpi
        sch_h_mm = px_h * 25.4 / dpi
        # Build the data-URI element.
        SVG_NS = "http://www.w3.org/2000/svg"
        XLINK_NS = "http://www.w3.org/1999/xlink"
        image = _lxml_etree.Element(
            f"{{{SVG_NS}}}image",
            nsmap={"xlink": XLINK_NS},
        )
        image.set(f"{{{XLINK_NS}}}href", _pillow_to_data_uri(p, dpi=dpi))
        image.set("x", "0")
        image.set("y", "0")
        # Width/height in the schematic's intrinsic mm coordinate system;
        # the wrapper <g> scales from mm→user-units.
        image.set("width", f"{sch_w_mm:.6f}")
        image.set("height", f"{sch_h_mm:.6f}")
        image.set("preserveAspectRatio", "none")
        return image, (sch_w_mm, sch_h_mm), "raster"

    raise ComposerVectorError(
        f"unsupported schematic extension {ext!r} for {p.name!r}",
        path=str(p),
        source_error=None,
    )
