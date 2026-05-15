"""Schematic input → in-memory single-page PDF bytes.

Each branch returns a ``(pdf_bytes, source_kind)`` tuple where
``source_kind`` is ``'vector'`` (PDF/SVG) or ``'raster'`` (PNG/JPG/TIFF).
The orchestrator in :mod:`pdf` uses ``source_kind`` to decide whether
``strict_vectors=True`` should reject a raster fallback.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple, Union

from publiplots.composer.exceptions import ComposerVectorError


_VECTOR_EXTS = {".pdf", ".svg"}
_RASTER_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
_DEFAULT_RASTER_DPI = 300.0


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
            dpi = img.info.get("dpi", (_DEFAULT_RASTER_DPI, _DEFAULT_RASTER_DPI))
            img.save(buf, format="PDF", resolution=float(dpi[0]))
            return buf.getvalue(), "raster"
        except Exception as e:
            raise ComposerVectorError(
                f"Pillow failed to convert {p.name!r}: {e}",
                path=str(p),
                source_error=str(e),
            ) from e

    raise ComposerVectorError(
        f"unsupported schematic extension {ext!r} for {p.name!r}",
        path=str(p),
        source_error=None,
    )
