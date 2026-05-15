"""Vector-SVG compositing pipeline.

The single public entry point is :func:`savefig_svg`. ``Canvas.savefig``
dispatches `.svg` paths here. The pipeline mirrors :mod:`pdf` but
operates on the canvas SVG byte stream rather than a pypdf writer:

1. Render the matplotlib Figure (with empty PanelImage slots) to an
   SVG byte buffer with deterministic-friendly settings (pinned
   ``svg.hashsalt`` rcParam + pinned ``metadata={"Date": ...}``).
2. Parse the canvas SVG via ``lxml.etree`` (preserves the source's
   namespace prefixes — stdlib ``xml.etree.ElementTree`` rewrites
   ``xlink:`` to ``ns0:`` which breaks Inkscape/Illustrator-authored
   schematics).
3. For each PanelImage, load the schematic to an SVG ``<g>``-wrappable
   element via :mod:`._resources`, compute the mm→user-unit transform
   via :mod:`._geometry`, and append the wrapper ``<g>`` to the canvas
   tree.
4. Serialize via ``lxml.etree.tostring`` and write to disk.

The ``strict_vectors`` flag controls behavior on schematic-load
failure: ``True`` re-raises :class:`ComposerVectorError`; ``False``
attempts a raster fallback (Pillow re-render → ``<image>`` data-URI)
and emits a ``UserWarning``.

Limitations (PR 6a):
- PDF-source schematics in SVG output → ``ComposerVectorError`` (no
  in-tree library round-trips PDF→SVG with vector preservation).
- ``<style>`` blocks in schematic SVGs are NOT merged into the canvas
  ``<defs>``; only inline-attribute styling is preserved (open
  question 8 from the spike, deferred to a follow-up PR).
"""
from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from publiplots.composer.exceptions import ComposerVectorError
from publiplots.composer.compositing._constants import (
    _DEFAULT_DATE,
    _SVG_HASHSALT,
)


def _filter_image_panels(panels: Iterable[Any]) -> list[Any]:
    """Return only the panels that need vector compositing."""
    return [p for p in panels if getattr(p, "kind", None) == "image"]


def savefig_svg(
    figure: Figure,
    path: Union[str, Path],
    *,
    panels: Sequence[Any],
    strict_vectors: bool = False,
    metadata_date: Optional[str] = None,
    **savefig_kwargs: Any,
) -> None:
    """Vector-compose the canvas Figure to an SVG at ``path``.

    Parameters
    ----------
    figure
        The matplotlib Figure for the canvas (with empty image slots).
    path
        Output SVG path.
    panels
        The full panel list from ``canvas._panels_list``. Only entries
        with ``panel.kind == 'image'`` trigger compositing.
    strict_vectors
        On any schematic-load failure: ``True`` raises
        :class:`ComposerVectorError`; ``False`` falls back to a raster
        ``<image>`` data-URI element + emits ``UserWarning``.
    metadata_date
        SVG ``<dc:date>`` value. Three-state semantics:

        - ``None`` (default) → write the deterministic
          ``_DEFAULT_DATE`` literal so the byte stream is reproducible.
        - ``"omit"`` → matplotlib strips ``<dc:date>`` entirely (the
          tag is suppressed, NOT defaulted).
        - any other string → write that literal value.

        The default is the deterministic literal because byte
        determinism is part of the composer's contract.

    Raises
    ------
    ComposerVectorError
        ``strict_vectors=True`` AND a schematic failed; OR lxml is not
        installed.
    """
    # lxml is the only NEW [composer] dep; mirror the cairosvg/pypdf
    # install-hint pattern.
    try:
        from lxml import etree as _lxml_etree
    except ImportError as e:
        raise ComposerVectorError(
            "lxml is required for `canvas.savefig('*.svg')`. "
            "Install with `pip install publiplots[composer]`.",
            source_error=str(e),
        ) from e

    output_path = Path(path)
    image_panels = _filter_image_panels(panels)

    # Step 1: render the empty-slot canvas to an SVG buffer with
    # deterministic-friendly metadata. matplotlib's SVG writer accepts
    # a `metadata` dict; passing Date=None suppresses the auto-generated
    # timestamp (matplotlib will then NOT emit `<dc:date>` at all).
    if metadata_date is None:
        date_value: Optional[str] = _DEFAULT_DATE
    elif metadata_date == "omit":
        date_value = None
    else:
        date_value = metadata_date

    canvas_buf = io.BytesIO()
    with plt.rc_context({"svg.hashsalt": _SVG_HASHSALT}):
        figure.savefig(
            canvas_buf,
            format="svg",
            metadata={"Date": date_value},
            **savefig_kwargs,
        )
    canvas_buf.seek(0)

    # Step 2: parse the canvas SVG.
    canvas_tree = _lxml_etree.parse(canvas_buf)
    canvas_root = canvas_tree.getroot()

    # Step 3: stamp each PanelImage.
    for idx, panel in enumerate(image_panels):
        _compose_panel_into(
            canvas_root=canvas_root,
            panel=panel,
            idx=idx,
            strict_vectors=strict_vectors,
            etree=_lxml_etree,
        )

    # Step 4: serialize + write.
    output_bytes = _lxml_etree.tostring(
        canvas_tree,
        pretty_print=False,
        xml_declaration=True,
        standalone=False,
        encoding="utf-8",
    )
    with open(output_path, "wb") as f:
        f.write(output_bytes)


def _compose_panel_into(
    *,
    canvas_root: Any,
    panel: Any,
    idx: int,
    strict_vectors: bool,
    etree: Any,
) -> None:
    """Stamp ``panel``'s schematic into ``canvas_root``.

    Reads ``panel.image_path``, ``panel.image_align``, ``panel.image_clip``,
    ``panel.bbox_mm``, and ``panel.label`` from the post-finalize Panel
    record.

    Builds a ``<g id="publiplots-panel-image-{idx}-{label_or_unlabeled}"
    transform="translate(...) scale(...)">`` wrapper and appends it to
    ``canvas_root``. The wrapper ``id`` follows the
    ``publiplots-panel-image-{idx}-...`` scheme so callers can XPath
    them out for golden-fixture structure assertions; the ``idx``
    suffix disambiguates ``label=False``/``label=None`` collisions
    (SVG ``id`` must be unique per document).
    """
    # Imports lazy: keep _resources's lxml/cairosvg/Pillow imports
    # inside its own functions (mirrors PR 5).
    import re
    from publiplots.composer.compositing._geometry import (
        _resolve_svg_units,
        compute_svg_transform,
    )
    from publiplots.composer.compositing._resources import (
        load_schematic_as_svg_element,
    )

    raw_label = getattr(panel, "label", None)
    if raw_label in (None, False):
        label_str = "unlabeled"
    else:
        # Sanitize for SVG id-attribute safety: XML ids must match
        # [A-Za-z_][\w.\-]*. Replace any other char with `_`; prefix `_`
        # if the result starts with a digit. Labels like "Panel 1",
        # "(i)", "a:b" all become valid ids without collision risk
        # because the wrapper id also carries the panel's idx suffix.
        sanitized = re.sub(r"[^\w.\-]", "_", str(raw_label))
        if sanitized and sanitized[0].isdigit():
            sanitized = f"_{sanitized}"
        label_str = sanitized or "unlabeled"
    path = panel.image_path
    align = panel.image_align if panel.image_align is not None else "center"
    clip = panel.image_clip if panel.image_clip is not None else "fit"
    bbox_mm = panel.bbox_mm  # (x_mm, y_mm_bottom, w_mm, h_mm)

    # Determine the canvas's user-unit geometry FIRST so we can do
    # the bottom-up→top-down y-flip with the canvas's true height.
    vb_x, vb_y, vb_w, vb_h, mm_per_uu = _resolve_svg_units(canvas_root)
    fig_h_mm = vb_h * mm_per_uu

    # Convert slot from BOTTOM-UP mm to TOP-DOWN mm:
    # y_top_mm = fig_h_mm - y_bottom_mm - h_mm
    slot_x_mm, slot_y_bottom_mm, slot_w_mm, slot_h_mm = bbox_mm
    slot_y_top_mm = fig_h_mm - slot_y_bottom_mm - slot_h_mm
    slot_bbox_mm_topdown = (slot_x_mm, slot_y_top_mm, slot_w_mm, slot_h_mm)

    try:
        sch_element, sch_size_mm, _kind = load_schematic_as_svg_element(
            path, label=raw_label,
        )
    except ComposerVectorError:
        if strict_vectors:
            raise
        warnings.warn(
            f"PanelImage {label_str!r}: vector load of {path!r} failed; "
            f"falling back to raster <image> element.",
            UserWarning,
            stacklevel=3,
        )
        sch_element, sch_size_mm = _raster_fallback_to_image_element(
            path, slot_bbox_mm_topdown, etree=etree,
        )

    # Compute the wrapper transform.
    sx, sy, tx_uu, ty_uu = compute_svg_transform(
        slot_bbox_mm_topdown, sch_size_mm,
        canvas_mm_per_user_unit=mm_per_uu,
        canvas_vb_origin=(vb_x, vb_y),
        align=align, clip=clip,
    )

    # Build wrapper. SVG transform syntax: `translate(tx, ty) scale(sx, sy)`.
    SVG_NS = "http://www.w3.org/2000/svg"
    wrapper = etree.SubElement(canvas_root, f"{{{SVG_NS}}}g")
    wrapper.set(
        "id", f"publiplots-panel-image-{idx}-{label_str}",
    )
    wrapper.set(
        "transform",
        f"translate({tx_uu:.6f}, {ty_uu:.6f}) "
        f"scale({sx:.6f}, {sy:.6f})",
    )
    wrapper.append(sch_element)


def _raster_fallback_to_image_element(
    path: Union[str, Path],
    slot_bbox_mm_topdown: tuple,
    *,
    etree: Any,
) -> tuple:
    """Last-ditch raster fallback: build an ``<image>`` data-URI element.

    Pillow opens the file (regardless of extension), re-saves to PNG
    bytes, base64-encodes; the returned ``<image>`` element is sized
    in mm matching the slot dimensions (so the wrapper's
    ``compute_svg_transform`` math is identity-safe).

    Raises ``ComposerVectorError`` if Pillow can't open the file.
    """
    from publiplots.composer.compositing._resources import (
        _pillow_to_data_uri,
    )
    SVG_NS = "http://www.w3.org/2000/svg"
    XLINK_NS = "http://www.w3.org/1999/xlink"
    data_uri = _pillow_to_data_uri(path)
    _, _, slot_w_mm, slot_h_mm = slot_bbox_mm_topdown
    image = etree.Element(
        f"{{{SVG_NS}}}image",
        nsmap={"xlink": XLINK_NS},
    )
    image.set(f"{{{XLINK_NS}}}href", data_uri)
    image.set("x", "0")
    image.set("y", "0")
    image.set("width", f"{slot_w_mm:.6f}")
    image.set("height", f"{slot_h_mm:.6f}")
    image.set("preserveAspectRatio", "none")
    return image, (slot_w_mm, slot_h_mm)
