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
    external_raster: bool = False,
    canvas: Any = None,
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
    external_raster
        When ``True``, raster PanelImage sources are written as sidecar
        PNG files alongside the SVG (``<output_stem>-<idx>-<label>.png``)
        and referenced via a relative ``<image href="...">`` rather
        than inline ``data:image/png;base64,`` URIs. Default ``False``
        preserves PR 6a's inline-data-URI behavior. Silent no-op when
        no raster sources are present.

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
            external_raster=external_raster,
            output_path=output_path,
            canvas=canvas,
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
    external_raster: bool = False,
    output_path: Optional[Path] = None,
    canvas: Any = None,
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
    embedded_figure = getattr(panel, "embedded_figure", None)
    embed_anchor = getattr(panel, "embedded_figure_anchor", "figure")
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

    # PR 6c: anchor='axes' branch — early dispatch with axes-data-box
    # alignment. Strict ordering: settle → extract → check → render →
    # apply. Mirrors compositing/pdf.py's branch.
    if embedded_figure is not None and embed_anchor == "axes":
        from publiplots.composer.compositing._embed import (
            _settle_subplots_auto_layout,
            check_decoration_overflow,
            extract_side_axes_bbox,
            render_figure_to_svg_bytes,
        )
        # 1. Settle.
        _settle_subplots_auto_layout(embedded_figure)
        # 2. Extract axes-data bbox in pt (BOTTOM-UP).
        axes_bbox_pt = extract_side_axes_bbox(embedded_figure)
        axes_left_pt, axes_bottom_pt, axes_w_pt, axes_h_pt = axes_bbox_pt
        # 3. Check overflow vs canvas budget.
        if canvas is not None:
            decoration_budget_mm = canvas._panel_decoration_budget_mm(panel)
            check_decoration_overflow(
                embedded_figure, axes_bbox_pt,
                decoration_budget_mm,
                panel_label=panel.label,
                slot_size_mm=(slot_w_mm, slot_h_mm),
            )
        # 4. Render side fig to deterministic SVG bytes; parse via lxml.
        sch_bytes = render_figure_to_svg_bytes(embedded_figure)
        sch_element = etree.fromstring(sch_bytes)
        sch_vb_x, sch_vb_y, sch_vb_w, sch_vb_h, sch_mm_per_uu = (
            _resolve_svg_units(sch_element)
        )
        # Side fig's full mediabox in mm.
        sch_w_mm = sch_vb_w * sch_mm_per_uu
        sch_h_mm = sch_vb_h * sch_mm_per_uu

        # 5. Compute transform. Side fig's axes-data box (in pt) maps
        # to the slot rect (in mm). pt → mm via 25.4/72.
        from publiplots.composer.compositing._embed import _PT2MM as _PT2MM
        if axes_w_pt <= 0 or axes_h_pt <= 0:
            # Degenerate: fall back to figure-anchor semantics by
            # computing mediabox transform.
            sx, sy, tx_uu, ty_uu = compute_svg_transform(
                slot_bbox_mm_topdown, (sch_w_mm, sch_h_mm),
                canvas_mm_per_user_unit=mm_per_uu,
                canvas_vb_origin=(vb_x, vb_y),
                align=align, clip=clip,
            )
        else:
            # Side fig's axes-data extents in mm.
            axes_left_mm = axes_left_pt * _PT2MM
            axes_w_mm = axes_w_pt * _PT2MM
            axes_h_mm = axes_h_pt * _PT2MM
            # PDF y is bottom-up; SVG side fig's mediabox is also (the
            # render bytes carry matplotlib's bottom-up SVG which has
            # been flipped during render). For SVG, convert side-fig
            # axes-data y from BOTTOM-UP to TOP-DOWN within the side
            # fig's own mediabox.
            axes_bottom_mm = axes_bottom_pt * _PT2MM
            axes_top_mm_from_side_top = sch_h_mm - axes_bottom_mm - axes_h_mm

            # Scale: slot_w_mm / axes_w_mm (anisotropic).
            scale_x = slot_w_mm / axes_w_mm
            scale_y = slot_h_mm / axes_h_mm

            # User-unit conversion: scale factors are dimensionless
            # (mm/mm); translate is in canvas user units.
            sx = scale_x
            sy = scale_y
            # After scale, the side fig's axes-data top-left in side-fig
            # user units lands at (axes_left_mm * sx / sch_mm_per_uu,
            # axes_top_mm_from_side_top * sy / sch_mm_per_uu) in CANVAS
            # user units. Wait — the SCH element retains its own
            # viewBox; the wrapper's scale + translate operate on
            # CANVAS user units, but apply to the inner SCH coordinate
            # system. We need to think in canvas user units (uu):
            #
            # The inner SCH content is in side-fig user units (sch_uu).
            # After the wrapper transform, sch_uu maps to canvas_uu via
            # scale = (sch_mm_per_uu / mm_per_uu) baseline; we
            # additionally need to scale so that sch's axes-data box
            # (axes_w_mm wide in side-fig mm) lands at slot_w_mm in
            # canvas mm. Combined scale (sch_uu → canvas_uu):
            #   sx = (slot_w_mm / axes_w_mm) * (sch_mm_per_uu / mm_per_uu)
            sx = (slot_w_mm / axes_w_mm) * (sch_mm_per_uu / mm_per_uu)
            sy = (slot_h_mm / axes_h_mm) * (sch_mm_per_uu / mm_per_uu)

            # Translation: after the wrapper applies (translate, scale),
            # the SCH element's content is sx * sch_uu away from origin.
            # The SCH's axes-data top-left in SCH user units is
            # (axes_left_mm / sch_mm_per_uu, axes_top_mm_from_side_top
            # / sch_mm_per_uu). We need that point to land at the slot's
            # top-left in canvas user units (slot_x_mm / mm_per_uu,
            # slot_y_top_mm / mm_per_uu) PLUS the canvas viewBox origin
            # offset (vb_x, vb_y).
            slot_x_uu = slot_x_mm / mm_per_uu
            slot_y_top_uu = slot_y_top_mm / mm_per_uu
            sch_axes_left_uu = axes_left_mm / sch_mm_per_uu
            sch_axes_top_uu = axes_top_mm_from_side_top / sch_mm_per_uu
            # tx_uu + sx * sch_axes_left_uu = slot_x_uu + vb_x
            tx_uu = slot_x_uu + vb_x - sx * sch_axes_left_uu
            ty_uu = slot_y_top_uu + vb_y - sy * sch_axes_top_uu

        # Build wrapper.
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
        return

    if embedded_figure is not None:
        # PR 6b: render the Figure to deterministic SVG bytes, parse
        # via lxml, treat like a vector schematic. Reuse
        # _resolve_svg_units to pull the schematic's mm size from its
        # own viewBox + width/height attrs.
        from publiplots.composer.compositing._embed import (
            render_figure_to_svg_bytes,
        )
        sch_bytes = render_figure_to_svg_bytes(embedded_figure)
        sch_element = etree.fromstring(sch_bytes)
        sch_vb_x, sch_vb_y, sch_vb_w, sch_vb_h, sch_mm_per_uu = (
            _resolve_svg_units(sch_element)
        )
        sch_size_mm = (sch_vb_w * sch_mm_per_uu, sch_vb_h * sch_mm_per_uu)
    elif path is None:
        # Unfilled PanelImage with no embedded figure — actionable error.
        raise ComposerVectorError(
            f"PanelImage {label_str!r} has no path and no embedded figure; "
            f"either pass path= at construction or call "
            f"canvas.embed_figure(label, fig) before savefig.",
            panel_label=label_str if label_str != "unlabeled" else None,
        )
    else:
        try:
            sch_element, sch_size_mm, sch_kind = load_schematic_as_svg_element(
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
                external_raster=external_raster,
                output_path=output_path,
                idx=idx,
                label=label_str,
            )
            sch_kind = "raster"

        # PR 6b external_raster: when the schematic loaded as a raster
        # data-URI, swap the inline href for a sidecar PNG reference.
        # Vector schematics (SVG) and embedded figures are unaffected.
        if (
            external_raster
            and sch_kind == "raster"
            and output_path is not None
        ):
            _swap_data_uri_for_sidecar(
                sch_element,
                source_path=path,
                output_path=output_path,
                idx=idx,
                label=label_str,
                etree=etree,
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
    external_raster: bool = False,
    output_path: Optional[Path] = None,
    idx: int = 0,
    label: str = "unlabeled",
) -> tuple:
    """Last-ditch raster fallback: build an ``<image>`` element.

    Pillow opens the file (regardless of extension); the returned
    ``<image>`` element is sized in mm matching the slot dimensions
    (so the wrapper's ``compute_svg_transform`` math is identity-safe).

    PR 6b: when ``external_raster=True`` and ``output_path`` is given,
    write a sidecar PNG instead of inlining a base64 data URI.

    Raises ``ComposerVectorError`` if Pillow can't open the file.
    """
    SVG_NS = "http://www.w3.org/2000/svg"
    XLINK_NS = "http://www.w3.org/1999/xlink"
    _, _, slot_w_mm, slot_h_mm = slot_bbox_mm_topdown
    image = etree.Element(
        f"{{{SVG_NS}}}image",
        nsmap={"xlink": XLINK_NS},
    )
    if external_raster and output_path is not None:
        sidecar = _write_sidecar_png(
            source_path=path,
            output_path=output_path,
            idx=idx,
            label=label,
        )
        image.set(f"{{{XLINK_NS}}}href", sidecar)
    else:
        from publiplots.composer.compositing._resources import (
            _pillow_to_data_uri,
        )
        image.set(f"{{{XLINK_NS}}}href", _pillow_to_data_uri(path))
    image.set("x", "0")
    image.set("y", "0")
    image.set("width", f"{slot_w_mm:.6f}")
    image.set("height", f"{slot_h_mm:.6f}")
    image.set("preserveAspectRatio", "none")
    return image, (slot_w_mm, slot_h_mm)


def _write_sidecar_png(
    *,
    source_path: Union[str, Path],
    output_path: Path,
    idx: int,
    label: str,
) -> str:
    """Write the raster source to a sidecar PNG; return its relative
    href for use in the SVG ``<image>`` element.

    The sidecar lives next to the SVG: ``<output_stem>-<idx>-<label>.png``.
    Uses Pillow to re-save (so even non-PNG sources land as PNG, matching
    the inline data-URI behavior).
    """
    try:
        from PIL import Image
    except ImportError as e:
        raise ComposerVectorError(
            "Pillow is required for external_raster sidecar emission. "
            "Install with `pip install publiplots[composer]`.",
            source_error=str(e),
        ) from e

    src = Path(source_path)
    sidecar_name = f"{output_path.stem}-{idx}-{label}.png"
    sidecar_path = output_path.parent / sidecar_name
    img = Image.open(src)
    img.load()
    img.save(sidecar_path, format="PNG")
    return sidecar_name


def _swap_data_uri_for_sidecar(
    sch_element: Any,
    *,
    source_path: Union[str, Path],
    output_path: Path,
    idx: int,
    label: str,
    etree: Any,
) -> None:
    """Mutate a raster ``<image>`` element returned by
    :func:`load_schematic_as_svg_element`: replace its inline data-URI
    ``xlink:href`` with a sidecar PNG reference.

    Used when the user passed ``external_raster=True`` and the
    schematic loaded as a raster source.
    """
    XLINK_NS = "http://www.w3.org/1999/xlink"
    sidecar_href = _write_sidecar_png(
        source_path=source_path,
        output_path=output_path,
        idx=idx,
        label=label,
    )
    # The element returned by load_schematic_as_svg_element for raster
    # sources is the <image> itself.
    sch_element.set(f"{{{XLINK_NS}}}href", sidecar_href)
    # Also clear any plain ``href`` (some lxml flows duplicate).
    if sch_element.get("href") is not None:
        del sch_element.attrib["href"]
