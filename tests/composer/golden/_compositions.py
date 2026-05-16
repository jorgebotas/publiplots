"""Canonical composition registry for golden-fixture tests.

Each entry: ``(name: str, build_fn: Callable[[], pp.Canvas])``. Build
functions construct a :class:`pp.Canvas` deterministically — no rng,
no env reads, no time-dependent inputs. Snapshots taken from these
canvases are committed under ``tests/composer/golden/`` and gated in
``test_mm_snapshots.py`` + ``test_png_regression.py``.

Add a new composition by appending a ``(name, build_fn)`` tuple to
``COMPOSITIONS``, then running ``python tools/composer/regen_fixtures.py``
to write the JSON + PNG fixtures.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Tuple

import publiplots as pp


def _build_cell_2col_simple() -> pp.Canvas:
    """Single row, two PanelAxes (one pinned + one flex). Mirrors examples/composer/cell_2col_simple.py."""
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelAxes(label=None, size=(70, 50)),       # → 'A'
        pp.PanelAxes(label=None, size=("flex", 50)),   # → 'B'
    )
    return canvas


def _build_cell_2col_multirow() -> pp.Canvas:
    """Two rows; A+B+C+D abc across rows; PanelText. Mirrors examples/composer/cell_2col_multirow.py.

    NOTE: uses `canvas.align(["A", "C"], edge="left")` like the example;
    align() must be called BEFORE any panel access (which would trigger
    finalization).
    """
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70, 40)),
        pp.PanelAxes(label="B", size=("flex", 40)),
    )
    canvas.add_row(
        pp.PanelAxes(label="C", size=(70, 40)),
        pp.PanelText(label="D", text="n = 1,234\nP < 0.001",
                     size=("flex", 40)),
    )
    canvas.align(["A", "C"], edge="left")
    return canvas


def _build_nature_2col_abc() -> pp.Canvas:
    """3 flex panels with mix of auto + verbatim labels + per-panel
    label_style. Mirrors examples/composer/nature_2col_with_abc.py.

    Exercises: flex sizing across all 3 panels, label=None auto-letter,
    label='b.i' verbatim, per-panel label_style override.
    """
    canvas = pp.Canvas("nature-2col")  # abc='lower' is the preset default
    canvas.add_row(
        pp.PanelAxes(label=None, size=("flex", 40)),                          # → 'a'
        pp.PanelAxes(label="b.i", size=("flex", 40),
                     label_style={"size": 10}),                               # custom label + size
        pp.PanelAxes(label=None, size=("flex", 40)),                          # → 'c' (b consumed)
    )
    return canvas


def _build_nature_2col_panel_grid() -> pp.Canvas:
    """PanelAxes + PanelGrid on row 0; PanelText caption on row 1.
    Mirrors examples/composer/nature_2col_panel_grid.py.

    Width budget: 183mm canvas. Panel 'a' = 60mm; PanelGrid cells = 28mm
    gives 28*3 + 2*2 = 88mm grid width. Decorations + pads ~31mm.
    Total ~179mm. Fits.
    """
    canvas = pp.Canvas("nature-2col")
    canvas.add_row(
        pp.PanelAxes(label="a", size=(60, 40)),
        pp.PanelGrid(label="b", shape=(1, 3), axes_size=(28, 40)),
    )
    canvas.add_row(
        pp.PanelText(label="c",
                     text=r"Linear regression on 100 samples ($\alpha = 0.05$)",
                     size=("flex", 8)),
    )
    return canvas


# Fixtures for PR 5 PanelImage compositions live under
# `tests/composer/golden/fixtures/`. The build_fns below use absolute paths
# resolved relative to this module so the goldens are reproducible from
# any CWD.
_FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _build_cell_2col_with_svg_schematic() -> pp.Canvas:
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelImage(label="A", path=_FIXTURES_DIR / "schematic.svg",
                      size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )
    return canvas


def _build_cell_2col_with_png_schematic() -> pp.Canvas:
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelImage(label="A", path=_FIXTURES_DIR / "schematic.png",
                      size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )
    return canvas


def _build_cell_2col_with_embed_figure() -> pp.Canvas:
    """PR 6c headline composition: PanelAxes scatter + embed_figure of a
    pp.subplots-built lineplot anchored by axes-data box.

    Demonstrates the "kitchen sink" use case from the design spec:
    the right slot is filled at compose time by a separately-constructed
    matplotlib Figure rather than a path-based schematic. Uses
    ``anchor='axes'`` (PR 6c) with ``axes_size=(70, 50)`` matching
    Panel B's slot dims so the side figure's axes-data box aligns
    1:1 with the slot rect (paper-figure axis alignment).
    """
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70, 50)),
        pp.PanelImage(label="B", size=(70, 50)),
    )
    # Build a deterministic side figure using pp.subplots (no rng,
    # no time, no env).
    fig, ax = pp.subplots(axes_size=(70, 50))
    ax.plot([1, 2, 3], [4, 5, 6])
    canvas.embed_figure("B", fig, anchor="axes")
    return canvas


COMPOSITIONS: List[Tuple[str, Callable[[], pp.Canvas]]] = [
    ("cell-2col-simple", _build_cell_2col_simple),
    ("cell-2col-multirow", _build_cell_2col_multirow),
    ("nature-2col-abc", _build_nature_2col_abc),
    ("nature-2col-panel-grid", _build_nature_2col_panel_grid),
    ("cell-2col-with-svg-schematic", _build_cell_2col_with_svg_schematic),
    ("cell-2col-with-png-schematic", _build_cell_2col_with_png_schematic),
    ("cell-2col-with-embed-figure", _build_cell_2col_with_embed_figure),
]
