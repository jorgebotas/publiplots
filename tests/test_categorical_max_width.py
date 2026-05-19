"""Tests for the bar/box/violin max_width rcParams (mm-valued cap).

Each rcParam (``bar.max_width``, ``box.max_width``, ``violin.max_width``)
caps the rendered width on the categorical axis to a millimeter ceiling
while preserving each artist's center on that axis.
"""

from __future__ import annotations

import contextlib
from typing import Iterator

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import FillBetweenPolyCollection
from matplotlib.patches import PathPatch, Rectangle

import publiplots as pp
from publiplots.annotate._positioning import mm_to_data
from publiplots.utils.rounding import _RoundedBarPatch


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@contextlib.contextmanager
def _rc(**overrides) -> Iterator[None]:
    """Set publiplots rcParams for the duration of the with-block."""
    saved = {k: pp.rcParams[k] for k in overrides}
    try:
        for k, v in overrides.items():
            pp.rcParams[k] = v
        yield
    finally:
        for k, v in saved.items():
            pp.rcParams[k] = v


def _data_to_mm(width_data: float, ax, axis: str) -> float:
    """Inverse of mm_to_data: convert a data-coord delta to mm."""
    one_mm = mm_to_data(1.0, ax, axis)
    return float(width_data) / float(one_mm)


# ---------------------------------------------------------------------------
# Bars
# ---------------------------------------------------------------------------


def _bars(ax):
    return [p for p in ax.patches if hasattr(p, "get_height")]


class TestBarMaxWidth:
    """`bar.max_width` clamps and centers vertical/horizontal bars."""

    def _df_two_cats(self):
        df = pd.DataFrame({"x": ["A", "B"], "y": [3.0, 2.0]})
        df["x"] = pd.Categorical(df["x"], categories=["A", "B"])
        return df

    def test_cap_fires(self):
        df = self._df_two_cats()
        fig, ax = pp.subplots(axes_size=(60, 40))
        with _rc(**{"bar.max_width": 5.0}):
            pp.barplot(data=df, x="x", y="y", ax=ax)
            fig.canvas.draw()
            for p in _bars(ax):
                w_mm = _data_to_mm(p.get_width(), ax, "x")
                assert w_mm <= 5.0 + 1e-3, f"bar width {w_mm:.3f}mm > 5mm cap"

    def test_cap_is_ceiling(self):
        # 5 categories on a 60mm axes — natural slot is 12mm/bar, well below
        # a 50mm cap. Widths should match the un-capped baseline.
        df = pd.DataFrame({"x": list("abcde"), "y": [1.0, 2, 3, 4, 5]})
        df["x"] = pd.Categorical(df["x"], categories=list("abcde"))

        fig, ax = pp.subplots(axes_size=(60, 40))
        pp.barplot(data=df, x="x", y="y", ax=ax)
        fig.canvas.draw()
        baseline = sorted(p.get_width() for p in _bars(ax))

        fig2, ax2 = pp.subplots(axes_size=(60, 40))
        with _rc(**{"bar.max_width": 50.0}):
            pp.barplot(data=df, x="x", y="y", ax=ax2)
            fig2.canvas.draw()
            capped = sorted(p.get_width() for p in _bars(ax2))

        for b, c in zip(baseline, capped):
            assert abs(b - c) < 1e-9

    def test_center_preserved(self):
        df = self._df_two_cats()
        fig, ax = pp.subplots(axes_size=(80, 40))
        with _rc(**{"bar.max_width": 5.0}):
            pp.barplot(data=df, x="x", y="y", ax=ax)
            fig.canvas.draw()
            # Two cats land at integer x positions 0 and 1 in seaborn.
            centers = sorted(p.get_x() + p.get_width() / 2.0 for p in _bars(ax))
            tol_mm = 0.5
            tol_data = abs(mm_to_data(tol_mm, ax, "x"))
            assert abs(centers[0] - 0.0) < tol_data
            assert abs(centers[1] - 1.0) < tol_data

    def test_composes_with_border_radius(self):
        df = self._df_two_cats()
        fig, ax = pp.subplots(axes_size=(80, 40))
        with _rc(**{"bar.max_width": 5.0, "bar.border_radius": (1.0, 0.0)}):
            pp.barplot(data=df, x="x", y="y", ax=ax)
            fig.canvas.draw()
            rounded = [p for p in _bars(ax) if isinstance(p, _RoundedBarPatch)]
            assert len(rounded) == 2
            for p in rounded:
                w_mm = _data_to_mm(p.get_width(), ax, "x")
                assert w_mm <= 5.0 + 1e-3

    def test_horizontal_bars_clamped_on_y(self):
        df = pd.DataFrame({"y": ["A", "B"], "x": [3.0, 2.0]})
        df["y"] = pd.Categorical(df["y"], categories=["A", "B"])
        fig, ax = pp.subplots(axes_size=(40, 60))
        with _rc(**{"bar.max_width": 5.0}):
            pp.barplot(data=df, x="x", y="y", ax=ax)
            fig.canvas.draw()
            for p in _bars(ax):
                h_mm = _data_to_mm(p.get_height(), ax, "y")
                assert h_mm <= 5.0 + 1e-3

    def test_no_cap_default(self):
        df = self._df_two_cats()
        fig, ax = pp.subplots(axes_size=(60, 40))
        pp.barplot(data=df, x="x", y="y", ax=ax)
        fig.canvas.draw()
        # No cap → bars span their full slot (slot - gap, in data units).
        # On a 60mm/2-bar axes a 0.72 data-coord bar corresponds to ~21mm;
        # the ceiling test elsewhere already guards "cap doesn't shrink".
        # Just confirm uncapped widths exceed any reasonable mm cap.
        for p in _bars(ax):
            w_mm = _data_to_mm(p.get_width(), ax, "x")
            assert w_mm > 10.0, f"uncapped bar should be wide; got {w_mm:.3f}mm"


# ---------------------------------------------------------------------------
# Boxes
# ---------------------------------------------------------------------------


def _boxes(ax):
    return [p for p in ax.patches if isinstance(p, PathPatch)]


def _box_extent(p: PathPatch, axis: str) -> float:
    bbox = p.get_path().get_extents()
    return bbox.width if axis == "x" else bbox.height


def _box_center(p: PathPatch, axis: str) -> float:
    bbox = p.get_path().get_extents()
    return (bbox.x0 + bbox.x1) / 2.0 if axis == "x" else (bbox.y0 + bbox.y1) / 2.0


class TestBoxMaxWidth:
    """`box.max_width` clamps boxplot IQR boxes."""

    def _df(self):
        rng = np.random.default_rng(0)
        rows = []
        for cat in ("A", "B"):
            for v in rng.normal(size=30):
                rows.append({"g": cat, "y": float(v)})
        df = pd.DataFrame(rows)
        df["g"] = pd.Categorical(df["g"], categories=["A", "B"])
        return df

    def test_cap_fires(self):
        df = self._df()
        fig, ax = pp.subplots(axes_size=(60, 40))
        with _rc(**{"box.max_width": 5.0}):
            pp.boxplot(data=df, x="g", y="y", ax=ax)
            fig.canvas.draw()
            for p in _boxes(ax):
                w_mm = _data_to_mm(_box_extent(p, "x"), ax, "x")
                assert w_mm <= 5.0 + 1e-3

    def test_cap_is_ceiling(self):
        # 5 cats on 60mm — natural width well below 50mm cap.
        rng = np.random.default_rng(1)
        rows = []
        for cat in list("abcde"):
            for v in rng.normal(size=30):
                rows.append({"g": cat, "y": float(v)})
        df = pd.DataFrame(rows)
        df["g"] = pd.Categorical(df["g"], categories=list("abcde"))

        fig, ax = pp.subplots(axes_size=(60, 40))
        pp.boxplot(data=df, x="g", y="y", ax=ax)
        fig.canvas.draw()
        baseline = sorted(_box_extent(p, "x") for p in _boxes(ax))

        fig2, ax2 = pp.subplots(axes_size=(60, 40))
        with _rc(**{"box.max_width": 50.0}):
            pp.boxplot(data=df, x="g", y="y", ax=ax2)
            fig2.canvas.draw()
            capped = sorted(_box_extent(p, "x") for p in _boxes(ax2))

        for b, c in zip(baseline, capped):
            assert abs(b - c) < 1e-9

    def test_center_preserved(self):
        df = self._df()
        fig, ax = pp.subplots(axes_size=(80, 40))
        with _rc(**{"box.max_width": 5.0}):
            pp.boxplot(data=df, x="g", y="y", ax=ax)
            fig.canvas.draw()
            centers = sorted(_box_center(p, "x") for p in _boxes(ax))
            tol_data = abs(mm_to_data(0.5, ax, "x"))
            assert abs(centers[0] - 0.0) < tol_data
            assert abs(centers[1] - 1.0) < tol_data

    def test_composes_with_border_radius(self):
        df = self._df()
        fig, ax = pp.subplots(axes_size=(80, 40))
        with _rc(**{"box.max_width": 5.0, "box.border_radius": (1.0, 0.0)}):
            pp.boxplot(data=df, x="g", y="y", ax=ax)
            fig.canvas.draw()
            rounded = [p for p in ax.patches if isinstance(p, _RoundedBarPatch)]
            assert len(rounded) == 2
            for p in rounded:
                w_mm = _data_to_mm(p.get_width(), ax, "x")
                assert w_mm <= 5.0 + 1e-3


# ---------------------------------------------------------------------------
# Violins
# ---------------------------------------------------------------------------


def _violin_collections(ax):
    return [c for c in ax.collections if isinstance(c, FillBetweenPolyCollection)]


def _violin_x_range(coll: FillBetweenPolyCollection):
    """Return (vmin, vmax) on x for each path's vertices."""
    out = []
    for path in coll.get_paths():
        xs = path.vertices[:, 0]
        out.append((float(xs.min()), float(xs.max())))
    return out


class TestViolinMaxWidth:
    """`violin.max_width` clamps violin half-width on the categorical axis."""

    def _df(self):
        rng = np.random.default_rng(2)
        rows = []
        for cat in ("A", "B"):
            for v in rng.normal(size=200):
                rows.append({"g": cat, "y": float(v)})
        df = pd.DataFrame(rows)
        df["g"] = pd.Categorical(df["g"], categories=["A", "B"])
        return df

    def test_cap_fires(self):
        df = self._df()
        fig, ax = pp.subplots(axes_size=(60, 40))
        with _rc(**{"violin.max_width": 5.0}):
            pp.violinplot(data=df, x="g", y="y", ax=ax)
            fig.canvas.draw()
            for coll in _violin_collections(ax):
                for vmin, vmax in _violin_x_range(coll):
                    w_mm = _data_to_mm(vmax - vmin, ax, "x")
                    assert w_mm <= 5.0 + 1e-3, f"violin width {w_mm:.3f}mm > 5mm"

    def test_cap_is_ceiling(self):
        rng = np.random.default_rng(3)
        rows = []
        for cat in list("abcde"):
            for v in rng.normal(size=80):
                rows.append({"g": cat, "y": float(v)})
        df = pd.DataFrame(rows)
        df["g"] = pd.Categorical(df["g"], categories=list("abcde"))

        fig, ax = pp.subplots(axes_size=(60, 40))
        pp.violinplot(data=df, x="g", y="y", ax=ax)
        fig.canvas.draw()
        baseline = sorted(
            (vmax - vmin)
            for coll in _violin_collections(ax)
            for vmin, vmax in _violin_x_range(coll)
        )

        fig2, ax2 = pp.subplots(axes_size=(60, 40))
        with _rc(**{"violin.max_width": 50.0}):
            pp.violinplot(data=df, x="g", y="y", ax=ax2)
            fig2.canvas.draw()
            capped = sorted(
                (vmax - vmin)
                for coll in _violin_collections(ax2)
                for vmin, vmax in _violin_x_range(coll)
            )

        for b, c in zip(baseline, capped):
            assert abs(b - c) < 1e-9

    def test_center_preserved(self):
        df = self._df()
        fig, ax = pp.subplots(axes_size=(80, 40))
        with _rc(**{"violin.max_width": 5.0}):
            pp.violinplot(data=df, x="g", y="y", ax=ax)
            fig.canvas.draw()
            centers = []
            for coll in _violin_collections(ax):
                for vmin, vmax in _violin_x_range(coll):
                    centers.append((vmin + vmax) / 2.0)
            centers.sort()
            tol_data = abs(mm_to_data(0.5, ax, "x"))
            assert abs(centers[0] - 0.0) < tol_data
            assert abs(centers[1] - 1.0) < tol_data

    def test_composes_with_side_left(self):
        df = self._df()
        fig, ax = pp.subplots(axes_size=(80, 40))
        max_mm = 4.0
        with _rc(**{"violin.max_width": max_mm}):
            pp.violinplot(data=df, x="g", y="y", side="left", ax=ax)
            fig.canvas.draw()
            # Left-clipped: the right edge of each path is the category center.
            cats = [0.0, 1.0]
            paths_seen = []
            for coll in _violin_collections(ax):
                for vmin, vmax in _violin_x_range(coll):
                    paths_seen.append((vmin, vmax))
            # Each path's right edge sits at one of the category centers.
            tol_data = abs(mm_to_data(0.2, ax, "x"))
            max_data = abs(mm_to_data(max_mm, ax, "x"))
            for vmin, vmax in paths_seen:
                # Edge should be at a category center.
                edge_at_cat = any(abs(vmax - c) < tol_data for c in cats)
                assert edge_at_cat, (
                    f"left-clipped violin's right edge {vmax:.3f} not at a category center"
                )
                # Half-width within max_data + tolerance.
                assert (vmax - vmin) <= max_data + tol_data
