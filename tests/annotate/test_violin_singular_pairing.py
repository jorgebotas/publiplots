"""Violin records pair only with the bodies seaborn drew.

seaborn draws a group with fewer than two observations, or zero spread, as a
bare Line2D rather than a body. The builder used to pair records to bodies by
index, so every group after a singular one took its neighbour's body.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import publiplots as pp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _frame(singular: str, kind: str) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    groups = {"A": rng.normal(0, 1, 40), "B": rng.normal(10, 1, 40),
              "C": rng.normal(20, 1, 40)}
    groups[singular] = np.array([5.0]) if kind == "n1" else np.full(10, 5.0)
    return pd.DataFrame([{"cat": c, "v": float(v)}
                         for c, vs in groups.items() for v in vs])


@pytest.mark.parametrize("kind", ["n1", "flat"])
@pytest.mark.parametrize("singular", ["A", "B", "C"])
def test_violin_records_pair_with_their_own_bodies(singular, kind):
    fig, ax = pp.subplots()
    pp.violinplot(data=_frame(singular, kind), x="cat", y="v",
                  inner=None, legend=False, ax=ax)
    boxes = ax._publiplots_box_meta.boxes
    assert [b.category for b in boxes] == [c for c in "ABC" if c != singular]
    for b in boxes:
        assert b.center_pos == pytest.approx("ABC".index(b.category), abs=1e-6)


def test_violin_annotate_skips_the_singular_group():
    df = _frame("A", "n1")
    fig, ax = pp.subplots()
    pp.violinplot(data=df, x="cat", y="v", inner=None, legend=False, ax=ax,
                  annotate={"fmt": ".1f"})
    medians = df.groupby("cat")["v"].median()
    assert [t.get_text() for t in ax.texts] == [
        f"{medians['B']:.1f}", f"{medians['C']:.1f}"]


def test_boxplot_keeps_single_observation_groups():
    """matplotlib's bxp draws a box for n=1, so boxplot must not skip it."""
    fig, ax = pp.subplots()
    pp.boxplot(data=_frame("B", "n1"), x="cat", y="v", legend=False, ax=ax)
    boxes = ax._publiplots_box_meta.boxes
    assert [b.category for b in boxes] == ["A", "B", "C"]
    assert [b.center_pos for b in boxes] == pytest.approx([0.0, 1.0, 2.0])
