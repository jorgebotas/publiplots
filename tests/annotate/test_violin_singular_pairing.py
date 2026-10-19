"""Violin records pair only with the bodies seaborn drew.

seaborn draws a group as a bare Line2D rather than a body when it has fewer
than two observations, or when its density estimate cannot be fitted (in
practice, identical values whose covariance comes out exactly zero). The
builder used to pair records to bodies by index, so every group after a
singular one took its neighbour's body.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import publiplots as pp
from publiplots.annotate._builders import _is_singular


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


def test_constant_group_seaborn_fits_keeps_its_record():
    """21 x 5.0 has a nonzero float covariance, so seaborn draws a body."""
    rng = np.random.default_rng(1)
    groups = {"A": rng.normal(0, 1, 40), "B": np.full(21, 5.0),
              "C": rng.normal(20, 1, 40)}
    df = pd.DataFrame([{"cat": c, "v": float(v)}
                       for c, vs in groups.items() for v in vs])
    fig, ax = pp.subplots()
    pp.violinplot(data=df, x="cat", y="v", inner=None, legend=False, ax=ax)
    boxes = ax._publiplots_box_meta.boxes
    assert [b.category for b in boxes] == ["A", "B", "C"]
    assert [b.center_pos for b in boxes] == pytest.approx([0.0, 1.0, 2.0],
                                                          abs=1e-6)


def _hue_frame(constant_ay: bool) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    rows = []
    for ci, c in enumerate("AB"):
        for h in "xy":
            if constant_ay and (c, h) == ("A", "y"):
                vals = np.full(21, 5.0)
            else:
                vals = rng.normal(10 * ci, 1, 40)
            rows += [{"cat": c, "h": h, "v": float(v)} for v in vals]
    return pd.DataFrame(rows)


def _hue_centres(df: pd.DataFrame) -> dict:
    fig, ax = pp.subplots()
    pp.violinplot(data=df, x="cat", y="v", hue="h", inner=None,
                  legend=False, ax=ax)
    return {(b.category, b.hue_value): b.center_pos
            for b in ax._publiplots_box_meta.boxes}


def test_hue_constant_group_seaborn_fits_pairs_every_record():
    """A/y = 21 x 5.0 is drawn as a body, so no later record may shift."""
    reference = _hue_centres(_hue_frame(constant_ay=False))
    got = _hue_centres(_hue_frame(constant_ay=True))
    assert list(got) == [("A", "x"), ("A", "y"), ("B", "x"), ("B", "y")]
    for key, centre in got.items():
        assert centre == pytest.approx(reference[key], abs=1e-6), key


def test_is_singular_matches_seaborn_drawing_no_body():
    """Oracle: raw seaborn on a bare axes decides which constant groups get a
    body; ``_is_singular`` must agree for every size."""
    from matplotlib.collections import PolyCollection
    import seaborn as sns

    fig = plt.figure()
    mismatches = []
    for val in (5.0, 0.3):
        for n in range(2, 41):
            vals = np.full(n, val)
            ax = fig.add_subplot()
            sns.violinplot(y=vals, inner=None, ax=ax)
            no_body = not any(isinstance(c, PolyCollection)
                              for c in ax.collections)
            ax.remove()
            if _is_singular(vals) != no_body:
                mismatches.append((val, n, no_body))
    assert mismatches == []


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
