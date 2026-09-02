"""Rounding a patch must not move it within the axes (issue #236).

``apply_border_radius`` replaces each patch with a ``_RoundedBarPatch`` via
``remove()`` + ``add_patch()``, which appends. Any patch the conversion
skips therefore keeps its original slot while its converted siblings move
to the end, permuting ``ax.patches``. Two things break: the annotate meta
builders pair records to patches by draw order, so labels land on the wrong
box; and overlapping artists that share a zorder swap paint order.

Every test below fails without the fix, except the one control and the
two guards in the final section.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.patches import Circle, Rectangle

import publiplots as pp
from publiplots.utils.rounding import _RoundedBarPatch, apply_border_radius


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _zero_iqr_df():
    """Group B is constant, so seaborn draws it as a degenerate box."""
    return pd.DataFrame({
        "cat": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
        "v": [1, 2, 3, 4, 5] + [7] * 5 + [1, 3, 5, 7, 9],
    })


# ---- the reported symptom ----

def test_boxplot_meta_pairing_survives_border_radius():
    """A zero-IQR box must not swap places with its neighbour."""
    fig, ax = pp.subplots()
    pp.boxplot(data=_zero_iqr_df(), x="cat", y="v",
               border_radius=1.5, legend=False, ax=ax)

    meta = ax._publiplots_box_meta
    assert [b.category for b in meta.boxes] == ["A", "B", "C"]
    assert [b.center_pos for b in meta.boxes] == pytest.approx(
        [0.0, 1.0, 2.0], abs=1e-9)


def test_boxplot_meta_pairing_without_border_radius_is_the_control():
    """Control: the default (0, 0) radius short-circuits before any swap.

    Passes with or without the fix — it exists to show the divergence is
    the rounding pass and not the box builder.
    """
    fig, ax = pp.subplots()
    pp.boxplot(data=_zero_iqr_df(), x="cat", y="v", legend=False, ax=ax)

    meta = ax._publiplots_box_meta
    assert [b.category for b in meta.boxes] == ["A", "B", "C"]
    assert [b.center_pos for b in meta.boxes] == pytest.approx(
        [0.0, 1.0, 2.0], abs=1e-9)


def test_boxplot_border_radius_preserves_patch_order():
    """The underlying invariant: rounding permutes nothing in ax.patches."""
    df = _zero_iqr_df()

    fig, ax_plain = pp.subplots()
    pp.boxplot(data=df, x="cat", y="v", legend=False, ax=ax_plain)
    plain = [p.get_path().get_extents().x0 for p in ax_plain.patches]

    fig2, ax_round = pp.subplots()
    pp.boxplot(data=df, x="cat", y="v", border_radius=1.5,
               legend=False, ax=ax_round)
    rounded = [p.get_path().get_extents().x0 for p in ax_round.patches]

    assert len(plain) == len(rounded)
    for a, b in zip(plain, rounded):
        assert a == pytest.approx(b, abs=1e-6), (
            f"patch order changed under border_radius: {plain} vs {rounded}"
        )


def test_zero_iqr_hue_level_keeps_every_box_paired():
    """A whole hue level that is constant, under dodge.

    Stronger than the flat case: all four boxes are mislabelled without the
    fix, and it proves the pairing holds once dodging puts two boxes inside
    each category.
    """
    df = pd.DataFrame({
        "cat": list("AABB") * 5,
        "h": list("xy") * 10,
        "v": [1, 5, 2, 5, 3, 5, 4, 5, 5, 5, 6, 5, 7, 5, 8, 5, 9, 5, 10, 5],
    })
    fig, ax = pp.subplots()
    pp.boxplot(data=df, x="cat", y="v", hue="h", border_radius=1.5,
               legend=False, ax=ax)

    got = [(b.category, b.hue_value) for b in ax._publiplots_box_meta.boxes]
    assert got == [("A", "x"), ("B", "x"), ("A", "y"), ("B", "y")]

    centers = [b.center_pos for b in ax._publiplots_box_meta.boxes]
    assert centers == pytest.approx([-0.2, 0.8, 0.2, 1.2], abs=1e-6)


# ---- the general invariant, unit-tested where the public API can't reach ----

@pytest.mark.parametrize("pattern", ["RCR", "RRC", "RRCC", "RCCR", "CRCRC"])
def test_skipped_patch_kinds_do_not_permute_the_rest(pattern):
    """Interleaved convertible and skipped patches keep their slots.

    The ``else: continue`` branch cannot be reached through ``pp.*`` — the
    bar path only ever hands `apply_border_radius` its own Rectangles — so
    the general invariant is unit-tested directly. Without the fix each of
    these patterns comes back permuted.
    """
    fig, ax = plt.subplots()
    added = []
    for i, kind in enumerate(pattern):
        p = (Rectangle((i, 0), 0.8, 1.0) if kind == "R"
             else Circle((i, 0), 0.1))
        p.set_label(str(i))
        ax.add_patch(p)
        added.append(p)

    apply_border_radius(added, (1.5, 1.5), ax, orient="v")

    assert [p.get_label() for p in ax.patches] == list("01234"[:len(pattern)])


def test_rounding_preserves_paint_order_of_overlapping_artists():
    """Appending a replacement also inverted z-order among equal zorders.

    Matplotlib paints equal-zorder artists in child order, so moving a
    rounded patch to the end put it on top of something drawn after it.
    The rounded render must match the un-rounded baseline.
    """
    def render(radius):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.set_axis_off()
        rect = Rectangle((0, 0), 2, 2, facecolor="red", zorder=1)
        ax.add_patch(rect)
        circle = Circle((1, 1), 0.5, facecolor="blue", zorder=1)
        ax.add_patch(circle)
        if radius:
            apply_border_radius([rect], (radius, radius), ax, orient="v")
        fig.canvas.draw()
        # buffer_rgba() is a memoryview and cannot be sub-sliced; wrap it.
        buf = np.asarray(fig.canvas.buffer_rgba())
        h, w = buf.shape[0], buf.shape[1]
        centre = tuple(int(c) for c in buf[h // 2, w // 2][:3])
        plt.close(fig)
        return centre

    baseline = render(0)
    assert baseline == (0, 0, 255), "precondition: blue circle paints on top"
    assert render(1.5) == baseline, (
        "rounding the rectangle moved it above the circle drawn after it"
    )


# ---- guards: the fix must not stop it doing its job ----

def test_boxplot_border_radius_still_rounds_the_healthy_boxes():
    """Guard, not a regression test: order preservation must still round."""
    fig, ax = pp.subplots()
    pp.boxplot(data=_zero_iqr_df(), x="cat", y="v", border_radius=1.5,
               legend=False, ax=ax)

    rounded = [p for p in ax.patches if isinstance(p, _RoundedBarPatch)]
    # A and C are healthy and get rounded; B is degenerate and is left alone.
    assert len(rounded) == 2


def test_rounding_twice_is_idempotent():
    """A second pass must not double-round or reorder.

    ``_RoundedBarPatch`` is neither a ``Rectangle`` nor a ``PathPatch``, so
    the second pass skips every patch — which is exactly the skip branch
    that used to permute.
    """
    df = pd.DataFrame({"cat": list("ABC"), "val": [1.0, 2.0, 3.0]})
    fig, ax = pp.subplots()
    pp.barplot(data=df, x="cat", y="val", border_radius=1.5,
               legend=False, ax=ax)

    before = [(id(p), p.get_path().vertices.shape) for p in ax.patches]
    apply_border_radius(list(ax.patches), (1.5, 1.5), ax, orient="v")
    after = [(id(p), p.get_path().vertices.shape) for p in ax.patches]

    assert before == after
