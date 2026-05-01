"""Integration: pp.barplot(..., annotate=...) cache-building + end-to-end."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import publiplots as pp
from publiplots.annotate._cache import BarValueMeta


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _simple_df():
    return pd.DataFrame({
        "category": pd.Categorical(["A", "B", "C"]),
        "value": [1.0, 2.0, 3.0],
    })


def _grouped_df():
    rng = np.random.default_rng(0)
    rows = []
    for group in ("A", "B"):
        for cond in ("ctrl", "trt"):
            for v in rng.normal(loc=(1 if group == "A" else 2) + (0 if cond == "ctrl" else 1),
                                scale=0.1, size=8):
                rows.append({"group": group, "cond": cond, "y": float(v)})
    df = pd.DataFrame(rows)
    df["group"] = df["group"].astype("category")
    df["cond"] = df["cond"].astype("category")
    return df


def test_barplot_annotate_false_attaches_no_meta():
    fig, ax = pp.barplot(data=_simple_df(), x="category", y="value")
    assert not hasattr(ax, "_publiplots_bar_meta")


def test_barplot_annotate_true_attaches_meta_and_draws_labels():
    fig, ax = pp.barplot(data=_simple_df(), x="category", y="value", annotate=True)
    assert isinstance(ax._publiplots_bar_meta, BarValueMeta)
    assert ax._publiplots_bar_meta.owner_is_publiplots is True
    texts = [t for t in ax.texts]
    assert len(texts) == 3


def test_barplot_annotate_dict_forwarded():
    fig, ax = pp.barplot(data=_simple_df(), x="category", y="value",
                         annotate={"fmt": ".3f"})
    labels = [t.get_text() for t in ax.texts]
    assert labels == ["1.000", "2.000", "3.000"]


def test_barplot_annotate_with_hue_has_hue_active():
    fig, ax = pp.barplot(data=_grouped_df(), x="group", y="y", hue="cond",
                         annotate=True)
    meta = ax._publiplots_bar_meta
    assert meta.hue_active is True
    assert all(b.hue_color is not None for b in meta.bars)


def test_barplot_annotate_no_hue_hue_active_false():
    fig, ax = pp.barplot(data=_simple_df(), x="category", y="value", annotate=True)
    assert ax._publiplots_bar_meta.hue_active is False


def test_barplot_annotate_expands_ylim():
    df = pd.DataFrame({
        "category": pd.Categorical(["A", "B"]),
        "value": [10.0, 10.0],
    })
    fig, ax = pp.barplot(data=df, x="category", y="value", annotate=True)
    top = ax.get_ylim()[1]
    # seaborn's default would stop at ~10 + a small pad; with annotate the
    # label must fit above that, so top should be noticeably above 10.
    assert top > 10.0


def test_barplot_annotate_with_errorbars_anchors_past_cap():
    """With errorbar='se' and annotate=True, label y should sit above the cap."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "group": pd.Categorical(np.repeat(["A", "B"], 20)),
        "y": np.concatenate([
            rng.normal(1.0, 0.5, 20),
            rng.normal(2.0, 0.5, 20),
        ]),
    })
    fig, ax = pp.barplot(data=df, x="group", y="y", errorbar="se", annotate=True)
    meta = ax._publiplots_bar_meta
    # Each bar has an err_high > value (standard error extends above mean).
    for bar in meta.bars:
        assert bar.err_high is not None
        assert bar.err_high > bar.value
    # Corresponding label y coordinates should be at or above err_high.
    for bar, text in zip(meta.bars, ax.texts):
        _, y = text.get_position()
        assert y >= bar.err_high


def test_barplot_annotate_hue_labels_pair_to_correct_bars():
    """Labels must match bar heights, not swap across dodge groups.

    With asymmetric heights, a wrong loop order swaps labels silently.
    """
    import numpy as np
    df = pd.DataFrame({
        "g": pd.Categorical(["A", "B", "A", "B"]),
        "c": pd.Categorical(["ctrl", "ctrl", "trt", "trt"]),
        "y": [1.0, 2.0, 10.0, 20.0],
    })
    fig, ax = pp.barplot(data=df, x="g", y="y", hue="c",
                         errorbar=None, annotate={"fmt": ".1f"})
    # Sort texts left-to-right by x position; seaborn draws hue-outer, cat-inner
    # so order is: ctrl-A=1, ctrl-B=2, trt-A=10, trt-B=20.
    by_x = sorted(ax.texts, key=lambda t: t.get_position()[0])
    labels = [t.get_text() for t in by_x]
    assert labels == ["1.0", "10.0", "2.0", "20.0"], (
        f"Got {labels}; bar-label pairing is broken."
    )


def test_barplot_annotate_errorbar_ci_pulls_from_drawn_artists():
    """For errorbar='ci' (bootstrap percentile), err_high should come
    from the drawn cap, not from a re-aggregated normal approx."""
    import numpy as np
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "g": pd.Categorical(np.repeat(["A", "B"], 50)),
        "y": np.concatenate([
            rng.normal(1.0, 0.5, 50),
            rng.normal(2.0, 0.5, 50),
        ]),
    })
    fig, ax = pp.barplot(data=df, x="g", y="y", errorbar="ci", annotate=True)
    meta = ax._publiplots_bar_meta
    for bar in meta.bars:
        assert bar.err_high is not None
        # err_high must come from the drawn cap (not silent 0.0 fallback nor wrong math)
        assert bar.err_high > bar.value


def test_barplot_annotate_errorbar_none_all_none():
    df = pd.DataFrame({
        "g": pd.Categorical(["A", "B"]),
        "y": [1.0, 2.0],
    })
    fig, ax = pp.barplot(data=df, x="g", y="y", errorbar=None, annotate=True)
    meta = ax._publiplots_bar_meta
    for bar in meta.bars:
        assert bar.err_low is None
        assert bar.err_high is None


def test_barplot_annotate_color_hue_without_hue_warns():
    """annotate={'color': 'hue'} on a no-hue barplot warns and falls back."""
    df = pd.DataFrame({
        "category": pd.Categorical(["A", "B", "C"]),
        "value": [1.0, 2.0, 3.0],
    })
    with pytest.warns(UserWarning, match="plot has no hue"):
        fig, ax = pp.barplot(data=df, x="category", y="value",
                             annotate={"color": "hue"})
