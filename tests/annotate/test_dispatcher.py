"""Tests for the annotate() dispatcher: validation and routing."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from publiplots.annotate import annotate


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_unknown_kind_raises_valueerror_with_known_list():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="unknown kind"):
        annotate(ax, kind="not_a_kind")


def test_invalid_anchor_raises():
    fig, ax = plt.subplots()
    ax.bar([0], [1.0])
    with pytest.raises(ValueError, match="anchor"):
        annotate(ax, kind="bar_values", anchor="sideways")


def test_negative_offset_raises():
    fig, ax = plt.subplots()
    ax.bar([0], [1.0])
    with pytest.raises(ValueError, match=">= 0"):
        annotate(ax, kind="bar_values", offset=-1)


def test_negative_pad_raises():
    fig, ax = plt.subplots()
    ax.bar([0], [1.0])
    with pytest.raises(ValueError, match=">= 0"):
        annotate(ax, kind="bar_values", pad=-0.5)


def test_default_kind_is_bar_values():
    fig, ax = plt.subplots()
    ax.bar([0, 1], [1.0, 2.0])
    fig.canvas.draw()
    texts = annotate(ax)  # no kind arg
    assert len(texts) == 2


def test_registry_contains_known_strategies():
    from publiplots.annotate._dispatcher import _STRATEGIES
    assert set(_STRATEGIES.keys()) == {"bar_values", "point_values"}


def test_text_kws_forwarded():
    fig, ax = plt.subplots()
    ax.bar([0], [1.0])
    fig.canvas.draw()
    texts = annotate(ax, kind="bar_values", fontsize=14, fontweight="bold")
    assert texts[0].get_fontsize() == 14
    assert texts[0].get_fontweight() == "bold"


def test_annotate_is_exposed_on_pp():
    import publiplots as pp
    assert pp.annotate is annotate
