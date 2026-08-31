"""Regression tests: saved raster output must actually contain ink.

Guards the class of bug where the layout engine resizes the figure from
inside ``draw_event`` while an output render is in progress. Agg keys its
renderer cache on ``figure.bbox.size``, so a mid-render resize discards
the renderer that was just drawn into and the file writer receives a
freshly allocated, never-drawn (fully transparent) buffer -- a completely
blank image, written without any error.

Every other layout test asserts on geometry or on artist state, all of
which stay perfectly consistent while this happens. Only reading the
saved pixels back catches it, so that is what these tests do.

``savefig.transparent=True`` is a publiplots default, so the assertion is
on the alpha channel: a blank file is not white, it is transparent.

``plt.imread`` is used rather than Pillow -- matplotlib reads PNG natively
and Pillow is not a test dependency of this project.
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


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "x": rng.normal(size=80),
        "y": rng.normal(size=80),
        "g": rng.choice(["one", "two", "three"], size=80),
    })


# Minimum acceptable fraction of non-transparent pixels. Healthy figures
# in this file measure 0.09-0.18, and the bug produces exactly 0.0, so any
# threshold in between discriminates. A real floor rather than ``> 0.0``
# also catches a partially cropped or clipped band, which a
# single-non-transparent-pixel test would pass.
_MIN_INK = 0.02


def _ink_fraction(path):
    """Fraction of pixels in the saved PNG with non-zero alpha."""
    img = plt.imread(str(path))
    assert img.ndim == 3 and img.shape[2] == 4, (
        f"expected RGBA (transparent=True is a publiplots default), got {img.shape}"
    )
    return float((img[..., 3] > 0).sum()) / (img.shape[0] * img.shape[1])


def _save_and_measure(tmp_path, name, dpi=None):
    path = tmp_path / f"{name}.png"
    if dpi is None:
        pp.savefig(str(path))          # publiplots' own default dpi
    else:
        pp.savefig(str(path), dpi=dpi)
    assert path.exists()
    return _ink_fraction(path)


def _figure_level_bottom_legend(df):
    fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
    pp.scatterplot(data=df, x="x", y="y", hue="g", ax=ax)
    pp.legend(side="bottom")
    return fig


def _per_axes_bottom_legend(df):
    fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
    pp.scatterplot(
        data=df, x="x", y="y", hue="g", ax=ax, legend_kws={"side": "bottom"}
    )
    return fig


def test_figure_level_bottom_legend_png_has_ink_at_default_dpi(df, tmp_path):
    """pp.legend(side='bottom') + pp.savefig() at the DEFAULT dpi.

    This is the exact call the blocker reproduced with: the file was
    written with zero non-transparent pixels at ``savefig.dpi=600``.
    """
    _figure_level_bottom_legend(df)
    assert _save_and_measure(tmp_path, "figure_bottom") > _MIN_INK


def test_per_axes_bottom_legend_png_has_ink_at_default_dpi(df, tmp_path):
    """legend_kws={'side': 'bottom'} + pp.savefig() at the DEFAULT dpi."""
    _per_axes_bottom_legend(df)
    assert _save_and_measure(tmp_path, "per_axes_bottom") > _MIN_INK


@pytest.mark.parametrize("dpi", [100, 150, 250, 300, 600])
@pytest.mark.parametrize(
    "build", [_figure_level_bottom_legend, _per_axes_bottom_legend],
    ids=["figure_level", "per_axes"],
)
def test_bottom_legend_png_has_ink_across_dpis(df, tmp_path, build, dpi):
    """A fix must not merely move the blank output to another dpi.

    The failure is dpi-dependent because text metrics are not
    dpi-invariant: a layout that has converged at ``figure.dpi`` can
    still measure differently at the save dpi and trigger a mid-render
    resize. Sweep both sides of the default.
    """
    build(df)
    assert _save_and_measure(tmp_path, f"bottom_{dpi}", dpi=dpi) > _MIN_INK


def test_right_legend_and_no_legend_png_have_ink(df, tmp_path):
    """Control cases -- these were never blank and must stay that way."""
    fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
    pp.scatterplot(data=df, x="x", y="y", hue="g", ax=ax)
    pp.legend(side="right")
    assert _save_and_measure(tmp_path, "right_legend") > _MIN_INK
    plt.close(fig)

    fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
    pp.scatterplot(data=df, x="x", y="y", ax=ax)
    assert _save_and_measure(tmp_path, "no_legend") > _MIN_INK


def test_saved_pixel_dimensions_match_the_settled_figure_size(df, tmp_path):
    """The written file must have the pixel size of the settled layout.

    A resize applied from inside the output render is exactly what blanks
    the file, and it also truncates the written raster: the writer emits
    the buffer that was drawn (the pre-resize size) while the figure ends
    up at the post-resize size. Reading the saved dimensions back is what
    makes this discriminating -- asserting on ``fig.get_size_inches()``
    after the fact does not, because the next draw restores the settled
    size before the assertion runs.

    Measured with the freeze reverted: the saved PNG came back
    ``(1129, 1135)`` against an expected ``(1135, 1135)``.

    ``bbox_inches`` is ``None`` for ``pp.savefig``, so pixel dimensions
    are exactly ``size_inches * dpi``.
    """
    fig = _figure_level_bottom_legend(df)
    fig._publiplots_auto_layout.settle()
    settled = tuple(fig.get_size_inches())
    dpi = pp.rcParams["savefig.dpi"]  # publiplots' own default, 600
    path = tmp_path / "stable.png"
    pp.savefig(str(path))

    width_in, height_in = settled
    exact = (height_in * dpi, width_in * dpi)  # imread: (h, w)
    actual = plt.imread(str(path)).shape[:2]
    # Agg allocates ``_RendererAgg(int(width), int(height), dpi)`` -- it
    # truncates, it does not round -- so the exact product is never an
    # integer and a strict equality against ``round()`` is a coin flip on
    # the fractional part. Verified here: the per-axes builder in this
    # file settles at height x 600 = 1026.5604, writes 1026, and
    # ``round()`` says 1027. A +/-1 tolerance loses no discriminating
    # power: the bug this guards against was off by 6 px (1129 vs 1135).
    assert all(abs(a - e) <= 1 for a, e in zip(actual, exact)), (
        f"saved raster dimensions {actual} disagree with the settled figure "
        f"size {exact} -- the layout resized the figure mid-render"
    )

    # The figure itself must also still be the size settle() converged on.
    assert tuple(fig.get_size_inches()) == pytest.approx(settled, abs=1e-9)


def test_inline_print_figure_has_ink(df, tmp_path):
    """IPython's inline display renders via canvas.print_figure directly.

    It never calls ``fig.savefig``, so it relies on the same
    settle-then-freeze guard being installed at the print_figure level.
    """
    import io

    _figure_level_bottom_legend(df)
    buf = io.BytesIO()
    plt.gcf().canvas.print_figure(buf, format="png")
    buf.seek(0)
    img = plt.imread(buf)
    assert float((img[..., :3] < 1.0).sum()) > 0.0
