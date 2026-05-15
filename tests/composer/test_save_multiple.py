"""Tests for Canvas.save_multiple — multi-format sugar.

PR 6b adds ``Canvas.save_multiple(stem, formats=None, **kwargs)`` as a
multi-panel-aware counterpart to the free function ``pp.save_multiple``.
The defining differences:

- Default ``formats=None`` matches ``pp.save_multiple`` (``['png',
  'pdf']``) for ergonomic parity.
- Pre-validates the format list BEFORE writing any file (no partial
  state on bad input).
- Pre-validates ``cmyk=True`` against vector formats up-front so a
  multi-format save with ``formats=['tif', 'pdf']`` doesn't write the
  ``.tif`` then fail on the ``.pdf``.

The free-function ``pp.save_multiple`` is intentionally permissive
(forwards bad formats to matplotlib) — Canvas.save_multiple is stricter
because it knows the supported ext set.
"""
from __future__ import annotations

import pytest

import publiplots as pp


@pytest.fixture
def simple_canvas():
    """A small canvas with no PanelImage — easy to round-trip across formats."""
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )
    return canvas


def test_save_multiple_writes_all_formats(simple_canvas, tmp_path):
    """``formats=['pdf', 'svg', 'png']`` writes 3 files."""
    stem = tmp_path / "figure"
    paths = simple_canvas.save_multiple(stem, formats=["pdf", "svg", "png"])
    assert len(paths) == 3
    for p in paths:
        assert p.exists()
    assert {p.suffix for p in paths} == {".pdf", ".svg", ".png"}


def test_save_multiple_default_formats(simple_canvas, tmp_path):
    """``formats=None`` writes png + pdf (parity with ``pp.save_multiple``)."""
    stem = tmp_path / "figure"
    paths = simple_canvas.save_multiple(stem)
    assert len(paths) == 2
    assert {p.suffix for p in paths} == {".png", ".pdf"}


def test_save_multiple_returns_paths(simple_canvas, tmp_path):
    """The return value lists the written paths in the order of formats."""
    stem = tmp_path / "figure"
    paths = simple_canvas.save_multiple(stem, formats=["pdf", "png"])
    assert paths[0].suffix == ".pdf"
    assert paths[1].suffix == ".png"


def test_save_multiple_accepts_dot_prefixed_formats(simple_canvas, tmp_path):
    """``formats=['.pdf']`` works the same as ``formats=['pdf']``."""
    stem = tmp_path / "figure"
    paths = simple_canvas.save_multiple(stem, formats=[".pdf"])
    assert len(paths) == 1
    assert paths[0].suffix == ".pdf"


def test_save_multiple_empty_formats_raises(simple_canvas, tmp_path):
    """``formats=[]`` raises ValueError BEFORE writing any file."""
    stem = tmp_path / "figure"
    with pytest.raises(ValueError, match=r"formats"):
        simple_canvas.save_multiple(stem, formats=[])
    # No file should have been written.
    assert list(tmp_path.iterdir()) == []


def test_save_multiple_unknown_ext_raises_pre_write(simple_canvas, tmp_path):
    """Unknown ext raises ValueError BEFORE writing any file."""
    stem = tmp_path / "figure"
    with pytest.raises(ValueError, match=r"xyz|unsupported|unknown"):
        simple_canvas.save_multiple(stem, formats=["pdf", "xyz"])
    # No partial-write: neither .pdf nor .xyz was written.
    assert not (tmp_path / "figure.pdf").exists()
    assert not (tmp_path / "figure.xyz").exists()


def test_save_multiple_duplicates_raise(simple_canvas, tmp_path):
    """Duplicate format entries raise ValueError BEFORE writing."""
    stem = tmp_path / "figure"
    with pytest.raises(ValueError, match=r"duplicate"):
        simple_canvas.save_multiple(stem, formats=["png", "png"])


def test_save_multiple_non_str_format_raises(simple_canvas, tmp_path):
    """Non-string entries raise ValueError BEFORE writing."""
    stem = tmp_path / "figure"
    with pytest.raises(ValueError, match=r"str|string"):
        simple_canvas.save_multiple(stem, formats=["pdf", 123])


def test_save_multiple_cmyk_with_vector_format_raises_pre_write(
    simple_canvas, tmp_path,
):
    """``cmyk=True + formats=['tif', 'pdf']`` raises BEFORE writing
    the .tif (no partial state)."""
    stem = tmp_path / "figure"
    with pytest.raises(ValueError, match=r"cmyk.*pdf|cmyk.*vector|cmyk.*raster"):
        simple_canvas.save_multiple(stem, formats=["tif", "pdf"], cmyk=True)
    # Neither file should exist — pre-validate caught it.
    assert not (tmp_path / "figure.tif").exists()
    assert not (tmp_path / "figure.pdf").exists()


def test_save_multiple_path_stem_with_existing_suffix_drops_suffix(
    simple_canvas, tmp_path,
):
    """``stem='foo.bar'`` is treated as a stem (Path.with_suffix replaces
    any suffix)."""
    stem = tmp_path / "figure.original"
    paths = simple_canvas.save_multiple(stem, formats=["pdf"])
    # Path.with_suffix('.pdf') turns 'figure.original' into 'figure.pdf'
    assert paths[0] == tmp_path / "figure.pdf"
    assert paths[0].exists()
