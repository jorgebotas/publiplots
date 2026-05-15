"""Tests for PanelImage dataclass — PR 5 promotes from stub to real panel kind.

PanelImage references an external schematic file (PDF/SVG/PNG/JPG/TIFF)
to be vector-stamped (or raster-fallback) into a reserved canvas slot.
Construction-time validation enforces ext support, align/clip vocab,
and path existence.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import publiplots as pp
from publiplots.composer.exceptions import ComposerError, ComposerVectorError
from publiplots.composer.panels import PanelImage


# ---------------------------------------------------------------------------
# ComposerVectorError exception
# ---------------------------------------------------------------------------

def test_composer_vector_error_subclass_of_composer_error():
    assert issubclass(ComposerVectorError, ComposerError)


def test_composer_vector_error_carries_context():
    err = ComposerVectorError(
        "cairosvg failed",
        panel_label="A",
        path="/tmp/missing.svg",
        source_error="OSError: no such file",
    )
    assert err.panel_label == "A"
    assert err.path == "/tmp/missing.svg"
    assert "cairosvg failed" in str(err)


# ---------------------------------------------------------------------------
# PanelImage dataclass
# ---------------------------------------------------------------------------

def test_panel_image_basic_construction(tmp_path):
    p = tmp_path / "schematic.svg"
    p.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='10mm' height='10mm'/>")
    panel = PanelImage(label="A", path=p, size=(40.0, 30.0))
    assert panel.label == "A"
    assert panel.path == p
    assert panel.size == (40.0, 30.0)
    assert panel.align == "center"
    assert panel.clip == "fit"


def test_panel_image_accepts_string_path_and_normalizes(tmp_path):
    """str path → normalized to Path in __post_init__ (architect blocker #7)."""
    p = tmp_path / "schematic.svg"
    p.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    panel = PanelImage(label="A", path=str(p), size=(40.0, 30.0))
    assert isinstance(panel.path, Path)
    assert panel.path == p


def test_panel_image_rejects_unsupported_extension(tmp_path):
    p = tmp_path / "schematic.eps"
    p.write_text("dummy")
    with pytest.raises(ValueError, match=r"unsupported.*\.eps"):
        PanelImage(label="A", path=p, size=(40.0, 30.0))


def test_panel_image_rejects_missing_path(tmp_path):
    p = tmp_path / "does-not-exist.svg"
    with pytest.raises(FileNotFoundError, match=r"does-not-exist\.svg"):
        PanelImage(label="A", path=p, size=(40.0, 30.0))


def test_panel_image_rejects_invalid_align(tmp_path):
    p = tmp_path / "schematic.svg"
    p.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    with pytest.raises(ValueError, match=r"align.*'center'.*9"):
        PanelImage(label="A", path=p, size=(40.0, 30.0), align="middle")


def test_panel_image_rejects_invalid_clip(tmp_path):
    p = tmp_path / "schematic.svg"
    p.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    with pytest.raises(ValueError, match=r"clip.*'fit'.*'fill'.*'stretch'"):
        PanelImage(label="A", path=p, size=(40.0, 30.0), clip="cover")


def test_panel_image_accepts_all_9_align_values(tmp_path):
    p = tmp_path / "schematic.svg"
    p.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    for align in ("top-left", "top", "top-right", "left", "center",
                  "right", "bottom-left", "bottom", "bottom-right"):
        panel = PanelImage(label="A", path=p, size=(40.0, 30.0), align=align)
        assert panel.align == align


def test_panel_image_accepts_all_3_clip_values(tmp_path):
    p = tmp_path / "schematic.svg"
    p.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    for clip in ("fit", "fill", "stretch"):
        panel = PanelImage(label="A", path=p, size=(40.0, 30.0), clip=clip)
        assert panel.clip == clip


def test_panel_image_accepts_all_supported_extensions(tmp_path):
    contents = {
        ".svg": "<svg xmlns='http://www.w3.org/2000/svg'/>",
        ".pdf": "%PDF-1.4\n%dummy\n",  # not a valid PDF, but extension check is what we test
        ".png": "\x89PNG\r\n\x1a\n",
        ".jpg": "\xff\xd8\xff\xe0",
        ".jpeg": "\xff\xd8\xff\xe0",
        ".tif": "II*\x00",
        ".tiff": "II*\x00",
    }
    for ext, content in contents.items():
        p = tmp_path / f"schematic{ext}"
        p.write_bytes(content.encode("latin-1") if isinstance(content, str) else content)
        panel = PanelImage(label="A", path=p, size=(40.0, 30.0))
        assert panel.path == p


def test_panel_image_is_frozen_dataclass(tmp_path):
    p = tmp_path / "schematic.svg"
    p.write_text("<svg/>")
    panel = PanelImage(label="A", path=p, size=(40.0, 30.0))
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        panel.label = "B"


def test_panel_image_size_validation_propagates(tmp_path):
    """PR 1's _validate_panel_size shared helper rejects bad sizes."""
    p = tmp_path / "schematic.svg"
    p.write_text("<svg/>")
    with pytest.raises((ValueError, TypeError)):
        PanelImage(label="A", path=p, size=(0.0, 30.0))  # zero width
