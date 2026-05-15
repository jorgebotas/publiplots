"""Compositing pipeline subpackage.

PR 5: PDF compositing via pypdf + cairosvg.
PR 6a: SVG compositing (in-tree) via lxml + cairosvg.
PR 6b: embed_figure; raster polish; TIFF + CMYK; save_multiple.
"""
from publiplots.composer.compositing.pdf import savefig_pdf
from publiplots.composer.compositing.svg import savefig_svg

__all__ = ["savefig_pdf", "savefig_svg"]
