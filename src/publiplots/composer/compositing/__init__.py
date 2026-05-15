"""Compositing pipeline subpackage.

PR 5: PDF compositing via pypdf + cairosvg.
PR 6: SVG compositing (in-tree); embed_figure; raster polish.
"""
from publiplots.composer.compositing.pdf import savefig_pdf

__all__ = ["savefig_pdf"]
