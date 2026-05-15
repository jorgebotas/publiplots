"""Shared determinism strings for the compositing pipelines.

PR 5 (PDF) and PR 6a (SVG) each pinned a small set of literal strings
to make the byte stream reproducible:

- ``_DEFAULT_CREATION_DATE`` — PDF ``/CreationDate`` value (must match
  pypdf's ``D:YYYYMMDDHHMMSSZ`` format).
- ``_DEFAULT_DATE`` — SVG ``<dc:date>`` value (matplotlib's metadata
  ``Date`` key passes through unchanged).
- ``_SVG_HASHSALT`` — matplotlib rcParam ``svg.hashsalt`` pin so the
  ``<defs>`` element id suffixes are deterministic.
- ``_PRODUCER`` — PDF ``/Producer`` metadata value.

PR 6b adds ``compositing/_embed.py`` (Figure→PDF/SVG bytes helpers
for ``canvas.embed_figure``) which need to reuse the same literals so
the embedded figure's bytes are deterministic AND match the hashsalt
of the surrounding canvas SVG. Lifting the constants up here avoids
circular imports between ``_embed.py`` and ``pdf.py`` / ``svg.py``.

The values are intentionally string-literal constants (no environment
or argument injection) — the composer's byte-determinism contract
demands a pinned default. Callers that want a different value pass it
explicitly via ``metadata_creation_date`` / ``metadata_date``.
"""
from __future__ import annotations


_DEFAULT_CREATION_DATE = "D:20260101000000Z"
"""PDF ``/CreationDate`` default value (pypdf-formatted UTC literal)."""

_DEFAULT_DATE = "2026-01-01T00:00:00"
"""SVG ``<dc:date>`` default value (ISO-8601 literal, no timezone)."""

_SVG_HASHSALT = "publiplots-composer"
"""matplotlib ``svg.hashsalt`` rcParam pin for deterministic ``<defs>`` ids."""

_PRODUCER = "publiplots-composer"
"""PDF ``/Producer`` metadata value."""
