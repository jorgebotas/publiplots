Claude Code Plugin
==================

.. image:: _static/claudecode-color.svg
   :width: 96
   :align: center
   :alt: Claude Code

.. raw:: html

   <p></p>

PubliPlots ships an official `Claude Code <https://claude.com/claude-code>`_
plugin that teaches the assistant how to use the library idiomatically. The
plugin bundles two skills — ``publiplots-guide`` and ``legend-placement`` —
that Claude loads on demand when you work on plotting code, so you get
publication-correct layouts, palettes, and legend placement without having to
restate conventions in every prompt.

Installation
------------

The plugin lives in this repository under ``.claude-plugin/`` and is
distributed through the Claude Code plugin marketplace. Install it with two
commands inside any Claude Code session:

.. code-block:: bash

   /plugin marketplace add jorgebotas/publiplots
   /plugin install publiplots@publiplots

.. tip::
   Pin to a tagged release for reproducibility — e.g.
   ``/plugin install publiplots@publiplots@v0.10.1``. Plugin versions track
   library releases, so the skill content always matches the ``publiplots``
   API you have installed.

Skills included
---------------

Both skills ship as plain Markdown under the repository's ``skills/``
directory. The summaries below cover the essentials; browse the full source
on GitHub for the complete reference:

* `skills/publiplots-guide/SKILL.md <https://github.com/jorgebotas/publiplots/blob/main/skills/publiplots-guide/SKILL.md>`_
* `skills/legend-placement/SKILL.md <https://github.com/jorgebotas/publiplots/blob/main/skills/legend-placement/SKILL.md>`_

``publiplots-guide``
~~~~~~~~~~~~~~~~~~~~

The primary skill. It teaches Claude the library's philosophy and the full
public API surface:

* **Millimeter-based layout.** Axes are sized in mm via
  ``axes_size=(w_mm, h_mm)``; the figure grows to fit decorations. ``figsize``
  is rejected with a ``TypeError``.
* **rcParams auto-apply on import.** Arial, 8pt labels, 0.75pt strokes,
  ``fonttype=42``, ``savefig.dpi=600``, transparent background.
* **Palette system.** ``palette='pastel'`` for named palettes, or a dict to
  pin hue levels to colors when panels see different subsets.
* **Legend stashing.** Plot calls stash entries; ``pp.legend(...)`` collects
  them later as an independent artist managed by the layout reactor.
* **``pp.savefig`` does not force ``bbox_inches='tight'``** — the figure is
  already laid out to mm-precise margins, and tight-cropping would shift
  figure-anchored legend bands.

It also enumerates the ``pp.*`` public API (plots, layout, legend, I/O,
axes utilities, theming, markers/hatches), canonical idioms (``pp.subplots``
over ``plt.subplots``, dict palettes across panels, ``pp.rcParams['edgecolor']``
as a global edge toggle), and common gotchas.

``legend-placement``
~~~~~~~~~~~~~~~~~~~~

A deep-dive skill focused on ``pp.legend``. It contains a **scoping decision
tree** that maps intent to the right call:

* **Per-axes internal:** ``pp.legend(ax)`` — counted in ``ax.tightbbox``,
  behaves like a tick label.
* **Per-axes external band:** ``pp.legend(anchor=ax)`` — overhangs past the
  axes edge, absorbing the cell's ``right`` / ``xlabel_space`` reservation.
* **Row band:** ``pp.legend(axes[0], side='top')``.
* **Column band:** ``pp.legend(axes[:, 0], side='left')``.
* **Full-figure band:** ``pp.legend(side='right')`` / ``'bottom'`` / ``'top'``
  / ``'left'``.

.. note::
   The ``pp.legend(ax)`` vs. ``pp.legend(anchor=ax)`` asymmetry is the most
   common point of confusion. The first argument is the **scope** (which
   axes contribute entries); ``anchor=`` is a geometric override that pins
   the legend to a specific axes' edge as an external band.

Usage
-----

Auto-invocation
~~~~~~~~~~~~~~~

Claude detects publiplots-related work from your prompt and loads the
relevant skill automatically. Examples:

.. code-block:: text

   "Make a 2x3 scatter grid with a shared legend on the right."
   # -> loads publiplots-guide, uses pp.subplots + pp.legend(side='right')

   "How do I put a legend above just the top row of my grid?"
   # -> loads legend-placement, answers with pp.legend(axes[0], side='top')

   "Refactor this plt.subplots call to use publiplots conventions."
   # -> loads publiplots-guide, rewrites with pp.subplots(axes_size=(...))

Manual invocation
~~~~~~~~~~~~~~~~~

Invoke a skill explicitly when you want its full contents in context before
asking a follow-up question:

.. code-block:: bash

   /publiplots:publiplots-guide
   /publiplots:legend-placement

This is useful when you're exploring the API before writing code, or when
you want Claude to ground a design discussion in the library's conventions.

Updating the plugin
-------------------

Plugin versions track library releases, so update alongside ``publiplots``
itself:

.. code-block:: bash

   /plugin marketplace update publiplots
   /plugin update publiplots

Feedback and issues
-------------------

Found a skill gap, an outdated idiom, or a bug in the plugin manifest?
Open an issue on the
`GitHub issue tracker <https://github.com/jorgebotas/publiplots/issues>`_.
The skills live alongside the library source, so fixes ship with the next
tagged release.
