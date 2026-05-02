PubliPlots Documentation
========================

**PubliPlots** is a Python library for creating publication-ready plots with a clean, modular API.
It provides a seaborn-like interface with sensible defaults and extensive customization options.

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/github/license/jorgebotas/publiplots.svg
   :target: https://github.com/jorgebotas/publiplots/blob/main/LICENSE
   :alt: License

Features
--------

* **Publication-ready plots**: Beautiful, journal-quality visualizations out of the box
* **Seaborn-like API**: Familiar, intuitive interface for matplotlib users
* **Extensive customization**: Fine-grained control over all visual elements
* **Advanced plot types**: Venn diagrams (2-5 way), UpSet plots, bubble plots, and more
* **Flexible styling**: Built-in themes for notebook and publication formats
* **Type hints**: Full type annotation support for better IDE integration

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. tip::
   For faster installation, we recommend using `uv <https://github.com/astral-sh/uv>`_:

   .. code-block:: bash

      uv pip install publiplots

   Or using pip:

.. code-block:: bash

   pip install publiplots

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import publiplots as pp
   import pandas as pd
   import numpy as np

   # Create sample data
   data = pd.DataFrame({
       'category': ['A', 'B', 'C', 'D'],
       'value': [23, 45, 38, 52]
   })

   # Create a simple bar plot
   ax = pp.barplot(
       data=data,
       x='category',
       y='value',
       title='Simple Bar Plot',
       xlabel='Category',
       ylabel='Value'
   )

   # Save the figure
   pp.savefig('output.png')

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Examples Gallery

   auto_examples/index

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
