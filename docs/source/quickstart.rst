Quick Start Guide
=================

This guide will help you get started with PubliPlots quickly.

Setting Up Your Environment
----------------------------

First, import the necessary libraries:

.. code-block:: python

   import publiplots as pp
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   # Set the style for your workflow
   pp.set_notebook_style()  # For interactive work
   # pp.set_publication_style()  # For final figures

Creating Your First Plot
-------------------------

Bar Plot
~~~~~~~~

Create a simple bar plot from a DataFrame:

.. code-block:: python

   # Create sample data
   data = pd.DataFrame({
       'category': ['A', 'B', 'C', 'D'],
       'value': [23, 45, 38, 52]
   })

   # Create bar plot
   fig, ax = pp.barplot(
       data=data,
       x='category',
       y='value',
       title='My First Plot',
       xlabel='Category',
       ylabel='Value',
       palette='pastel'
   )

   plt.show()

Scatter Plot
~~~~~~~~~~~~

Create a scatter plot with color and size encoding:

.. code-block:: python

   # Create sample data
   data = pd.DataFrame({
       'x': np.random.randn(100),
       'y': np.random.randn(100),
       'size': np.random.uniform(1, 10, 100),
       'group': np.random.choice(['A', 'B', 'C'], 100)
   })

   # Create scatter plot
   fig, ax = pp.scatterplot(
       data=data,
       x='x',
       y='y',
       hue='group',
       size='size',
       sizes=(50, 500),
       palette='pastel',
       title='Scatter Plot Example'
   )

   plt.show()

Customizing Your Plots
-----------------------

Using Error Bars
~~~~~~~~~~~~~~~~

Add error bars to show variability:

.. code-block:: python

   # Create data with multiple measurements
   data = pd.DataFrame({
       'treatment': np.repeat(['Control', 'Drug A', 'Drug B'], 10),
       'response': np.concatenate([
           np.random.normal(100, 15, 10),
           np.random.normal(120, 12, 10),
           np.random.normal(135, 18, 10),
       ])
   })

   # Create bar plot with error bars
   fig, ax = pp.barplot(
       data=data,
       x='treatment',
       y='response',
       errorbar='se',  # Standard error
       capsize=0.1,
       title='Drug Response'
   )

Using Hatch Patterns
~~~~~~~~~~~~~~~~~~~~

Add hatch patterns for black-and-white publications:

.. code-block:: python

   fig, ax = pp.barplot(
       data=data,
       x='treatment',
       y='response',
       hatch='treatment',
       hatch_map={'Control': '', 'Drug A': '//', 'Drug B': 'xx'},
       alpha=0.0,
       color='#5D83C3'
   )

Advanced Plots
--------------

Venn Diagrams
~~~~~~~~~~~~~

Create Venn diagrams for set intersections:

.. code-block:: python

   # Create sets
   set_a = set(range(1, 50))
   set_b = set(range(30, 80))
   set_c = set(range(60, 100))

   # Create 3-way Venn diagram
   fig, ax = pp.venn(
       sets=[set_a, set_b, set_c],
       labels=['Set A', 'Set B', 'Set C'],
       colors=pp.color_palette('pastel', n_colors=3)
   )

UpSet Plots
~~~~~~~~~~~

Create UpSet plots for many-set intersections:

.. code-block:: python

   # Create sets
   sets = {
       'Group A': set(range(1, 60)),
       'Group B': set(range(40, 100)),
       'Group C': set(range(70, 130)),
       'Group D': set(range(30, 90))
   }

   # Create UpSet plot
   fig, axes = pp.upsetplot(
       data=sets,
       sort_by='size',
       title='Set Intersections',
       show_counts=15
   )

Saving Your Figures
-------------------

Save figures in various formats:

.. code-block:: python

   # Save as PNG (high resolution)
   pp.savefig(fig, 'my_plot.png', dpi=300)

   # Save as PDF (vector format)
   pp.savefig(fig, 'my_plot.pdf')

   # Save as SVG (editable vector format)
   pp.savefig(fig, 'my_plot.svg')

   # Save multiple figures at once
   pp.save_multiple([fig1, fig2, fig3], 'output_dir')

Configuration
-------------

Global Settings
~~~~~~~~~~~~~~~

Configure global plotting parameters using ``pp.rcParams``:

.. code-block:: python

   # Set default colors and transparency
   pp.rcParams['color'] = '#E67E7E'
   pp.rcParams['alpha'] = 0.3

   # Set a global edge color for patches and marker outlines.
   # Default is None (each plot picks its own — typically the face color).
   # Per-call ``edgecolor=`` arguments override the rcParam.
   pp.rcParams['edgecolor'] = 'black'

   # Set figure size
   pp.rcParams['figure.figsize'] = (8, 6)

   # Set hatch pattern density
   pp.set_hatch_mode(2)  # 1=sparse, 2=medium, 3=dense

Next Steps
----------

* Explore the :doc:`auto_examples/index` for more detailed examples
* Check the :doc:`api/index` for complete function documentation
* Read about advanced customization options in the examples gallery
