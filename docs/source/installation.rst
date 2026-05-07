Installation
============

Requirements
------------

PubliPlots requires Python 3.9 or later and the following packages:

* matplotlib >= 3.5.0
* seaborn >= 0.12.0
* numpy >= 1.21.0
* pandas >= 1.3.0

Installing from PyPI
--------------------

Using uv (Recommended)
~~~~~~~~~~~~~~~~~~~~~~

For the fastest installation, we recommend using `uv <https://github.com/astral-sh/uv>`_:

.. code-block:: bash

   uv pip install publiplots

Using pip
~~~~~~~~~

Alternatively, you can use pip:

.. code-block:: bash

   pip install publiplots

Installing from Source
----------------------

To install the latest development version from GitHub:

.. code-block:: bash

   git clone https://github.com/jorgebotas/publiplots.git
   cd publiplots
   uv pip install -e .

Or with pip:

.. code-block:: bash

   pip install -e .

Development Installation
------------------------

If you want to contribute to PubliPlots, install the development dependencies:

.. code-block:: bash

   uv pip install -e ".[dev]"

Or with pip:

.. code-block:: bash

   pip install -e ".[dev]"

This will install additional packages for testing and code quality:

* pytest
* pytest-cov
* black
* mypy
* ruff

Documentation Dependencies
--------------------------

To build the documentation locally:

.. code-block:: bash

   uv pip install -e ".[docs]"

Or with pip:

.. code-block:: bash

   pip install -e ".[docs]"

This will install Sphinx, sphinx-gallery, and related documentation tools.

Verifying Installation
-----------------------

To verify that PubliPlots is installed correctly:

.. code-block:: python

   import publiplots as pp
   print(pp.__version__)

This should print the installed version number.

.. tip::
   If you use `Claude Code <https://claude.com/claude-code>`_, check out the
   :doc:`claude_code_plugin` for install instructions and usage examples.
