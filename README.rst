========
Overview
========

Threshold independent detection and localization of diffraction-limited spots.

* Free software: MIT license

Installation
============

::

    pip install deepblink

You can also install the in-development version with::

    pip install git+ssh://git@github.com/bbquercus/deepblink/bbquercus/deepblink.git@master

Documentation
=============


https://deepblink.readthedocs.io/


Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
