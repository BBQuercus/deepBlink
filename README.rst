.. image:: https://badge.fury.io/py/deepblink.svg
    :target: https://badge.fury.io/py/deepblink
    :alt: Pypi package version number.
.. image:: https://travis-ci.org/BBQuercus/deepBlink.svg?branch=master
    :target: https://travis-ci.org/BBQuercus/deepBlink
    :alt: Travis CI build status.
.. image:: https://ci.appveyor.com/api/projects/status/86ylig998derkv0c/branch/master?svg=true
    :target: https://ci.appveyor.com/project/BBQuercus/deepblink/branch/master
    :alt: Appveyor build status.
.. image:: https://img.shields.io/badge/License-MIT-brightgreen.svg
    :alt: GitHub code licence is MIT.
.. .. image:: https://codecov.io/gh/BBQuercus/deepBlink/branch/master/graph/badge.svg
..     :target: https://codecov.io/gh/BBQuercus/deepBlink
..     :alt: Codecov test coverage.

.. raw:: html

    <img src="https://github.com/bbquercus/deepblink/raw/master/images/logo.jpg" width="200px" align="right" alt="Logo of deepBlink.">

============
deepBlink
============

Threshold independent detection and localization of diffraction-limited spots.


Overview
============
In biomedical microscopy data, a common task involves the detection of
diffraction-limited spots that visualize single proteins, domains, mRNAs,
and many more. These spots were traditionally detected with mathematical
operators such as Laplacian of Gaussian. These operators, however, rely
on human input ranging from image-intensity thresholds, approximative
spot sizes, etc. This process is tedious and not always reliable. DeepBlink
relies on neural networks to automatically find spots without the need for
human intervention. DeepBlink is available as a ready-to-use command-line
interface.

Example images will follow shortly.

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
