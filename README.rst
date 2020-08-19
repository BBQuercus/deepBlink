.. image:: https://badge.fury.io/py/deepblink.svg
    :target: https://badge.fury.io/py/deepblink
    :alt: Pypi package version number.
.. image:: https://travis-ci.org/BBQuercus/deepBlink.svg?branch=master
    :target: https://travis-ci.org/BBQuercus/deepBlink
    :alt: Travis CI build status.
.. image:: https://ci.appveyor.com/api/projects/status/86ylig998derkv0c/branch/master?svg=true
    :target: https://ci.appveyor.com/project/BBQuercus/deepblink/branch/master
    :alt: Appveyor build status.
.. image:: https://img.shields.io/badge/license-MIT-brightgreen.svg
    :target: https://raw.githubusercontent.com/BBQuercus/deepBlink/master/LICENSE
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

.. raw:: html

    <table width="100%">
      <tr>
        <th>Usage</th>
        <th>Example</th>
      </tr>
      <tr>
        <th min-width="200px" width="50%"><img src="https://github.com/bbquercus/deepblink/raw/master/images/usage.jpg" alt="Basic usage example of deepBlink."></th>
        <th min-width="200px" width="50%"><img src="https://github.com/bbquercus/deepblink/raw/master/images/example.jpg" alt="Example images processed with deepBlink."></th>
      </tr>
    </table>


Installation
============

This package is built for `Python <https://www.python.org/downloads/>`_ versions newer than 3.6.

DeepBlink can easily be installed with pip: ::

    pip install deepblink

Additionally for GPU support, install ``tensorflow-gpu`` through pip and with the
appropriate ``CUDA`` and ``cuDNN`` verions matching your `GPU setup <https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html>`_.

Usage
=============

Inferencing on deepBlink is performed at the command line as follows: ::

    deepblink [-h] [-o OUTPUT] [-t {csv,txt}] [-v] [-V] MODEL INPUT

More detailed information is availabe in our `documentation <https://deepblink.readthedocs.io/>`_.
