[![Github Actions Status](https://github.com/bbquercus/deepblink/workflows/main/badge.svg)](https://github.com/bbquercus/deepblink/actions)
[![GitHub code licence is MIT](https://img.shields.io/badge/license-MIT-brightgreen.svg)](https://raw.githubusercontent.com/BBQuercus/deepBlink/master/LICENSE)
[![Pypi package version number](https://badge.fury.io/py/deepblink.svg)](https://badge.fury.io/py/deepblink)
[![DOI for deepBlink](https://zenodo.org/badge/DOI/10.5281/zenodo.3992543.svg)](https://doi.org/10.5281/zenodo.3992543)
<!-- [![Codecov test coverage](https://codecov.io/gh/BBQuercus/deepBlink/branch/master/graph/badge.svg)](https://codecov.io/gh/BBQuercus/deepBlink) -->

<img src="https://github.com/bbquercus/deepblink/raw/master/images/logo.jpg" width="200px" align="right" alt="Logo of deepBlink.">

# deepBlink

Threshold independent detection and localization of diffraction-limited spots.

## Contents
- [Contents](#contents)
- [Overview](#overview)
- [Documentation](#documentation)
- [Installation](#installation)
- [Usage](#usage)

## Overview
In biomedical microscopy data, a common task involves the detection of
diffraction-limited spots that visualize single proteins, domains, mRNAs,
and many more. These spots were traditionally detected with mathematical
operators such as Laplacian of Gaussian. These operators, however, rely
on human input ranging from image-intensity thresholds, approximative
spot sizes, etc. This process is tedious and not always reliable. DeepBlink
relies on neural networks to automatically find spots without the need for
human intervention. DeepBlink is available as a ready-to-use command-line
interface.

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


## Documentation

More documentation about deepBlink including how to train, create a dataset, contribute etc. is available at [https://github.com/BBQuercus/deepBlink/wiki](https://github.com/BBQuercus/deepBlink/wiki).


## Installation
This package is built for [Python](https://www.python.org/downloads/) versions newer than 3.6 and can easily be installed with pip:
```bash
pip install deepblink
```

Or using conda:
```bash
conda install -c bbquercus deepblink
```


Additionally for GPU support, install `tensorflow-gpu` through pip and with the
appropriate `CUDA` and `cuDNN` verions matching your [GPU setup](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html).


## Usage
Inferencing on deepBlink is performed at the command line as follows:

```bash
deepblink predict -m MODEL -i INPUT [-o OUTPUT] [-r RADIUS] [-s SHAPE]
```

With `MODEL` being a pre-trained or custom model and `INPUT` being the path to a input image or folder containing images.