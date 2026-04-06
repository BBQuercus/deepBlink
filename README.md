[![Github Actions Status](https://github.com/bbquercus/deepblink/workflows/main/badge.svg)](https://github.com/bbquercus/deepblink/actions)
[![GitHub code licence is MIT](https://img.shields.io/badge/license-MIT-brightgreen.svg)](https://raw.githubusercontent.com/BBQuercus/deepBlink/master/LICENSE)
[![Pypi package version number](https://badge.fury.io/py/deepblink.svg)](https://badge.fury.io/py/deepblink)
[![Pypi download statistics](https://img.shields.io/pypi/dm/deepblink.svg)](https://badge.fury.io/py/deepblink)
[![DOI for deepBlink](https://zenodo.org/badge/DOI/10.5281/zenodo.3992543.svg)](https://doi.org/10.5281/zenodo.3992543)
[![Codecov test coverage](https://codecov.io/gh/BBQuercus/deepBlink/branch/master/graph/badge.svg)](https://codecov.io/gh/BBQuercus/deepBlink)

<img src="https://github.com/bbquercus/deepblink/raw/master/images/logo.jpg" width="200px" align="right" alt="Logo of deepBlink.">


# deepBlink

Threshold independent detection and localization of diffraction-limited spots.

## Contents
- [Contents](#contents)
- [Overview](#overview)
- [Documentation](#documentation)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)

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
    <th min-width="200px" width="50%"><img src="https://github.com/bbquercus/deepblink/raw/master/images/usage.png" alt="Basic usage example of deepBlink."></th>
    <th min-width="200px" width="50%"><img src="https://github.com/bbquercus/deepblink/raw/master/images/example.jpg" alt="Example images processed with deepBlink."></th>
    </tr>
</table>


## Documentation

More documentation about deepBlink including how to train, create a dataset, contribute etc. is available at [https://github.com/BBQuercus/deepBlink/wiki](https://github.com/BBQuercus/deepBlink/wiki).


## Installation
This package is built for [Python](https://www.python.org/downloads/) versions 3.11+ and can easily be installed with pip:
```bash
pip install deepblink
```

Or using conda:
```bash
conda install -c bbquercus deepblink
```

For GPU support, TensorFlow 2.16+ ships CUDA via pip — no manual CUDA/cuDNN setup needed:
```bash
pip install 'tensorflow[and-cuda]'
```

You can also use our [KNIME node](https://kni.me/c/phip4SLhBhzPtMwI) for inference. Please follow the installation instructions on KNIME hub.

## Usage

### Python API

**Detect spots in an image:**
```python
import deepblink

model  = deepblink.load_model("model.h5")
image  = deepblink.load_image("image.tif")
coords = deepblink.predict(image, model, probability=0.5)
# coords is an (N, 2) array of [row, col] coordinates
```

**Measure intensities around detected spots:**
```python
intensities = deepblink.get_intensities(image, coords, radius=3)
```

**Evaluate predictions against ground truth:**
```python
df = deepblink.compute_metrics(pred=coords, true=ground_truth, mdist=3)
```

**Load a deepBlink dataset:**
```python
x_train, y_train, x_valid, y_valid, x_test, y_test = deepblink.load_npz("dataset.npz")
```

All submodules (`deepblink.losses`, `deepblink.networks`, `deepblink.augment`, etc.)
are still accessible for advanced use cases like custom training loops.

### Command Line
A video overview can be found [here](https://www.youtube.com/watch?v=vlXMg4k79LQ). Inferencing on deepBlink is performed at the command line as follows:

```bash
deepblink predict -m MODEL -i INPUT [-o OUTPUT] [-r RADIUS] [-s SHAPE]
```

With `MODEL` being a pre-trained or custom model and `INPUT` being the path to a input image or folder containing images.


## Citation
deepBlink is currently available on Nucleic Acid Research [here](https://academic.oup.com/nar/advance-article/doi/10.1093/nar/gkab546/6312733). If you find deepBlink useful, consider citing us:

```bibtex
@article{10.1093/nar/gkab546,
    author = {Eichenberger, Bastian Th and Zhan, YinXiu and Rempfler, Markus and Giorgetti, Luca and Chao, Jeffrey A},
    title = "{deepBlink: threshold-independent detection and localization of diffraction-limited spots}",
    journal = {Nucleic Acids Research},
    year = {2021},
    month = {07},
    issn = {0305-1048},
    doi = {10.1093/nar/gkab546},
    url = {https://doi.org/10.1093/nar/gkab546},
    note = {gkab546},
    eprint = {https://academic.oup.com/nar/advance-article-pdf/doi/10.1093/nar/gkab546/38848972/gkab546.pdf},
}
```
