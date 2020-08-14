
Changelog
=========

0.0.5 (2020-08-14)
------------------

    “It works fine on MY computer”

Those days are over as it's testing time. While the test-suite is not fully complete it gives us some more confidence over our commits.
Additional changes in this release include:

* Scipy requirement for a new `deepblink.metrics.f1_cutoff_score` way of calculating the F1 score directly from coordinates.
* Depreciation of `deepblink.metrics.compute_score` and `deepblink.metrics.weighted_f1_coordinates`.
* Addition of `deepblink.optimizers.amsgrad` function.
* Movement of functions:

    * `train_valid_split` to `util`
    * `load_image` to `io`

The next release will include benchmarking scripts and is most likely the last release before the first minor `0.1.0` and non-pre-release.


0.0.4 (2020-07-10)
------------------

* Addition of command line interface (currently without verbosity capabilities).
* Bin/ folder with ready to use training script.
* Removal of src/ directory. Deepblink is now sailing GitHub on its own.
* Major docs overhaul with the addition of more detailed input / functionality descriptions and the addition of a usage overview and example in README.
* @zhanyinx is now listed as co-author.


0.0.3 (2020-07-02)
------------------

* First custom code release.
* All utility functions and training loop included.
* CLI still cookiecutter and not functional.
* First of five patches leading up to the first minor version:

    * v0.0.3 Utility functions.
    * v0.0.4 Command line interface for inference and script folder for training.
    * v0.0.5 Evaluation added to measure performance on test dataset.
    * v0.0.6 Addition of tests and completion of current TODO's.
    * V0.0.7 Benchmarks/functions added to compare deepblink to other methods.

0.0.2 (2020-06-29)
------------------

* Still cookiecutter code but addition of docstrings to test sphinx apidoc capabilities.

0.0.1 (2020-06-24)
------------------

* First release on PyPI with cookiecutter HelloWorld template to set everything up.
