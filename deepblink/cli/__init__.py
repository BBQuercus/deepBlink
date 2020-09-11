"""Module that contains the command line app.

The CLI is modular containing four distinct submodules which will be
discussed in more detail below:
----------
1. Training
----------
2. Checking
----------
3. Prediction
Used for inferencing of new data with pretrained models.
Because the model is loaded into memory for every run, it is faster to
run multiple images at once by passing a directory as input.
----------
4. Evaluation
"""

from ._main import main
