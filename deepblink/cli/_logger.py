"""Logging related functions for deepblinks CLI."""

import logging
import sys


def _configure_logger(verbose: bool, debug: bool):
    """Return verbose logger with three levels.

    * Verbose false and debug false - no verbose logging.
    * Verbose true and debug false - only info level loginfo for standard users.
    * Debug true - debug mode for developers.
    """
    if debug:
        level = logging.DEBUG
        form = "%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s"
    else:
        if verbose:
            level = logging.INFO
        else:
            level = logging.ERROR
        form = "%(asctime)s: %(message)s"

    logging.basicConfig(format=form, stream=sys.stdout, level=level)
    logger = logging.getLogger("Verbose output logger")
    return logger
