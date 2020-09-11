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
    else:
        if verbose:
            level = logging.INFO
        else:
            level = logging.CRITICAL

    logging.basicConfig(
        format="%(asctime)s: %(message)s", stream=sys.stdout, level=level
    )
    logger = logging.getLogger("Verbose output logger")
    return logger
