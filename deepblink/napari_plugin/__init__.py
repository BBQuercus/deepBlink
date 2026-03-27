"""Napari plugin for interactive spot labeling with deepBlink.

Requires the optional ``napari`` dependency::

    pip install "deepblink[napari]"
"""


def __getattr__(name):
    if name == "SpotLabeler":
        from .widget import SpotLabeler

        return SpotLabeler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["SpotLabeler"]
