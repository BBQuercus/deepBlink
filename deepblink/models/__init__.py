"""Models module with the training loop and logic to handle data which feeds into the loop."""

from ._models import Model
from .spots import SpotsModel

__all__ = ["Model", "SpotsModel"]
