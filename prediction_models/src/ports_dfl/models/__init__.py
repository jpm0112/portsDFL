"""Model implementations: linear, RealMLP, TabM, NODE.

All models inherit from :class:`BaseModel` so they can be used
interchangeably by training and tuning code.
"""

from ports_dfl.models.base import BaseModel

__all__ = ["BaseModel"]
