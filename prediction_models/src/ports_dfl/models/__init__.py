"""Model implementations: linear, RealMLP, TabM, NODE.

All models inherit from :class:`BaseModel` so they can be used
interchangeably by training and tuning code.
"""

# This file makes the `models` folder a Python "package". Re-importing BaseModel
# here lets other code write `from ports_dfl.models import BaseModel` instead of
# the longer `from ports_dfl.models.base import BaseModel`.
from ports_dfl.models.base import BaseModel

# `__all__` lists the names exported by `from ports_dfl.models import *`. It also
# documents the package's public surface (what's meant to be used elsewhere).
__all__ = ["BaseModel"]
