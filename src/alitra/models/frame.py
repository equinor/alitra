from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Frame:
    """
    Frame is used by most of our models to describe in which frame a model lives,
    or the two frames a transform is between.
    """

    name: str
