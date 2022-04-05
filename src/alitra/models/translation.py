from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alitra.models import Frame


@dataclass
class Translation:
    """
    A translation between two frames represented as x, y and z
    """

    x: float
    y: float
    from_: Frame
    to_: Frame
    z: float = 0

    def to_array(self) -> np.ndarray:
        """
        :return: Numpy array of translation, shape (3,)
        """
        return np.array([self.x, self.y, self.z], dtype=float)

    @staticmethod
    def from_array(
        translation: np.ndarray,
        from_: Frame,
        to_: Frame,
    ) -> Translation:
        """
        :param translation: Numpy array of translation [x,y,z].
            Needs to be shape (3,)
        :param from_: Frame the translation is coming from
        :param to_: Frame the translation is going to
        """
        if translation.shape != (3,):
            raise ValueError("Translation should have shape (3,)")
        return Translation(
            x=translation[0], y=translation[1], z=translation[2], from_=from_, to_=to_
        )
