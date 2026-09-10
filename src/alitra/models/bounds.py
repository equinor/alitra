import math
from dataclasses import dataclass

from .position import Position


@dataclass
class Bounds:
    """
    Bounds defines the cube bounds of which the map should be valid inside.
    Position 1 and 2 should be on the diagonal corners of the bounding cube.
    """

    position1: Position
    position2: Position

    def __post_init__(self):
        if self.position1.frame == self.position2.frame:
            self.frame = self.position1.frame
        else:
            raise TypeError("The frames of the bounding Positions are not the same")
        positions: list = [self.position1, self.position2]
        self.x_max = max(position.x for position in positions)
        self.x_min = min(position.x for position in positions)
        self.y_max = max(position.y for position in positions)
        self.y_min = min(position.y for position in positions)
        self.z_max = max(position.z for position in positions)
        self.z_min = min(position.z for position in positions)

    def position_within_bounds(self, position: Position) -> bool:
        if not position.frame == self.frame:
            raise ValueError(
                f"The position is in {position.frame} frame and the bounds are in {self.frame} frame"
            )
        if position.x < self.x_min or position.x > self.x_max:
            return False
        if position.y < self.y_min or position.y > self.y_max:
            return False
        if position.z < self.z_min or position.z > self.z_max:
            return False
        return True

    def distance_to_position(self, position: Position) -> float:
        """
        The Euclidean distance from the position to the nearest point on the
        bounding cube. Zero if the position is within the bounds.
        """
        if not position.frame == self.frame:
            raise ValueError(
                f"The position is in {position.frame} frame and the bounds are in {self.frame} frame"
            )
        distance_x: float = max(self.x_min - position.x, 0.0, position.x - self.x_max)
        distance_y: float = max(self.y_min - position.y, 0.0, position.y - self.y_max)
        distance_z: float = max(self.z_min - position.z, 0.0, position.z - self.z_max)
        return math.sqrt(
            distance_x * distance_x + distance_y * distance_y + distance_z * distance_z
        )
