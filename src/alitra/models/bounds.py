from dataclasses import dataclass

from alitra.frame_dataclasses import Point


# Point 1 and 2 should be on the diagonal corners of the bounding cube
@dataclass
class Bounds:
    point1: Point
    point2: Point

    def __post_init__(self):
        if self.point1.frame == self.point2.frame:
            self.frame = self.point1.frame
        else:
            raise FrameException("The frames of the bounding points are not the same")
        points: list = [self.point1, self.point2]
        self.x_max = max(point.x for point in points)
        self.x_min = min(point.x for point in points)
        self.y_max = max(point.y for point in points)
        self.y_min = min(point.y for point in points)
        self.z_max = max(point.z for point in points)
        self.z_min = min(point.z for point in points)

    def point_within_bounds(self, point: Point) -> bool:
        if not point.frame == self.frame:
            raise FrameException(
                f"The point is in {point.frame} frame and the bounds are in {self.frame} frame"
            )
        if point.x < self.x_min or point.x > self.x_max:
            return False
        if point.y < self.y_min or point.y > self.y_max:
            return False
        if point.z < self.z_min or point.z > self.z_max:
            return False
        return True


class FrameException(Exception):
    pass
