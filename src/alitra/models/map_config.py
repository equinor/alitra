import json
from dataclasses import dataclass
from pathlib import Path

from dacite import from_dict

from alitra.frame_dataclasses import PointList
from alitra.models.bounds import Bounds


@dataclass
class MapConfig:
    map_name: str
    robot_reference_points: PointList
    asset_reference_points: PointList
    bounds: Bounds = None


def load_map_config(map_config_path: Path) -> MapConfig:
    with open(map_config_path) as json_file:
        map_config_dict = json.load(json_file)

    return from_dict(
        data_class=MapConfig,
        data=map_config_dict,
    )
