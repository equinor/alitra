from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from dacite import from_dict

from .bounds import Bounds
from .frame import Frame
from .position import Positions


@dataclass
class Map:
    """
    Map contains a set of coordinates in a given frame. Two maps can be used to create
    a transform. A map may also contain bounds.
    """

    name: str
    frame: Frame
    reference_positions: Positions
    bounds: Bounds = None

    @staticmethod
    def from_config(map_config_path: Path) -> Map:
        """
        Loads a Map from a json-file using dacite
        """
        with open(map_config_path) as json_file:
            map_config_dict = json.load(json_file)

        return from_dict(data_class=Map, data=map_config_dict)


@dataclass
class MapAlignment:
    """
    MapAlignment contains two maps that can be used to create a transform
    """

    name: str
    map_from: Map
    map_to: Map

    @staticmethod
    def from_config(map_config_path: Path) -> MapAlignment:
        """
        Loads a MapAlignment from a json-file using dacite
        """
        with open(map_config_path) as json_file:
            map_config_dict = json.load(json_file)

        return from_dict(data_class=MapAlignment, data=map_config_dict)
