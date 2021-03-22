# type: ignore
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from alitra import AlignFrames, Point, PointList

""" Manual alignment. Rotation about the z-axis:
1- Find 2-4 matching coordinates in both Echo and the local robot map (
2- For coordinates in Echo use the marker tool
3- For coordinates in robot-frame use the robot map and manaully read coordinates either using ROS or
load it into python as in this example.
4- Match the coordinates using alignFrames.align_frames.
"""


def manual_alignment_example() -> None:
    path_to_map = "./examples/localization-inspector.png"
    path_to_echo_img = "./examples/Echo.png"

    map = plt.imread(path_to_map)
    fig, axs = plt.subplots(2)

    resolution = 0.02
    origin_x = 0
    origin_y = 0
    extent = [
        origin_y,
        np.size(map, 1) * resolution + origin_y,
        origin_x,
        np.size(map, 0) * resolution + origin_x,
    ]
    axs[0].imshow(map, cmap="gray", extent=extent)

    p_loc = PointList(
        points=[
            Point(x=3.44, y=9.1, frame="robot"),
            Point(x=1.8, y=12.44, frame="robot"),
            Point(x=1.6, y=19.05, frame="robot"),
        ],
        frame="robot",
    )
    for coordinates in p_loc.points:
        axs[0].plot(coordinates.x, coordinates.y, marker="o")
        axs[0].annotate(
            f"{coordinates.x} ,{coordinates.y}", (coordinates.x, coordinates.y + 0.2)
        )
    p_glob = PointList(
        points=[
            Point(x=20198.4, y=5247.2, frame="asset"),
            Point(x=20196.940, y=5250.344, frame="asset"),
            Point(x=20196.63, y=5256.961, frame="asset"),
        ],
        frame="asset",
    )

    img = mpimg.imread(path_to_echo_img)
    imgplot = axs[1].imshow(img)
    axs[1].axis("off")

    plt.show()

    cf_frame = AlignFrames.align_frames(p_loc, p_glob, "z")
    # Verifying that estimations are decent:
    norm_transform_error = np.linalg.norm(
        cf_frame.transform_point(p_glob, from_="asset", to_="robot").as_np_array()
        - p_loc.as_np_array(),
        axis=1,
    )
    assert min(norm_transform_error) < 0.3
    print(f"root mean square distance error: {norm_transform_error}")


def main():
    manual_alignment_example()


if __name__ == "__main__":
    main()
