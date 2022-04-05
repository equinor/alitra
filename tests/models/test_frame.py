from alitra import Frame


def test_frame():
    expected_frame = Frame("test")
    frame: Frame = Frame("test")
    assert frame == expected_frame
