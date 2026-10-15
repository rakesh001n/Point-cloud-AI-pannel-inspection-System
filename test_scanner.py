import numpy as np

from scanner import (
    Settings, kalman_profile, laser_centerline, median_preprocess, rotate_slice,
    savitzky_golay_profile, triangulate, voxel_downsample, write_ply,
)


def test_triangulation_returns_positive_point_at_expected_depth():
    settings = Settings(laser_angle_deg=30, laser_baseline_mm=100, focal_length_px=700)
    # At the optical centre, z = baseline / -tan(angle) for this convention.
    points = triangulate(np.array([640.0]), np.array([360.0]), 1280, 720, settings)
    assert points.shape == (1, 3)
    assert np.isclose(points[0, 2], 173.205, atol=0.01)


def test_ply_writer_has_header_and_vertices(tmp_path):
    path = tmp_path / "points.ply"
    write_ply(path, np.array([[1, 2, 3]], dtype=np.float32))
    text = path.read_text()
    assert "element vertex 1" in text
    assert "1.0000 2.0000 3.0000" in text


def test_rotation_preserves_turntable_axis():
    points = rotate_slice(np.array([[10, 2, 300]], dtype=np.float32), 90, 300)
    assert np.allclose(points[0], [0, 2, 290], atol=1e-5)


def test_median_filter_preserves_input_and_validates_kernel():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[5, 5] = 255
    filtered = median_preprocess(image, 3)
    assert image[5, 5, 0] == 255
    assert filtered[5, 5, 0] == 0
    try:
        median_preprocess(image, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("even median kernels must be rejected")


def test_subpixel_centerline_returns_fractional_y():
    mask = np.zeros((12, 4), dtype=np.uint8)
    mask[5:8, 2] = 255
    frame = np.zeros((12, 4, 3), dtype=np.uint8)
    frame[5, 2, 2] = 100
    frame[6, 2, 2] = 200
    frame[7, 2, 2] = 150
    xs, ys = laser_centerline(mask, Settings(), frame, method="subpixel")
    assert xs.tolist() == [2.0]
    assert 5.0 < ys[0] < 7.0


def test_profile_filters_keep_shape_and_raw_input():
    points = np.column_stack((np.arange(9), np.zeros(9), np.linspace(10, 20, 9))).astype(np.float32)
    raw = points.copy()
    smooth = savitzky_golay_profile(points, 7, 2)
    kalman = kalman_profile(points, 0.01, 1.0)
    assert smooth.shape == points.shape
    assert kalman.shape == points.shape
    assert np.array_equal(points, raw)


def test_empty_and_invalid_cloud_filters_are_safe():
    empty = np.empty((0, 3), dtype=np.float32)
    assert len(voxel_downsample(empty, 1.0)) == 0
    try:
        kalman_profile(empty, measurement_noise=0)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid Kalman noise must be rejected")
