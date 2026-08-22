import numpy as np

from scanner import Settings, rotate_slice, triangulate, write_ply


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
