"""USB-camera laser triangulation scanner.

Coordinate system: X is camera-right, Y is camera-down, Z points from the
camera toward the reference surface. All distances are millimetres.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class Settings:
    camera_height_mm: float = 300.0
    laser_angle_deg: float = 30.0
    laser_baseline_mm: float = 100.0
    focal_length_px: float = 700.0
    principal_x_px: float = 0.0
    principal_y_px: float = 0.0
    min_saturation: int = 90
    min_value: int = 130
    line_width_px: int = 5


def laser_mask(frame: np.ndarray, settings: Settings) -> np.ndarray:
    """Return a cleaned mask for a red laser line."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Red wraps at the ends of OpenCV's hue range.
    mask = cv2.inRange(hsv, (0, settings.min_saturation, settings.min_value), (15, 255, 255))
    mask |= cv2.inRange(hsv, (165, settings.min_saturation, settings.min_value), (179, 255, 255))
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def median_preprocess(frame: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Return a median-filtered copy of a camera frame."""
    if frame is None or frame.size == 0:
        raise ValueError("Cannot filter an empty image")
    if kernel_size not in (3, 5, 7) or kernel_size % 2 == 0:
        raise ValueError("Median kernel size must be 3, 5, or 7")
    if frame.ndim < 2 or min(frame.shape[:2]) < kernel_size:
        raise ValueError("Median kernel must fit inside the image")
    return cv2.medianBlur(frame.copy(), kernel_size)


def laser_centerline(
    mask: np.ndarray,
    settings: Settings,
    frame: Optional[np.ndarray] = None,
    method: str = "median",
    window_radius: int = 3,
    min_confidence: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the laser coordinate per column.

    ``median`` preserves the original detector. ``subpixel`` computes an
    intensity-weighted centroid around the masked stripe and can return
    fractional image Y coordinates. Invalid or low-confidence columns are
    discarded rather than interpolated across object edges.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.empty(0), np.empty(0)
    if method not in ("median", "subpixel"):
        raise ValueError(f"Unknown laser centerline method: {method}")
    centers_x, centers_y = [], []
    for x in np.unique(xs):
        column = ys[xs == x]
        if method == "median" or frame is None:
            y = float(np.median(column))
        else:
            center = float(np.median(column))
            low = max(0, int(math.floor(center)) - max(1, window_radius))
            high = min(mask.shape[0], int(math.ceil(center)) + max(1, window_radius) + 1)
            local_y = np.arange(low, high)
            valid = mask[local_y, x] > 0
            local_y = local_y[valid]
            if len(local_y) < 2:
                continue
            if frame.ndim == 3:
                # Red laser energy is primarily in the BGR red channel.
                intensity = frame[local_y, x, 2].astype(np.float64)
            else:
                intensity = frame[local_y, x].astype(np.float64)
            if not len(intensity) or np.max(intensity) >= 254:
                continue
            baseline = np.percentile(intensity, 20)
            weights = np.maximum(intensity - baseline, 0.0)
            total = float(weights.sum())
            peak = float(weights.max())
            if total <= 0 or peak / total < min_confidence:
                continue
            y = float(np.sum(local_y * weights) / total)
        centers_x.append(float(x))
        centers_y.append(y)
    return np.asarray(centers_x), np.asarray(centers_y)


def savitzky_golay_profile(points: np.ndarray, window_length: int = 7, polyorder: int = 2) -> np.ndarray:
    """Smooth ordered profile depth while preserving X/Y correspondence."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Points must have shape (N, 3)")
    if len(points) == 0:
        return points.copy()
    if window_length <= polyorder or window_length % 2 == 0:
        raise ValueError("Savitzky-Golay window must be odd and greater than polynomial order")
    effective_window = min(window_length, len(points) if len(points) % 2 else len(points) - 1)
    if effective_window <= polyorder:
        return points.copy()
    from scipy.signal import savgol_filter
    filtered = points.astype(np.float32, copy=True)
    filtered[:, 2] = savgol_filter(filtered[:, 2], effective_window, polyorder).astype(np.float32)
    return filtered


def kalman_profile(points: np.ndarray, process_noise: float = 1e-2, measurement_noise: float = 1.0) -> np.ndarray:
    """Apply a scalar Kalman filter to ordered profile depth values."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Points must have shape (N, 3)")
    if process_noise < 0 or measurement_noise <= 0:
        raise ValueError("Process noise must be non-negative and measurement noise positive")
    filtered = points.astype(np.float32, copy=True)
    valid = np.isfinite(filtered[:, 2])
    if not np.any(valid):
        return filtered[:0]
    first = int(np.flatnonzero(valid)[0])
    estimate = float(filtered[first, 2])
    covariance = measurement_noise
    for index in range(first, len(filtered)):
        if not valid[index]:
            continue
        covariance += process_noise
        gain = covariance / (covariance + measurement_noise)
        estimate += gain * (float(filtered[index, 2]) - estimate)
        covariance = (1.0 - gain) * covariance
        filtered[index, 2] = estimate
    return filtered[valid]


def statistical_outlier_removal(points: np.ndarray, neighbors: int = 20, std_ratio: float = 2.0) -> np.ndarray:
    """Return a new cloud with isolated points removed using Open3D SOR."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Points must have shape (N, 3)")
    if neighbors <= 0 or std_ratio <= 0:
        raise ValueError("SOR neighbors and standard-deviation ratio must be positive")
    if len(points) <= 2:
        return points.astype(np.float32, copy=True)
    import open3d as o3d
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    effective_neighbors = min(neighbors, len(points) - 1)
    filtered, _ = cloud.remove_statistical_outlier(effective_neighbors, std_ratio)
    return np.asarray(filtered.points, dtype=np.float32).copy()


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Return a new cloud downsampled with an Open3D voxel grid."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Points must have shape (N, 3)")
    if voxel_size <= 0:
        raise ValueError("Voxel size must be positive")
    if len(points) == 0:
        return points.astype(np.float32, copy=True)
    import open3d as o3d
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    filtered = cloud.voxel_down_sample(voxel_size)
    return np.asarray(filtered.points, dtype=np.float32).copy()


def triangulate(xs: np.ndarray, ys: np.ndarray, width: int, height: int, settings: Settings) -> np.ndarray:
    """Intersect camera rays with a vertical laser plane.

    The laser emitter is assumed to be laser_baseline_mm to the camera's left.
    The angle is the angle of the laser plane from the camera optical axis. A
    negative baseline is valid for an emitter on the other side.
    """
    if not len(xs):
        return np.empty((0, 3), dtype=np.float32)
    fx = settings.focal_length_px
    cx = settings.principal_x_px if settings.principal_x_px else width / 2.0
    cy = settings.principal_y_px if settings.principal_y_px else height / 2.0
    theta = math.radians(settings.laser_angle_deg)
    # Plane: cos(theta) * (X + baseline) - sin(theta) * Z = 0.
    nx, nz = math.cos(theta), -math.sin(theta)
    ray_x = (xs - cx) / fx
    ray_y = (ys - cy) / fx
    denom = nx * ray_x + nz
    valid = np.abs(denom) > 1e-6
    depth = np.zeros_like(ray_x, dtype=np.float64)
    depth[valid] = -nx * settings.laser_baseline_mm / denom[valid]
    valid &= depth > 0
    points = np.column_stack((ray_x * depth, ray_y * depth, depth))
    return points[valid].astype(np.float32)


def draw_overlay(frame: np.ndarray, mask: np.ndarray, points: np.ndarray) -> np.ndarray:
    out = frame.copy()
    out[mask > 0] = (0, 255, 255)
    cv2.putText(out, f"line pixels: {len(points)}", (16, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def write_ply(path: Path, points: np.ndarray) -> None:
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for x, y, z in points:
            handle.write(f"{x:.4f} {y:.4f} {z:.4f}\n")


def rotate_slice(points: np.ndarray, angle_deg: float, axis_z: float) -> np.ndarray:
    """Move a camera-frame slice into turntable coordinates."""
    angle = math.radians(angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    relative = points - np.array([0.0, 0.0, axis_z], dtype=np.float32)
    rotated = np.column_stack((c * relative[:, 0] + s * relative[:, 2],
                               relative[:, 1],
                               -s * relative[:, 0] + c * relative[:, 2]))
    return rotated + np.array([0.0, 0.0, axis_z], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Laser triangulation point-cloud scanner")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--output", default="scan.ply", help="PLY output path")
    parser.add_argument("--height", type=float, default=300.0, help="Camera height/reference distance in mm")
    parser.add_argument("--angle", type=float, default=30.0, help="Laser plane angle in degrees")
    parser.add_argument("--baseline", type=float, default=100.0, help="Signed laser offset from camera in mm")
    parser.add_argument("--focal-length", type=float, default=700.0, help="Calibrated focal length in pixels")
    parser.add_argument("--turn-step", type=float, default=5.0,
                        help="Turntable angle advanced per SPACE capture, in degrees")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    settings = Settings(args.height, args.angle, args.baseline, args.focal_length)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open USB camera index {args.camera}")
    all_points: list[np.ndarray] = []
    scan_angle = 0.0
    saved_message = ""
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera frame could not be read")
            mask = laser_mask(frame, settings)
            xs, ys = laser_centerline(mask, settings)
            points = triangulate(xs, ys, frame.shape[1], frame.shape[0], settings)
            view = draw_overlay(frame, mask, points)
            status = (f"cloud: {sum(len(p) for p in all_points)} | angle: {scan_angle:.1f} deg | "
                      "SPACE capture | E export | C clear | Q quit")
            cv2.putText(view, status, (16, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 2, cv2.LINE_AA)
            if saved_message:
                cv2.putText(view, saved_message, (16, 58), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Laser triangulation scanner", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" ") and len(points):
                all_points.append(rotate_slice(points, scan_angle, settings.camera_height_mm))
                scan_angle += args.turn_step
                saved_message = f"Captured {len(points)} points"
            elif key == ord("c"):
                all_points.clear()
                scan_angle = 0.0
                saved_message = "Cloud cleared"
            elif key == ord("e"):
                cloud = np.vstack(all_points) if all_points else np.empty((0, 3), dtype=np.float32)
                write_ply(Path(args.output), cloud)
                saved_message = f"Saved {len(cloud)} points to {args.output}"
            elif key != 255:
                saved_message = ""
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run(parse_args())
