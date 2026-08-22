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


def laser_centerline(mask: np.ndarray, settings: Settings) -> tuple[np.ndarray, np.ndarray]:
    """Find the brightest laser pixel per column and return x/y arrays."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.empty(0), np.empty(0)
    centers_x, centers_y = [], []
    half_width = max(1, settings.line_width_px // 2)
    for x in np.unique(xs):
        column = ys[xs == x]
        # Median is stable when the laser is several pixels wide.
        y = float(np.median(column))
        if half_width and len(column) < 1:
            continue
        centers_x.append(float(x))
        centers_y.append(y)
    return np.asarray(centers_x), np.asarray(centers_y)


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
