# Laser Triangulation Scanner

This is a USB-camera scanner for a red line laser. The camera and laser remain stationary while an SG90-driven rack and pinion translates the object. The HMI detects the laser line, sends servo positions to an Arduino, translates each captured slice, and exports a 3D `PLY` point cloud.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python hmi.py
```

The HMI includes camera and serial-port dropdowns, live laser overlay, calibration fields, manual servo movement, automatic scan range, serial debug log, point-cloud count, and PLY save dialog. It is compatible with the supplied Arduino sketch and uses `9600` baud, sends a plain angle such as `90`, and accepts replies such as `Servo moved to: 90`.

The rack travel calibration is `50 mm / 180 degrees` by default. Change `Travel / 180 deg` if the actual rack displacement differs. `Direction` controls whether increasing servo angle adds or subtracts along the selected scan axis. Do not add the `MOVE` command prefix to the current Arduino sketch; the HMI sends only the number.

## Calibration

The height and 30 degree angle do not by themselves establish metric triangulation. Measure the laser baseline, and calibrate the camera intrinsics. A practical quick calibration is to place a flat board at the reference surface, adjust focal length/principal point until its exported cloud is flat at about 300 mm, then verify with a second known board distance. For reliable results, use OpenCV checkerboard calibration and pass the resulting focal length in pixels; lens distortion should be removed before scanning if the camera has a wide lens.

The plane convention is:

```text
cos(angle) * (X + baseline) - sin(angle) * Z = 0
```

If no points appear, increase laser exposure/brightness, reduce ambient light, or adjust `min_saturation` and `min_value` in `Settings`. The current detector expects a red laser; green lasers need a different HSV hue range.

PLY files can be opened in MeshLab, CloudCompare, Blender, or Open3D.

The linear scan assumes the selected scan axis is parallel to the rack travel. If the object moves in the camera's horizontal direction, leave `Scan axis` at `X`. The 30 cm camera height is the reference-surface distance; accurate metric depth still requires focal length and laser baseline calibration.
# Point-cloud-AI-pannel-inspection-System
