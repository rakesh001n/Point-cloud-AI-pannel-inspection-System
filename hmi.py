"""Graphical HMI for the stationary-camera, linear rack-and-pinion scanner."""

from __future__ import annotations

import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import serial
from PIL import Image, ImageTk
from serial.tools import list_ports

from scanner import Settings, laser_centerline, laser_mask, triangulate, write_ply


class ScannerHMI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Laser Triangulation Scanner")
        self.geometry("1180x760")
        self.minsize(980, 650)
        self.configure(bg="#20252b")
        self.camera: cv2.VideoCapture | None = None
        self.arduino: serial.Serial | None = None
        self.port_map: dict[str, str] = {}
        self.latest_frame: np.ndarray | None = None
        self.latest_points = np.empty((0, 3), dtype=np.float32)
        self.cloud: list[np.ndarray] = []
        self.scanning = False
        self.scan_angles: list[float] = []
        self.build_ui()
        self.refresh_cameras()
        self.refresh_ports()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(30, self.update_preview)

    def build_ui(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#20252b")
        style.configure("TLabel", background="#20252b", foreground="#e7edf2")
        style.configure("TLabelframe", background="#20252b", foreground="#e7edf2")
        style.configure("TLabelframe.Label", background="#20252b", foreground="#72d6c9")
        style.configure("TButton", padding=6)

        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=12)
        left.grid(row=0, column=0, sticky="ns")
        right = ttk.Frame(self, padding=(0, 12, 12, 12))
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=0)
        right.columnconfigure(0, weight=1)

        self.make_connections(left)
        self.make_calibration(left)
        self.make_motion(left)
        self.make_actions(left)

        self.preview = tk.Label(right, bg="#101316", text="Connect a camera to begin", fg="#9daab5")
        self.preview.grid(row=0, column=0, sticky="nsew")
        self.log = tk.Text(right, height=8, bg="#15191d", fg="#b9f4d8", insertbackground="white",
                           relief="flat", state="disabled")
        self.log.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self.status = tk.StringVar(value="Ready")
        ttk.Label(left, textvariable=self.status, wraplength=280, foreground="#72d6c9").pack(fill="x", pady=(10, 0))

    def labeled(self, parent: ttk.Frame, label: str, variable: tk.StringVar, row: int, width: int = 12) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(parent, textvariable=variable, width=width).grid(row=row, column=1, sticky="ew", pady=3)

    def make_connections(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Connections", padding=8)
        box.pack(fill="x")
        box.columnconfigure(1, weight=1)
        self.camera_var = tk.StringVar()
        self.port_var = tk.StringVar()
        ttk.Label(box, text="USB camera").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=3)
        self.camera_combo = ttk.Combobox(box, textvariable=self.camera_var, state="readonly", width=13)
        self.camera_combo.grid(row=0, column=1, pady=3)
        ttk.Button(box, text="Refresh", command=self.refresh_cameras).grid(row=0, column=2, padx=(5, 0))
        ttk.Button(box, text="Connect", command=self.connect_camera).grid(row=1, column=1, sticky="ew", pady=3)
        ttk.Label(box, text="Arduino port").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=3)
        self.port_combo = ttk.Combobox(box, textvariable=self.port_var, state="readonly", width=13)
        self.port_combo.grid(row=2, column=1, pady=3)
        ttk.Button(box, text="Refresh", command=self.refresh_ports).grid(row=2, column=2, padx=(5, 0))
        ttk.Button(box, text="Connect", command=self.connect_arduino).grid(row=3, column=1, sticky="ew", pady=3)

    def make_calibration(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Laser / camera calibration (mm, px)", padding=8)
        box.pack(fill="x", pady=(10, 0))
        self.height = tk.StringVar(value="300")
        self.angle = tk.StringVar(value="30")
        self.baseline = tk.StringVar(value="100")
        self.focal = tk.StringVar(value="700")
        for row, label, var in ((0, "Camera height", self.height), (1, "Laser angle", self.angle),
                                (2, "Laser baseline", self.baseline), (3, "Focal length", self.focal)):
            self.labeled(box, label, var, row)

    def make_motion(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Rack and pinion", padding=8)
        box.pack(fill="x", pady=(10, 0))
        self.start = tk.StringVar(value="0")
        self.end = tk.StringVar(value="180")
        self.step = tk.StringVar(value="5")
        self.mm_per_180 = tk.StringVar(value="50")
        self.settle = tk.StringVar(value="250")
        for row, label, var in ((0, "Start servo deg", self.start), (1, "End servo deg", self.end),
                                (2, "Step deg", self.step), (3, "Travel / 180 deg", self.mm_per_180),
                                (4, "Settle time ms", self.settle)):
            self.labeled(box, label, var, row)
        ttk.Label(box, text="Scan axis").grid(row=5, column=0, sticky="w", pady=3)
        self.axis = ttk.Combobox(box, values=("X", "Y", "Z"), state="readonly", width=10)
        self.axis.set("X")
        self.axis.grid(row=5, column=1, sticky="ew", pady=3)
        ttk.Label(box, text="Direction").grid(row=6, column=0, sticky="w", pady=3)
        self.direction = ttk.Combobox(box, values=("+", "-"), state="readonly", width=10)
        self.direction.set("+")
        self.direction.grid(row=6, column=1, sticky="ew", pady=3)

    def make_actions(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Scan", padding=8)
        box.pack(fill="x", pady=(10, 0))
        self.manual_angle = tk.StringVar(value="0")
        self.labeled(box, "Manual servo deg", self.manual_angle, 0)
        ttk.Button(box, text="Move servo", command=self.move_manual).grid(row=1, column=0, columnspan=2, sticky="ew", pady=3)
        ttk.Button(box, text="Capture slice", command=self.capture_slice).grid(row=2, column=0, columnspan=2, sticky="ew", pady=3)
        self.scan_button = ttk.Button(box, text="Start automatic scan", command=self.start_scan)
        self.scan_button.grid(row=3, column=0, columnspan=2, sticky="ew", pady=3)
        ttk.Button(box, text="Stop scan", command=self.stop_scan).grid(row=4, column=0, columnspan=2, sticky="ew", pady=3)
        ttk.Button(box, text="Clear cloud", command=self.clear_cloud).grid(row=5, column=0, columnspan=2, sticky="ew", pady=3)
        ttk.Button(box, text="Export PLY", command=self.export_cloud).grid(row=6, column=0, columnspan=2, sticky="ew", pady=3)
        ttk.Label(box, text="The automatic scan sends MOVE <degrees> to Arduino.", wraplength=260).grid(row=7, column=0, columnspan=2, pady=(8, 0))

    def log_message(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", f"{time.strftime('%H:%M:%S')}  {message}\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def refresh_cameras(self) -> None:
        found = []
        for index in range(10):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                found.append(str(index))
            cap.release()
        self.camera_combo["values"] = found
        if found and self.camera_var.get() not in found:
            self.camera_var.set(found[0])
        self.log_message(f"Cameras found: {', '.join(found) or 'none'}")

    def refresh_ports(self) -> None:
        self.port_map = {}
        displays = []
        for port in list_ports.comports():
            description = (port.description or "").strip()
            manufacturer = (port.manufacturer or "").strip()
            friendly_name = description if description and description.lower() != "n/a" else manufacturer
            if not friendly_name:
                friendly_name = "Serial device"
            display = f"{friendly_name} - {port.device}"
            if port.vid is not None and port.pid is not None:
                display += f" [VID:{port.vid:04X} PID:{port.pid:04X}]"
            self.port_map[display] = port.device
            displays.append(display)
        self.port_combo["values"] = displays
        if displays and self.port_var.get() not in displays:
            self.port_var.set(displays[0])
        self.log_message(f"Serial devices found: {', '.join(displays) or 'none'}")

    def connect_camera(self) -> None:
        if self.camera is not None:
            self.camera.release()
        if not self.camera_var.get():
            messagebox.showerror("Camera", "Select a camera first")
            return
        self.camera = cv2.VideoCapture(int(self.camera_var.get()))
        if not self.camera.isOpened():
            self.camera = None
            messagebox.showerror("Camera", "Could not open the selected camera")
            return
        self.log_message(f"Camera {self.camera_var.get()} connected")
        self.status.set("Camera connected")

    def connect_arduino(self) -> None:
        try:
            if self.arduino is not None:
                self.arduino.close()
            display = self.port_var.get()
            device = self.port_map.get(display, display)
            if not device:
                raise ValueError("Select an Arduino port first")
            # Matches the user's existing Arduino sketch.
            self.arduino = serial.Serial(device, 9600, timeout=1)
            time.sleep(2)
            self.log_message(f"Arduino connected: {display}")
            self.status.set("Camera and Arduino ready" if self.camera else "Arduino connected; connect camera")
        except (ValueError, serial.SerialException) as error:
            self.arduino = None
            messagebox.showerror("Arduino", str(error))
            self.log_message(f"Arduino connection failed: {error}")

    def settings(self) -> Settings:
        return Settings(float(self.height.get()), float(self.angle.get()), float(self.baseline.get()), float(self.focal.get()))

    def update_preview(self) -> None:
        if self.camera is not None and self.camera.isOpened():
            ok, frame = self.camera.read()
            if ok:
                self.latest_frame = frame
                settings = self.settings_safe()
                mask = laser_mask(frame, settings)
                xs, ys = laser_centerline(mask, settings)
                self.latest_points = triangulate(xs, ys, frame.shape[1], frame.shape[0], settings)
                frame[mask > 0] = (0, 255, 255)
                cv2.putText(frame, f"laser points: {len(self.latest_points)} | cloud: {self.cloud_size()}",
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image.thumbnail((850, 620))
                photo = ImageTk.PhotoImage(image)
                self.preview.configure(image=photo, text="")
                self.preview.image = photo
        self.after(30, self.update_preview)

    def settings_safe(self) -> Settings:
        try:
            return self.settings()
        except ValueError:
            return Settings()

    def cloud_size(self) -> int:
        return sum(len(points) for points in self.cloud)

    def send_move(self, angle: float) -> bool:
        if self.arduino is None or not self.arduino.is_open:
            self.log_message("No Arduino connected; movement command skipped")
            return False
        command = f"{max(0, min(180, angle)):.0f}\n"
        try:
            self.arduino.reset_input_buffer()
            self.arduino.write(command.encode("ascii"))
            reply = self.arduino.readline().decode("ascii", errors="replace").strip()
            self.log_message(f"> {command.strip()}   < {reply or 'no reply'}")
            return reply.startswith("Servo moved to:")
        except serial.SerialException as error:
            self.log_message(f"Serial error: {error}")
            return False

    def move_manual(self) -> None:
        try:
            angle = float(self.manual_angle.get())
            if not 0 <= angle <= 180:
                raise ValueError
        except ValueError:
            messagebox.showerror("Servo angle", "Enter an angle from 0 to 180")
            return
        self.send_move(angle)

    def displacement(self, angle: float) -> float:
        return angle / 180.0 * float(self.mm_per_180.get()) * (1 if self.direction.get() == "+" else -1)

    def capture_slice(self, angle: float | None = None) -> bool:
        if self.latest_frame is None or len(self.latest_points) == 0:
            self.log_message("Capture skipped: no laser line detected")
            self.status.set("No laser line detected")
            return False
        try:
            current_angle = float(self.manual_angle.get()) if angle is None else angle
            distance = self.displacement(current_angle)
            axis = "XYZ".index(self.axis.get())
            translated = self.latest_points.copy()
            translated[:, axis] += distance
            self.cloud.append(translated)
            self.log_message(f"Captured {len(translated)} points at {current_angle:.1f} deg, displacement {distance:.2f} mm")
            self.status.set(f"Cloud contains {self.cloud_size()} points")
            return True
        except (ValueError, KeyError):
            messagebox.showerror("Scan settings", "Check the numeric motion and calibration fields")
            return False

    def start_scan(self) -> None:
        if self.scanning:
            return
        try:
            start, end, step = float(self.start.get()), float(self.end.get()), abs(float(self.step.get()))
            if not 0 <= start <= 180 or not 0 <= end <= 180 or step == 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Scan range", "Use servo angles from 0 to 180 and a non-zero step")
            return
        if self.camera is None:
            messagebox.showerror("Scan", "Connect a camera first")
            return
        if self.arduino is None:
            messagebox.showerror("Scan", "Connect the Arduino first")
            return
        self.scanning = True
        self.scan_angles = list(np.arange(start, end + (step / 2), step if end >= start else -step))
        self.scan_button.configure(state="disabled")
        self.log_message(f"Automatic scan started: {len(self.scan_angles)} positions")
        self.run_next_position(0)

    def run_next_position(self, index: int) -> None:
        if not self.scanning or index >= len(self.scan_angles):
            self.scanning = False
            self.scan_button.configure(state="normal")
            self.status.set(f"Scan complete: {self.cloud_size()} points")
            self.log_message("Automatic scan complete")
            return
        angle = self.scan_angles[index]
        self.manual_angle.set(f"{angle:.1f}")
        if not self.send_move(angle):
            self.stop_scan()
            return
        try:
            delay = max(0, int(float(self.settle.get())))
        except ValueError:
            delay = 250
        self.status.set(f"Moving to {angle:.1f} deg, settling...")
        self.after(delay, lambda: self.capture_and_continue(angle, index))

    def capture_and_continue(self, angle: float, index: int) -> None:
        if self.scanning:
            self.capture_slice(angle)
            self.run_next_position(index + 1)

    def stop_scan(self) -> None:
        self.scanning = False
        self.scan_button.configure(state="normal")
        self.log_message("Scan stopped")

    def clear_cloud(self) -> None:
        self.cloud.clear()
        self.status.set("Cloud cleared")
        self.log_message("Point cloud cleared")

    def export_cloud(self) -> None:
        if not self.cloud:
            messagebox.showwarning("Export", "There are no captured points")
            return
        path = filedialog.asksaveasfilename(defaultextension=".ply", filetypes=(("PLY point cloud", "*.ply"),))
        if path:
            write_ply(Path(path), np.vstack(self.cloud))
            self.log_message(f"Exported {self.cloud_size()} points to {path}")
            self.status.set(f"Saved {self.cloud_size()} points")

    def close(self) -> None:
        self.scanning = False
        if self.camera is not None:
            self.camera.release()
        if self.arduino is not None:
            self.arduino.close()
        self.destroy()


if __name__ == "__main__":
    ScannerHMI().mainloop()
