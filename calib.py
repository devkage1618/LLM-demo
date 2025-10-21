import numpy as np
import cv2
import os
from main import *
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from ultralytics import YOLO
from pathlib import Path

class PanoramaViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Panorama Viewer")
        self.video_path = Path("input.mp4")
        self.cap = None
        self.frame_count = 0
        self.fps = 30
        self.yaw_max = 60
        self.roll_max = 30
        self.pitch_max = 30
        self.model = YOLO('model/yolov8m.pt')

        self.setup_gui()

    def setup_gui(self):
        container = ttk.Frame(self.root, padding=10)
        container.pack(fill="both", expand=True)
        
        # Left and Right columns
        right_col = ttk.Frame(container)
        right_col.pack(side="right", fill="both", expand=True, padx=5)

        left_col = ttk.Frame(container)
        left_col.pack(side="left", fill="both", expand=True, padx=5)
        
        # Frame selector
        frame = ttk.Frame(right_col, padding=5)
        frame.pack(fill="x", pady=5)
        ttk.Label(frame, text="Frame:").pack(side="left")
        self.frame_slider = ttk.Scale(frame, from_=0, to=100, orient="horizontal", command=self.on_frame_change)
        self.frame_slider.pack(fill="x", expand=True, side="left", padx=5)
        self.frame_label = ttk.Label(frame, text="0")
        self.frame_label.pack(side="left")

        # Left-Right
        yaw_frame = ttk.Frame(right_col, padding=5)
        yaw_frame.pack(fill="x",pady=5)
        ttk.Label(yaw_frame, text="Left-Right (°):").pack(side="left")
        self.yaw_slider = ttk.Scale(yaw_frame, from_=-self.yaw_max, to=self.yaw_max, orient="horizontal")
        self.yaw_slider.set(0)
        self.yaw_slider.pack(fill="x", expand=True, side="left", padx=5)
        self.yaw_label = ttk.Label(yaw_frame, text="0")
        self.yaw_label.pack(side="left")

        self.yaw_slider.bind("<Motion>", lambda e: self.yaw_label.config(text=f"{self.yaw_slider.get():.0f}"))

        # Preview button
        preview_btn = ttk.Button(right_col, text="Preview", command=self.preview_frame)
        preview_btn.pack(pady=10)

        # Zoom
        fov_frame = ttk.Frame(left_col, padding=5)
        fov_frame.pack(fill="x",pady=5)
        ttk.Label(fov_frame, text="Zoom (-/+):").pack(side="left")
        self.fov_slider = ttk.Scale(fov_frame, from_=150, to=30, orient="horizontal")
        self.fov_slider.set(80)
        self.fov_slider.pack(fill="x", expand=True, side="left", padx=5)
        self.fov_label = ttk.Label(fov_frame, text="80")
        self.fov_label.pack(side="left")

        self.fov_slider.bind("<Motion>", lambda e: self.fov_label.config(text=f"{self.fov_slider.get():.0f}")) 

        # ClockWise Rotate
        roll_frame = ttk.Frame(left_col, padding=5)
        roll_frame.pack(fill="x",pady=5)
        ttk.Label(roll_frame, text="ClockWise Roll(°):").pack(side="left")
        self.roll_slider = ttk.Scale(roll_frame, from_=0, to=45, orient="horizontal")
        self.roll_slider.set(self.roll_max)
        self.roll_slider.pack(fill="x", expand=True, side="left", padx=5)
        self.roll_label = ttk.Label(roll_frame, text=str(self.roll_max))
        self.roll_label.pack(side="left")
        self.roll_slider.bind("<Motion>", lambda e: self.roll_label.config(text=f"{self.roll_slider.get():.0f}"))

        # Look Up/Down
        pitch_frame = ttk.Frame(left_col, padding=5)
        pitch_frame.pack(fill="x",pady=5)
        ttk.Label(pitch_frame, text="Look Up/Down (°):").pack(side="left")
        self.pitch_slider = ttk.Scale(pitch_frame, from_=-self.pitch_max, to=self.pitch_max, orient="horizontal")
        self.pitch_slider.set(0)
        self.pitch_slider.pack(fill="x", expand=True, side="left", padx=5)
        self.pitch_label = ttk.Label(pitch_frame, text="0")
        self.pitch_label.pack(side="left")

        self.pitch_slider.bind("<Motion>", lambda e: self.pitch_label.config(text=f"{self.pitch_slider.get():.0f}"))

         # Brightness / Gamma
        gamma_frame = ttk.Frame(left_col, padding=5)
        gamma_frame.pack(fill="x",pady=5)
        ttk.Label(gamma_frame, text="Dark/Light:").pack(side="left")
        self.gamma_slider = ttk.Scale(gamma_frame, from_=0.3, to=2.0, orient="horizontal")
        self.gamma_slider.set(1.0)
        self.gamma_slider.pack(fill="x", expand=True, side="left", padx=5)
        self.gamma_label = ttk.Label(gamma_frame, text="1.0")
        self.gamma_label.pack(side="left")
        self.gamma_slider.bind("<Motion>", lambda e: self.gamma_label.config(text=f"{self.gamma_slider.get():.1f}"))

        # Crop Height Slider
        crop_hframe = ttk.Frame(left_col, padding=5)
        crop_hframe.pack(fill="x", pady=5)
        ttk.Label(crop_hframe, text="Crop Height:").pack(side="left")
        self.crop_hslider = ttk.Scale(crop_hframe, from_=720, to=1600, orient="horizontal")
        self.crop_hslider.set(1440)
        self.crop_hslider.pack(fill="x", expand=True, side="left", padx=5)
        self.crop_hlabel = ttk.Label(crop_hframe, text=str(1440))
        self.crop_hlabel.pack(side="left")
        self.crop_hslider.bind("<Motion>", lambda e: self.crop_hlabel.config(text=f"{int(self.crop_hslider.get())}"))

         # File control buttons
        file_controls = ttk.Frame(self.root, padding=10)
        file_controls.pack(pady=10)

        select_video_btn = ttk.Button(file_controls, text="Open", command=self.load_video)
        select_video_btn.pack(side="left", padx=3)

        save_image_btn = ttk.Button(file_controls, text="Save", command=self.save_config)
        save_image_btn.pack(side="left", padx=30)
  
    def on_frame_change(self, value):
        self.frame_label.config(text=str(int(float(value))))

    def load_video(self):
        filepath = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        if filepath:
            self.video_path = Path(filepath)
            self.cap = cv2.VideoCapture(filepath)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Failed to open video.")
                return

            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_slider.configure(to=self.frame_count - 3)
        
            config_path = self.video_path.with_name(f"{self.video_path.stem}_config.txt")
            if os.path.exists(config_path):
                config = {}
                with open(config_path, 'r') as f:
                    for line in f:
                        if '=' not in line:
                            continue
                        key, value = line.strip().split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                        config[key] = value
                self.crop_hslider.set(config.get('crop_h', 1440))
                self.fov_slider.set(config.get('fov', 70))
                self.pitch_slider.set(config.get('pitch', -10))
                self.roll_slider.set(config.get('roll_max', 30))
                self.gamma_slider.set(config.get('gamma', 1.0))
                self.fov_label.config(text=f"{self.fov_slider.get():.0f}")
                self.pitch_label.config(text=f"{self.pitch_slider.get():.0f}")
                self.roll_label.config(text=f"{self.roll_slider.get():.0f}")
                self.gamma_label.config(text=f"{self.gamma_slider.get():.1f}")
                self.crop_hlabel.config(text=f"{self.crop_hslider.get():.1f}")
        self.preview_frame()

    
    def save_config(self):
        if not self.cap:
            messagebox.showwarning("Warning", "No video loaded.")
            return

        config_path = self.video_path.with_name(f"{self.video_path.stem}_config.txt")
        # Define your config dictionary
        config = {
            'crop_h': int(self.crop_hslider.get()),
            'fov': float(self.fov_slider.get()),
            'pitch': float(self.pitch_slider.get()),
            'roll_max': float(self.roll_slider.get()),
            'gamma': float(self.gamma_slider.get())
        }

        # Save it to a file
        with open(config_path, 'w') as f:
            for key, value in config.items():
                f.write(f"{key} = {value}\n")

    def preview_frame(self):
        if not self.cap:
            messagebox.showerror("Error", f"Please load the video.")
            return

        frame_idx = int(self.frame_slider.get())
        fov = float(self.fov_slider.get())
        yaw = float(self.yaw_slider.get())
        pitch = float(self.pitch_slider.get())
        roll_max = float(self.roll_slider.get())
        gamma = float(self.gamma_slider.get())
        crop_h = int(self.crop_hslider.get())

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, pano = self.cap.read()
        if not ret:
            messagebox.showerror("Error", f"Could not read frame {frame_idx}")
            return
        
        border_line_y = pitch_orientation(pano)
        centerx = pano.shape[1] // 2
        start_point=(centerx-100, border_line_y)
        end_point=(centerx+100, border_line_y)
        cv2.line(pano, start_point, end_point, color=(255,0,255), thickness=3)

        roll_deg = np.interp(yaw, [-self.yaw_max, 0, self.yaw_max], [roll_max, 0, -roll_max])

        persp = half_cylindrical_panorama_to_perspective(
            pano,
            out_w=3200,
            out_h=1800,
            crop_bottom=1800,
            out_fov_x_deg=fov,
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=roll_deg
        )

        lookUpTable = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                                for i in np.arange(0, 256)]).astype("uint8")
        persp = cv2.LUT(persp, lookUpTable)

        persp_height, persp_width = persp.shape[:2]
        print(f"{persp_height},{persp_width}")

        center_x = int(persp_width / 2)
        center_y = int(persp_height / 2)
        dot_length = 20         # Length of each dot
        space_between = 10     # Space between dots
        for y in range(0, persp_height, dot_length + space_between):
            cv2.line(persp, (center_x, y), (center_x, min(y + dot_length, persp_height)), (255, 0, 0), 1)
        for x in range(0, persp_width, dot_length + space_between):
            cv2.line(persp, (x, center_y), (min(x + dot_length, persp_width), center_y),  (255, 0, 0), 1)

        crop_w = crop_h * persp_width / persp_height
        border_left = int((persp_width - crop_w) / 2)
        border_right = int(border_left + crop_w)
        top_left = (border_left, persp_height - crop_h)
        bottom_right = (border_right, persp_height)
        cv2.rectangle(persp, top_left, bottom_right, (255,0,0), thickness=2)

        view_w = 1600
        view_h = int(view_w * persp_height / persp_width)
        persp = cv2.resize(persp, (view_w, view_h))
        # Draw results
        detect_results = self.model(persp)
        for r in detect_results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.model.names[cls_id]

                if label in ['person', 'sports ball']:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = (0, 255, 0) if label == 'person' else (0, 0, 255)
                    cv2.rectangle(persp, (x1, y1), (x2, y2), color, 1)
                    cv2.putText(persp, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow("Perspective Preview", persp)
        cv2.waitKey(1)

# Run the GUI
if __name__ == '__main__':
    root = tk.Tk()
    app = PanoramaViewer(root)
    root.mainloop()
