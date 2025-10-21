import os
import sys
import math
import cv2
import av
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import trim_mean
import gc

def pitch_orientation(frame):
    """Detect the top soccer field border in panorama."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]

    height, width = h_channel.shape
    y_end = int(height * 0.6)
    y_pad = y_end - int(height * 0.5)

    x1_start, x1_end = int(width * 0.425), int(width * 0.475)
    x2_start, x2_end = int(width * 0.525), int(width * 0.575)

    crop1 = h_channel[:y_end, x1_start:x1_end]
    crop2 = h_channel[:y_end, x2_start:x2_end]

    blurred1 = cv2.GaussianBlur(crop1, (7, 7), 0)
    blurred2 = cv2.GaussianBlur(crop2, (7, 7), 0)

    avg_per_row1 = blurred1.mean(axis=1)[::-1]
    avg_per_row2 = blurred2.mean(axis=1)[::-1]

    if len(avg_per_row1) < 150:
        print("Image File Size too small")
        return 0, 0

    avg_first = trim_mean(avg_per_row1[:100], proportiontocut=0.1)
    topborder_soccerfield = pitch_deg = roll_deg = 0

    for idx in range(100, len(avg_per_row1) - 20):
        avg_recent1 = trim_mean(avg_per_row1[idx - 20:idx + 20], proportiontocut=0.1)
        avg_recent2 = trim_mean(avg_per_row2[idx - 20:idx + 20], proportiontocut=0.1)

        min_val, max_val = sorted([avg_recent1, avg_recent2])
        if (max_val - avg_first > 35) or (min_val - avg_first > 15) or \
           (max(avg_per_row1[idx], avg_per_row2[idx]) - avg_first > 50):
            topborder_soccerfield = y_end - idx - 20
            # delta = (idx - y_pad) / height
            # pitch_deg = delta * 40 + 2.5
            # roll_deg = delta * 75 + 7.5
            break
    else:
        print("⚠️ No green hue boundary detected. Pitch remains 0.")

    return topborder_soccerfield


def half_cylindrical_panorama_to_perspective(pano,
                                             out_w=3200, out_h=1800, crop_bottom=1600,
                                             out_fov_x_deg=90.0,
                                             yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    """Convert half cylindrical panorama to perspective view."""
    H_eq, W_eq = pano.shape[:2]
    horiz_span = math.radians(180.0)
    vert_span = horiz_span * (H_eq / float(W_eq))

    fx = (out_w / 2.0) / math.tan(math.radians(out_fov_x_deg) / 2.0)
    fy = fx
    cx = out_w / 2.0
    cy = out_h / 2.0

    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    R_y = np.array([[math.cos(yaw), 0, math.sin(yaw)],
                    [0, 1, 0],
                    [-math.sin(yaw), 0, math.cos(yaw)]])
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(pitch), -math.sin(pitch)],
                    [0, math.sin(pitch), math.cos(pitch)]])
    R_z = np.array([[math.cos(roll), -math.sin(roll), 0],
                    [math.sin(roll), math.cos(roll), 0],
                    [0, 0, 1]])

    R = R_y @ R_x @ R_z

    u = np.arange(out_w)
    v = np.arange(out_h)
    uu, vv = np.meshgrid(u, v)

    x_cam = (uu - cx) / fx
    y_cam = (vv - cy) / fy
    z_cam = np.ones_like(x_cam)

    norm = np.sqrt(x_cam**2 + y_cam**2 + z_cam**2)
    x_cam /= norm
    y_cam /= norm
    z_cam /= norm

    dirs = np.stack([x_cam, y_cam, z_cam], axis=-1).reshape(-1, 3).T
    dirs_world = (R @ dirs).T

    dx, dy, dz = dirs_world[:, 0], dirs_world[:, 1], dirs_world[:, 2]

    lam = np.arctan2(dx, dz)
    phi = np.arctan2(dy, np.sqrt(dx**2 + dz**2))

    u_src = (lam + horiz_span / 2.0) / horiz_span * (W_eq - 1.0)
    v_src = (vert_span / 2.0 - phi) / vert_span * (H_eq - 1.0)

    # u_src = np.mod(u_src, W_eq)
    # v_src = np.clip(v_src, 0, H_eq - 1)
    valid_u_mask = (u_src >= 0) & (u_src <= W_eq - 1)
    u_src[~valid_u_mask] = -1  # Will trigger black in remap
    valid_v_mask = (v_src >= 0) & (v_src <= H_eq - 1)
    v_src[~valid_v_mask] = -1  # Invalid value to trigger black border


    map_x = u_src.reshape(out_h, out_w).astype(np.float32)
    map_y = v_src.reshape(out_h, out_w).astype(np.float32)

    persp = cv2.remap(pano, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)

    crop_w = crop_bottom * out_w / out_h
    border_left = int((out_w - crop_w) / 2)
    border_right = int(border_left + out_w)

    return cv2.flip(persp[:crop_bottom, border_left:border_right], 0)


def load_focus_coordinates(txt_path, fps):
    """Load timestamped focus coordinates and convert times to frame indices."""

    if not os.path.exists(txt_path):
        print(f"Error: BallPos file '{txt_path}' not found.")
        sys.exit(1)

    times, xs, ys = [], [], []
    with open(txt_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            h, m, s = parts[0].split(':')
            sec = int(h) * 3600 + int(m) * 60 + float(s)
            times.append(sec)
            xs.append(float(parts[1]))
            ys.append(float(parts[2]))
    frames = [t * fps for t in times]
    return np.array(frames), np.array(xs), np.array(ys)


def track_ball_video(input_video="input.mp4", output_video="output.mp4", out_w=2560, out_h=1440, crop_h=1600, out_fov_x_deg=80, pitch_deg=0, roll_max_deg=20, gamma=1.0):
    """
    Process input panorama video and output perspective-tracked video.
    """
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise IOError(f"Cannot open input video: {input_video}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, frame = cap.read()
    if not ret:
        raise IOError("Cannot read first frame from input video.")

    H_eq, W_eq = frame.shape[:2]
    horiz_span_deg = 180.0 
    yaw_max = (horiz_span_deg - out_fov_x_deg) / 2

    # ---- VideoWriter setup ----
    # fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
    # writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, crop_h))

    output_container = av.open(output_video, mode='w')
    output_stream = output_container.add_stream('libx265', rate=int(round(fps)))
    output_stream.width = out_w
    output_stream.height = out_h
    output_stream.pix_fmt = 'yuv420p'
    output_stream.options = {'preset': 'fast', 'crf': '23'}

    # Reload first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    start_time = datetime.now()

    
    video_path = Path(input_video)
    ballpos_path = video_path.with_name(f"{video_path.stem}_coordinates.txt")
    frame_coords, xs, ys = load_focus_coordinates(ballpos_path, fps)
    yaws = (xs / W_eq - 0.5) * horiz_span_deg
    frame_indices = np.arange(frame_count)
    yaw_targets = np.interp(frame_indices, frame_coords, yaws)


    # ---- Physics-Based Tracking Setup ----
    smoothed_yaw = 0.0
    velocity_yaw = 0.0

    max_yaw_speed = 0.4       # deg/frame
    max_yaw_accel = 0.03      # deg/frame²
    dead_zone_threshold = 20.0  # deg
    prediction_horizon = 50    # frames

    past_yaws = []
    current_pitch = pitch_deg

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    
    lookUpTable = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                            for i in np.arange(0, 256)]).astype("uint8")
    
    for i in range(frame_count):
        ret, pano = cap.read()
        if not ret or pano is None:
            break

        # --- Get current ball yaw ---
        target_yaw = yaw_targets[i]

        # --- Update history for prediction ---
        past_yaws.append(target_yaw)
        if len(past_yaws) > prediction_horizon:
            past_yaws.pop(0)

        # --- Predict future yaw ---
        if len(past_yaws) >= 2:
            dyaw = past_yaws[-1] - past_yaws[0]
            dt = len(past_yaws)
            predicted_velocity = dyaw / dt
            predicted_yaw = target_yaw + predicted_velocity * prediction_horizon
        else:
            predicted_yaw = target_yaw

        # --- Compute desired yaw velocity ---
        yaw_error = predicted_yaw - smoothed_yaw

        if abs(yaw_error) < dead_zone_threshold:
            desired_velocity = 0.0
        else:
            desired_velocity = np.clip(yaw_error, -max_yaw_speed, max_yaw_speed)

        # --- Apply acceleration constraint ---
        velocity_change = desired_velocity - velocity_yaw
        velocity_change = np.clip(velocity_change, -max_yaw_accel, max_yaw_accel)
        velocity_yaw += velocity_change

        # --- Update smoothed camera yaw ---
        smoothed_yaw += velocity_yaw
        current_yaw = np.clip(smoothed_yaw, -yaw_max, yaw_max)
        # current_yaw = yaw_targets[i]

        # --- Optional: Apply roll proportional to yaw ---
        roll_deg = np.interp(current_yaw, [-yaw_max, 0, yaw_max], [roll_max_deg, 0, -roll_max_deg])

        persp = half_cylindrical_panorama_to_perspective(
            pano,
            out_w=3200,
            out_h=1800,
            crop_bottom=crop_h,
            out_fov_x_deg=out_fov_x_deg,
            yaw_deg=current_yaw,
            pitch_deg=current_pitch,
            roll_deg=roll_deg
        ) 

        gamma_corrected = cv2.LUT(persp, lookUpTable)

        #writer.write(persp)
        video_frame = av.VideoFrame.from_ndarray(gamma_corrected, format='bgr24')
        for packet in output_stream.encode(video_frame):
            output_container.mux(packet)

        print(f"Frame {i+1}/{frame_count} | yaw={current_yaw:.1f}°", end='\r', flush=True)

        if i % 100 == 0:
            gc.collect()

    # 🔚 Final encoder flush
    for packet in output_stream.encode():
        output_container.mux(packet)
    output_container.close()
    cap.release()
    #writer.release()
    print(f"\n✅ Output saved: {output_video} | Time elapsed: {datetime.now() - start_time}")
 
def main():
    if len(sys.argv) < 3:
        print("Usage: python main.py <input_video> <output_video>")
        sys.exit(1)

    input_video = sys.argv[1]
    output_video = sys.argv[2]

    if not os.path.exists(input_video):
        print(f"Error: Input Video' {input_video}' not found.")
        sys.exit(1)

    video_path = Path(input_video)
    config_path = video_path.with_name(f"{video_path.stem}_config.txt")
    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' not found. Please run 'python calib.py'")
        sys.exit(1)
    
    config = {}
    with open(config_path, 'r') as f:
        for line in f:
            if not line or '=' not in line or line.startswith('#'):
                continue
            key, value = line.strip().split('=', 1)
            key = key.strip()
            value = value.strip()
            try:
                value = float(value)
            except ValueError:
                pass
            config[key] = value

    crop_h = int(config.get('crop_h', 1440))
    out_fov_x_deg = config.get('fov', 90)
    pitch_deg = config.get('pitch', -10)
    roll_max_deg = config.get('roll_max', 30)
    gamma = config.get('gamma', 1.0)

    track_ball_video(input_video, output_video, out_w=2560, out_h=1440, crop_h=crop_h, out_fov_x_deg=out_fov_x_deg, pitch_deg=pitch_deg, roll_max_deg=roll_max_deg, gamma=gamma)

if __name__ == "__main__":
    main()