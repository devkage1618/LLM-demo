import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import trim_mean

def pitch_orientation(frame, n):

    """Estimate pitch and roll orientation based on green hue in panorama."""
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
           (max(avg_per_row1[idx], avg_per_row2[idx]) - avg_first > 40):
            topborder_soccerfield = y_end - idx - 10 
            break

    combined = np.hstack((crop1, crop2))
    start_point=(0, topborder_soccerfield)
    end_point=(combined.shape[1], topborder_soccerfield)
    cv2.line(combined, start_point, end_point, color=(255,255,255), thickness=2)
    cv2.imwrite(f"image/hk_input{n}.png",combined)

    start_point=(0, topborder_soccerfield)
    end_point=(width, topborder_soccerfield)
    cv2.line(frame, start_point, end_point, color=(255,255,255), thickness=2)
    cv2.imwrite(f"image/ht_input{n}.png",frame)

for n in range(1,8):
    frame = cv2.imread(f"image/clip{n}.png")  # Replace with your image path
    pitch_orientation(frame, n)

frame = cv2.imread(f"image/input.png")  # Replace with your image path
pitch_orientation(frame, 8)
 
