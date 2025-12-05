import time
import json
from collections import deque
import cv2
import requests
import numpy as np
import onnxruntime as ort
from pathlib import Path

# --------------------------------------------------------
# Load config
# --------------------------------------------------------
config = json.load(open("config.json"))
RTSP_URL = config["rtsp_url"]
TARGET_CLASS = config["target_class"]
API_URL = config["api_url"]
WINDOW_SECONDS = config["threshold_seconds"]
THRESHOLD = config["threshold_count"]
DISPLAY_DEBUG = config["display_debug"]

# --------------------------------------------------------
# Load ONNX model
# --------------------------------------------------------
MODEL_PATH = "best.onnx"
providers = ["CPUExecutionProvider"]

print(f"Loading model: {MODEL_PATH}")
session = ort.InferenceSession(MODEL_PATH, providers=providers)
input_name = session.get_inputs()[0].name

# YOLO parameters (student model is small)
IMG_SIZE = 640

# --------------------------------------------------------
# Helper: preprocess frame
# --------------------------------------------------------
def preprocess(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img[:, :, ::-1]  # BGR → RGB
    img = img.transpose(2, 0, 1)  # HWC → CHW
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0
    return img[None]

# --------------------------------------------------------
# Helper: parse YOLO output
# --------------------------------------------------------
def parse_output(pred):
    pred = pred[0]  # first batch
    conf = pred[:, 4]
    class_ids = pred[:, 5]
    return conf, class_ids

# --------------------------------------------------------
# Rolling window setup
# --------------------------------------------------------
detections_window = deque()

# --------------------------------------------------------
# Open RTSP stream
# Enhace connect_camera to notify on 30 minutes of downtime
# Let systemd handle restarts?
# --------------------------------------------------------
def connect_camera():
    while True:
        cap = cv2.VideoCapture(RTSP_URL)
        if cap.isOpened():
            return cap
        print("Camera not ready, retrying in 5 sec...")
        time.sleep(5)

cap = connect_camera()
if not cap.isOpened():
    raise RuntimeError("Failed to connect to RTSP stream. Check URL and camera.")

print("Streaming started. Running detection...")

# --------------------------------------------------------
# Main loop
# Enhance this to self-heal if the camera dies
# --------------------------------------------------------
while True:
    ok, frame = cap.read()
    if not ok:
        continue

    img = preprocess(frame)
    outputs = session.run(None, {input_name: img})

    # Parse outputs
    conf, class_ids = parse_output(outputs)

    now = time.time()

    # Filter detections for the target class
    for c, class_id in zip(conf, class_ids):
        if (c > 0.5).any() and (class_id == 0).any():  # adjust class index if needed
            detections_window.append(now)

    # Remove expired detections
    while detections_window and detections_window[0] < now - WINDOW_SECONDS:
        detections_window.popleft()

    # Trigger API
    if len(detections_window) >= THRESHOLD:
        print(f"API Trigger: {len(detections_window)} detections in window")
        try:
            requests.post(API_URL, json={"event": TARGET_CLASS, "count": len(detections_window)})
        except Exception as e:
            print("API Error:", e)
        detections_window.clear()

    # Optional debug display
    if DISPLAY_DEBUG:
        cv2.imshow("Edge AI Stream", frame)
        if cv2.waitKey(1) == ord('q'):
            break
