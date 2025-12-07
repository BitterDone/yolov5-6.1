import time
import json
from collections import deque
import cv2
import requests
import numpy as np
import onnxruntime as ort
from pathlib import Path

print(f"Loading config")
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
last_no_detection_msg = 0  # track last time we printed "no trains detected"

print(f"Loading ONNX model")
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

print(f"Rolling window setup")
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
            print("Camera ready!")
            return cap
        print("Camera not ready, retrying in 5 sec...")
        time.sleep(5)

print(f"Accessing camera")
cap = connect_camera()
if not cap.isOpened():
    raise RuntimeError("Failed to connect to RTSP stream. Check URL and camera.")

ok, frame = cap.read()
print("Frame read:", ok, flush=True)

print("Begin debugging script", flush=True)
img = cv2.imread("freight_train.jpg")
img_input = preprocess(img)
outputs = session.run(None, {input_name: img_input})

conf, class_ids = parse_output(outputs)
print(conf, class_ids)
