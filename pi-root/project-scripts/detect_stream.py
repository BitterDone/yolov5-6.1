import time
import json
from collections import deque
import cv2
import requests
import numpy as np
import onnxruntime as ort
from pathlib import Path
import os

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
    # resize to a square image, possibly distorting aspect ratio
    # For a first test, plain resize is fine, but for production, consider a letterbox function.
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img[:, :, ::-1]  # BGR → RGB
    img = img.transpose(2, 0, 1)  # HWC → CHW
    # Ensures the array is stored contiguously in memory, required by some libraries like ONNX Runtime.
    img = np.ascontiguousarray(img, dtype=np.float32)
    # Normalizes pixel values from [0, 255] → [0.0, 1.0].
    # Many models are trained on float images in the 0-1 range.
    img /= 255.0

    # Adds a batch dimension at the front.
    # ONNX Runtime expects input shape (batch_size, channels, height, width).
    # By returning img[None], you get shape (1, C, H, W).
    return img[None]

# --------------------------------------------------------
# Helper: parse YOLO output, but was returning bounding boxes?
# [[3.5810940e+01 7.3996449e+00 3.3653713e+01 1.5182625e+01 4.1127205e-06 1.9042850e-02 2.1468759e-02 1.4087200e-02 1.4146769e-01 2.6069313e-02 9.8165870e-03 7.7925146e-02 4.8476279e-02 3.3198088e-02 5.9680092e-01
# 5.9449852e-02]] 
# [[4.4976143e+01 7.4493561e+00 3.3781521e+01 1.5126505e+01 1.8477440e-06 2.0089865e-02 2.1678686e-02 1.5307128e-02 1.9232219e-01 2.5725007e-02 1.0918647e-02 7.5550646e-02 5.0171554e-02 3.6671728e-02 5.5309051e-01
# 6.0226411e-02]]
# --------------------------------------------------------
# def parse_output(pred):
#     pred = pred[0]  # first batch
#     conf = pred[:, 4]
#     class_ids = pred[:, 5]
#     return conf, class_ids

def parse_output(outputs, conf_thres=0.5):
    # Suppose the model returns one output, shape (1, N, 6)
    preds = outputs[0][0]  # drop batch dimension

    # confidence = index 4; class = index 5
    confs = preds[:, 4]
    class_ids = preds[:, 5].astype(int)

    # optional: filter out low-confidence
    mask = confs >= conf_thres
    return confs[mask], class_ids[mask]


print(f"Rolling window setup", flush=True)
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

print(f"Accessing camera", flush=True)
cap = connect_camera()
if not cap.isOpened():
    raise RuntimeError("Failed to connect to RTSP stream. Check URL and camera.")

ok, frame = cap.read()
print("Frame read:", ok, flush=True)

print("Streaming started. Running detection...", flush=True)


print("Begin debugging script", flush=True)
print("cwd =", os.getcwd(), flush=True)
frame = cv2.imread("/home/danbitter/yolov5-6.1/pi-root/project-scripts/freight_train.jpg")
print("frame is None?", frame is None, flush=True)

img = preprocess(frame)
# input_name must match your model’s input tensor name. You can print it using:
print("input_name to match model input tensor name", session.get_inputs()[0].name, flush=True)
outputs = session.run(None, {input_name: img})

conf, class_ids = parse_output(outputs)
print(conf, class_ids, flush=True)


# # --------------------------------------------------------
# # Main loop
# # Enhance this to self-heal if the camera dies
# # --------------------------------------------------------
# counterNotOk = 0
# counterNotOkThreshold = 10
# while True:
#     ok, frame = cap.read()
#     if not ok:
#         counterNotOk += 1

#         if counterNotOk > counterNotOkThreshold:
#             print(f"Last {counterNotOkThreshold} frames were not ok", flush=True)
#             counterNotOk = 0

#         continue

#     img = preprocess(frame)
#     outputs = session.run(None, {input_name: img})

#     # Parse outputs
#     conf, class_ids = parse_output(outputs)

#     now = time.time()
#     detected_this_frame = False

#     # Filter detections for the target class
#     for c, class_id in zip(conf, class_ids):
#         if (c > 0.5).any() and (class_id == TARGET_CLASS).any():  # adjust class index if needed
#             print("Train detected!", len(detections_window), flush=True)
#             detections_window.append(now)
#             detected_this_frame = True

#     # Remove expired detections
#     while detections_window and detections_window[0] < now - WINDOW_SECONDS:
#         detections_window.popleft()

#     # Trigger API
#     if len(detections_window) >= THRESHOLD:
#         print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - API Trigger: {len(detections_window)} detections in window", flush=True)
#         try:
#             requests.post(API_URL, json={"event": TARGET_CLASS, "count": len(detections_window)})
#         except Exception as e:
#             print("API Error:", e, flush=True)
#         detections_window.clear()

#     # Print "no trains detected" message if 30s passed with no detection, flush=True
#     if not detected_this_frame and (now - last_no_detection_msg) >= WINDOW_SECONDS:
#         print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - No trains detected in the last {WINDOW_SECONDS} seconds", flush=True)
#         last_no_detection_msg = now

#     # Optional debug display
#     if DISPLAY_DEBUG:
#         cv2.imshow("Edge AI Stream", frame)
#         if cv2.waitKey(1) == ord('q'):
#             break
