#!/usr/bin/env python3
from flask import Flask, Response
import cv2
import time
import numpy as np
import onnxruntime as ort
import json

app = Flask(__name__)

# ---- Your existing functions ----
# preprocess(), parse_output(), session, input_name, etc.

class_names = ['autorack', 'boxcar', 'cargo', 'container', 'flatcar',
               'flatcar_bulkhead', 'gondola', 'hopper', 'locomotive',
               'passenger', 'tank']

IMG_SIZE = 640  # Should match your model

cap = None

config = json.load(open("/home/danbitter/yolov5-6.1/pi-root/project-scripts/config.json"))
RTSP_URL = config["rtsp_url"]
MODEL_PATH = "/home/danbitter/yolov5-6.1/pi-root/models/best.onnx"
providers = ["CPUExecutionProvider"]

print(f"Loading model: {MODEL_PATH}")
session = ort.InferenceSession(MODEL_PATH, providers=providers)
input_name = session.get_inputs()[0].name

def draw_overlay(frame, conf, class_ids):
    y = 20
    if len(conf) == 0:
        cv2.putText(frame, "No detections", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        return frame

    for c, cls in zip(conf, class_ids):
        text = f"{class_names[cls]}: {c:.2f}"
        cv2.putText(frame, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        y += 25
    return frame

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

def decode_yolo_output(pred, conf_thres=0.5):
    """
    pred shape: (N, 16)
    Layout per row:
      [x, y, w, h, conf_class0, conf_class1, ... conf_class10]
    """
    boxes = pred[:, 0:4]
    scores = pred[:, 4:]

    class_ids = np.argmax(scores, axis=1)
    conf = scores[np.arange(len(scores)), class_ids]

    # Filter by confidence threshold
    mask = conf > conf_thres

    return boxes[mask], conf[mask], class_ids[mask]

# -----------------------
# Drawing
# -----------------------
def draw_boxes(frame, boxes, conf, class_ids):
    h, w = frame.shape[:2]

    for (x, y, bw, bh), c, cls in zip(boxes, conf, class_ids):
        # YOLO normally outputs xywh normalized between 0–1
        x1 = int((x - bw/2) * w)
        y1 = int((y - bh/2) * h)
        x2 = int((x + bw/2) * w)
        y2 = int((y + bh/2) * h)

        # clamp
        x1 = max(0, min(w-1, x1))
        y1 = max(0, min(h-1, y1))
        x2 = max(0, min(w-1, x2))
        y2 = max(0, min(h-1, y2))

        label = f"{class_names[cls]} {c:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, max(0,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    return frame


# -----------------------
# MJPEG generator
# -----------------------
def generate_stream():
    global cap
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        # Inference
        img = preprocess(frame)
        outputs = session.run(None, {input_name: img})

        # Use your parse_output if preferred, or decode here
        pred = outputs[0].reshape(-1, 16)  # YOLOv5 ONNX output
        boxes, conf, ids = decode_yolo_output(pred)

        frame = draw_boxes(frame, boxes, conf, ids)

        ret, jpeg = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               jpeg.tobytes() + b"\r\n")

def generate_stream():
    global cap
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        # Inference
        img = preprocess(frame)
        outputs = session.run(None, {input_name: img})

        # Use your parse_output if preferred, or decode here
        pred = outputs[0].reshape(-1, 16)  # YOLOv5 ONNX output
        boxes, conf, ids = decode_yolo_output(pred)

        frame = draw_boxes(frame, boxes, conf, ids)

        ret, jpeg = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               jpeg.tobytes() + b"\r\n")

def connect_camera():
    while True:
        cap = cv2.VideoCapture(RTSP_URL)
        if cap.isOpened():
            print("Camera ready!")
            return cap
        print("Camera not ready, retrying in 5 sec...")
        time.sleep(5)

@app.route("/debug")
def debug_feed():
    return Response(generate_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    print("Connecting to camera...")
    cap = connect_camera()
    if not cap or not cap.isOpened():
        raise RuntimeError("Failed to open camera.")
    print("Open http://192.168.0.1:5000/debug to view stream")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
