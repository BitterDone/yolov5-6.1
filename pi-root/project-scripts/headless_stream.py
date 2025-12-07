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

cap = None

config = json.load(open("config.json"))
RTSP_URL = config["rtsp_url"]
MODEL_PATH = "best.onnx"
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

def parse_output(outputs, conf_thres=0.5):
    # Rewrite parse_output() for your YOLOv8 ONNX model. Based on your output shapes (1, 25200, 16)
    # Suppose the model returns one output, shape (1, N, 6)
    # Use only the first output
    preds = outputs[0]  # shape: (1, 25200, 16)
    preds = preds[0]     # remove batch dim → (25200, 16)

    # Extract objectness and class probabilities
    confs = preds[:, 4]  # objectness score
    class_probs = preds[:, 5:]  # class probabilities
    class_ids = np.argmax(class_probs, axis=1)
    class_conf = np.max(class_probs, axis=1)

    # Combined confidence
    conf = confs * class_conf

    # Filter by confidence threshold
    mask = conf > conf_thres
    
    return conf[mask], class_ids[mask]

    # Previous parse_output logic for reference:
    # # confidence = index 4; class = index 5
    # confs = preds[:, 4]
    # class_ids = preds[:, 5].astype(int)

    # # optional: filter out low-confidence
    # mask = confs >= conf_thres
    # return confs[mask], class_ids[mask]

def generate_stream():
    global cap
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        img = preprocess(frame)
        outputs = session.run(None, {input_name: img})
        conf, class_ids = parse_output(outputs)

        frame = draw_overlay(frame, conf, class_ids)

        # Encode frame as JPEG
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
    print("Open http://192.168.0.1:5000/debug to view stream")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
