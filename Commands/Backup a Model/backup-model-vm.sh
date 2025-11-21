#!/bin/bash

# === Paths ===
EXP_DIR=" /root/monitoring-trains/yolov5-6.1/runs/train/railroad_detector7"
DATA_YAML="/root/monitoring-trains/yolov5-6.1/railroad-cars/data.yaml"
OUTPUT_TAR="model_export.tar.gz"

# === Go to a safe working directory ===
cd /root

# === Remove any previous archive ===
rm -f "$OUTPUT_TAR"

# === Create the archive ===
tar -czf "$OUTPUT_TAR" \
  -C "$EXP_DIR/weights" best.pt \
  -C "$EXP_DIR" results.png opt.yaml hyp.yaml \
  -C "$EXP_DIR" $(ls "$EXP_DIR"/train_batch*.jpg 2>/dev/null) \
  -C "$(dirname "$DATA_YAML")" "$(basename "$DATA_YAML")"

echo "✅ Export complete: $OUTPUT_TAR"

