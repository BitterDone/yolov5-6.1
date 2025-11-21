#!/bin/bash

# === CONFIGURATION ===
# Replace these with your actual VM details
REMOTE_USER="root"
REMOTE_HOST="167.99.182.230"  # e.g., 137.184.XX.XX
REMOTE_PATH="./runs/train/railroad_detector7"
REMOTE_DATA_YAML="./railroad-cars/data.yaml"
LOCAL_DEST="./railroad_model_backup"

# === Create local folder if needed ===
mkdir -p "$LOCAL_DEST"

# === Copy files ===
echo "📥 Downloading best.pt"
scp -i C:\r\me\monitoring-trains\monitoring-trains-key-openssh "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/weights/best.pt" "$LOCAL_DEST/"

echo "📥 Downloading results.png"
scp -i C:\r\me\monitoring-trains\monitoring-trains-key-openssh "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/results.png" "$LOCAL_DEST/"

echo "📥 Downloading opt.yaml"
scp -i C:\r\me\monitoring-trains\monitoring-trains-key-openssh "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/opt.yaml" "$LOCAL_DEST/"

echo "📥 Downloading hyp.yaml"
scp -i C:\r\me\monitoring-trains\monitoring-trains-key-openssh "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/hyp.yaml" "$LOCAL_DEST/"

echo "📥 Downloading train_batch*.jpg"
scp -i C:\r\me\monitoring-trains\monitoring-trains-key-openssh "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/train_batch*.jpg" "$LOCAL_DEST/"

echo "📥 Downloading data.yaml"
scp -i C:\r\me\monitoring-trains\monitoring-trains-key-openssh "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DATA_YAML" "$LOCAL_DEST/data.yaml"

echo "✅ All files downloaded to: $LOCAL_DEST"
