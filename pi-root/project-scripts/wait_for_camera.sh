#!/bin/bash
set -euo pipefail

# sudo chmod +x wait-for-camera.sh

# ✅ View logs from your detect.service
# -f = follow live (like tail -f)
#   sudo journalctl -u detect.service -f

# This will show all stdout/stderr produced by waitForCamera.sh, including:
#   Camera detected!
# or
#   Camera not found after 30 seconds. Starting service anyway...

# 🔍 View older logs
# To see the full history:
#   sudo journalctl -u detect.service

# 💡 If you want timestamps
#   sudo journalctl -u detect.service --no-pager -o short-precise

CAMERA_IP="192.168.0.100"   # <-- set your camera static IP
TIMEOUT=60                 # max seconds to wait

echo "Waiting for camera at $CAMERA_IP..."

for i in $(seq 1 $TIMEOUT); do
    if ping -c 1 -W 1 "$CAMERA_IP" &> /dev/null; then
        echo "Camera detected!"
        exit 0
    fi
    sleep 1
done

echo "Camera not found after $TIMEOUT seconds. Starting service anyway..."
exit 0
