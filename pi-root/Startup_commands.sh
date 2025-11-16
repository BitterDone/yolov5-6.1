sudo cp services/*.service /etc/systemd/system/

sudo systemctl daemon-reload

sudo systemctl enable detect.service
sudo systemctl enable twitch.service

sudo systemctl start detect.service
sudo systemctl start twitch.service

# Want optional upgrades?

# I can provide:

# ✅ MQTT-based alerts
# ✅ Local video recording on detections
# ✅ GPU acceleration for Jetson Nano/Orin
# ✅ Docker container version
# ✅ Watchdog to restart camera stream
# ✅ Remote update mechanism (git pull + reload services)
