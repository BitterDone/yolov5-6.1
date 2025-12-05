# Run from ~/yolov5-6.1/ directory
# 1. Place the service files
sudo rm /etc/systemd/system/detect.service
sudo cp pi-root/services/detect.service /etc/systemd/system/

sudo rm /usr/local/bin/wait_for_camera.sh
sudo cp pi-root/project-scripts/wait_for_camera.sh /usr/local/bin/
sudo chmod +x /usr/local/bin/wait_for_camera.sh

sudo mkdir -p /home/danbitter/detect/

sudo rm /home/danbitter/detect/detect_stream.py
sudo cp pi-root/project-scripts/detect_stream.py /home/danbitter/detect/
sudo chmod +x /home/danbitter/detect/detect_stream.py

sudo rm /home/danbitter/detect/config.json
sudo cp pi-root/project-scripts/config.json /home/danbitter/detect/
# Receives -rw-r--r-- by default
# sudo chmod +x /home/danbitter/detect/config.json

sudo rm /home/danbitter/detect/best.onnx
sudo cp pi-root/models/best.onnx /home/danbitter/detect/best.onnx
# Receives -rw-r--r-- by default
# sudo chmod +x /home/danbitter/detect/best.onnx
