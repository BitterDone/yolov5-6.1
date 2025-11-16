#!/bin/bash

# Make executable:
# chmod +x stream_to_twitch.sh

STREAM_KEY="YOUR_TWITCH_STREAM_KEY"
RTSP_URL="rtsp://user:pass@CAMERA_IP:554/h264Preview_01_main"

ffmpeg -rtsp_transport tcp -i "$RTSP_URL" \
  -vcodec copy -acodec aac \
  -f flv "rtmp://live.twitch.tv/app/$STREAM_KEY"
