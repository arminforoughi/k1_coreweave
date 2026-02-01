#!/bin/bash
# Launch the Jetson client on the Booster K1.
#
# The K1's ROS 2 stack uses a custom FastDDS profile that restricts data
# transport to localhost UDP + shared memory.  Without loading this profile
# our subscriber can *discover* topics but never receives any frames.

set -eo pipefail

export FASTRTPS_DEFAULT_PROFILES_FILE=/opt/booster/BoosterRos2/fastdds_profile.xml
source /opt/ros/humble/setup.bash

# Detect MacBook backend IP — override with BACKEND_URL env var if needed
BACKEND_URL="${BACKEND_URL:-http://10.20.15.165:8003}"

python3 perception/jetson_client.py \
  --backend "$BACKEND_URL" \
  --topic /booster_camera_bridge/image_left_raw \
  --no-depth \
  --fps 2.0 \
  --confidence 0.25
