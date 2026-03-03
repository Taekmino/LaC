#!/bin/bash
set -e

CATKIN_WS=/home/appuser/LaC/catkin_ws

# Source ROS setup
source /opt/ros/noetic/setup.bash

# Auto-build on first run: devel/ is not mounted, so it's absent on fresh containers.
if [ ! -f "${CATKIN_WS}/devel/setup.bash" ] && [ -d "${CATKIN_WS}/src" ]; then
    echo "[entrypoint] First run: building catkin workspace..."
    cd "$CATKIN_WS" && catkin_make
    echo "[entrypoint] Build complete."

    echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc
    echo "source ${CATKIN_WS}/devel/setup.bash" >> /root/.bashrc
fi

# Compile GroundingDINO CUDA ops on first run.
# docker build has no GPU driver, so build_ext must run inside the live container.
GDINO_SO=$(find /home/appuser/LaC/GroundingDINO/groundingdino -name '_C*.so' 2>/dev/null | head -1)
if [ -z "$GDINO_SO" ]; then
    echo "[entrypoint] Compiling GroundingDINO CUDA ops (first run)..."
    cd /home/appuser/LaC/GroundingDINO && python3 setup.py build_ext --inplace
    echo "[entrypoint] GroundingDINO CUDA ops compiled."
fi

# Execute the command
exec "$@"
