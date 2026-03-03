FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies + ROS Noetic
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    lsb-release \
    gnupg \
    build-essential \
    cmake \
    git \
    wget \
    ninja-build \
    python3-pip \
    python3-dev \
    python3-tk \
    libopencv-dev \
    libpcl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-base \
    # ros-noetic-rviz \
    ros-noetic-cv-bridge \
    ros-noetic-image-transport \
    ros-noetic-message-filters \
    ros-noetic-pcl-ros \
    ros-noetic-pcl-conversions \
    ros-noetic-tf \
    ros-noetic-nav-msgs \
    ros-noetic-sensor-msgs \
    python3-catkin-tools \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    langchain-openai \
    langchain-core \
    python-dotenv \
    opencv-python-headless \
    "numpy>=1.20"

# Install torch with CUDA 11.8
RUN pip3 install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Set up workspace
WORKDIR /home/appuser/LaC
ENV CATKIN_WS=/home/appuser/LaC/catkin_ws

# ---------------------------------------------------------------
# Install GroundingDINO
# ---------------------------------------------------------------
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git && \
    cd GroundingDINO && \
    pip3 install --no-cache-dir -e . && \
    mkdir -p weights && \
    wget -q -P weights https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# ---------------------------------------------------------------
# Install EdgeSAM
# ---------------------------------------------------------------
RUN pip3 install --no-cache-dir segment-anything

# Install OpenMMLab dependencies (mmcv pre-built wheel for cu118 + torch2.0)
RUN pip3 install --no-cache-dir \
    mmengine==0.10.3 \
    mmdet \
    pycocotools==2.0.6 \
    timm==0.4.12 \
    yacs==0.1.8 \
    loralib==0.1.2 \
    kornia==0.7.1
RUN pip3 install --no-cache-dir \
    mmcv==2.1.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

RUN git clone https://github.com/chongzhou96/EdgeSAM.git EfficientSAM && \
    cd EfficientSAM && \
    pip3 install --no-cache-dir -e . && \
    mkdir -p weights && \
    wget -q -P weights https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam_3x.pth

RUN SAM_PY=/home/appuser/LaC/EfficientSAM/edge_sam/modeling/sam.py && \
    sed -i '/^from projects\.EfficientDet import efficientdet$/d' "$SAM_PY" && \
    sed -i "s/elif rpn_head == 'efficient_det':/elif rpn_head == 'efficient_det':\n            from projects.EfficientDet import efficientdet/" "$SAM_PY"

RUN sed -i "s/    build_sam,$/    build_sam,\n    build_edge_sam,/" \
        /home/appuser/LaC/EfficientSAM/edge_sam/__init__.py

RUN echo '/usr/local/lib/python3.8/dist-packages/torch/lib' > /etc/ld.so.conf.d/torch.conf && ldconfig

# Set environment variables for model paths
ENV GROUNDING_DINO_CONFIG_PATH=/home/appuser/LaC/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
ENV GROUNDING_DINO_CHECKPOINT_PATH=/home/appuser/LaC/GroundingDINO/weights/groundingdino_swint_ogc.pth
ENV EDGE_SAM_CHECKPOINT_PATH=/home/appuser/LaC/EfficientSAM/weights/edge_sam_3x.pth
ENV EDGE_SAM_DIR=/home/appuser/LaC/EfficientSAM
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/torch/lib:${LD_LIBRARY_PATH}
ENV PYTHONPATH="${PYTHONPATH}:/home/appuser/LaC/EfficientSAM"

# Build the catkin workspace
SHELL ["/bin/bash", "-c"]

# Compile GroundingDINO CUDA ops on first run (requires GPU driver, cannot be done at build time).
COPY ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

RUN apt-get update && apt-get install -y --no-install-recommends ros-noetic-rviz && \
    rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash", "-l"]
