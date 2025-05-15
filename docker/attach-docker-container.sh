#!/bin/bash
# Container management script for Isaac Lab MC-RTC

# Configuration
IMAGE_NAME="mirrors.tencent.com/rl-for-control/isaac-lab-mc_rtc"
IMAGE_TAG="latest"
CONTAINER_NAME="isaac-lab-mc_rtc-yuquan"


# 强制清理同名旧容器
if docker ps -a | grep -q "$CONTAINER_NAME"; then
    echo "发现已存在的容器 $CONTAINER_NAME，正在清理..."
    docker rm -f "$CONTAINER_NAME"
fi


# Check image availability
if ! docker image inspect "${IMAGE_NAME}:${IMAGE_TAG}" >/dev/null 2>&1; then
    echo "Error: Required image ${IMAGE_NAME}:${IMAGE_TAG} not found!"
    exit 1
fi

# Find running container
EXISTING_CONTAINER=$(docker ps -q --filter "ancestor=${IMAGE_NAME}:${IMAGE_TAG}")

if [ -n "${EXISTING_CONTAINER}" ]; then
    echo "Attaching to existing container (ID: ${EXISTING_CONTAINER})"
    docker attach "${EXISTING_CONTAINER}"
else
    echo "Starting new container..."
    ./run-isaac-lab-docker.sh

    echo "Attaching to newly created container..."
    docker attach "${CONTAINER_NAME}"
fi
