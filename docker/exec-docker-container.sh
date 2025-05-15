#!/bin/bash
# Container management script for Isaac Lab MC-RTC

# Configuration
IMAGE_NAME="mirrors.tencent.com/rl-for-control/isaac-lab-mc_rtc"
IMAGE_TAG="latest"
CONTAINER_NAME="isaac-lab-mc_rtc-yuquan"

# Check image availability
if ! docker image inspect "${IMAGE_NAME}:${IMAGE_TAG}" >/dev/null 2>&1; then
    echo "Error: Required image ${IMAGE_NAME}:${IMAGE_TAG} not found!"
    exit 1
fi

# Find running container
EXISTING_CONTAINER=$(docker ps -q --filter "name=${CONTAINER_NAME}")

if [ -n "${EXISTING_CONTAINER}" ]; then
    echo "Entering an existing container (ID: ${EXISTING_CONTAINER})"
    docker exec -it "${CONTAINER_NAME}" /bin/bash
else
    echo "Error: Container ${CONTAINER_NAME} is not running!"
    exit 1
fi
