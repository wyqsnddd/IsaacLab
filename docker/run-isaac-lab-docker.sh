#!/bin/bash
# run-container.sh

# 设置变量
IMAGE_NAME="mirrors.tencent.com/rl-for-control/isaac-lab-mc_rtc"
IMAGE_TAG="latest"  # 或使用时间戳标签
CONTAINER_NAME="isaac-lab-mc_rtc-yuquan"

# 检查镜像是否存在
if ! docker image inspect ${IMAGE_NAME}:${IMAGE_TAG} >/dev/null 2>&1; then
    echo "Error: Image ${IMAGE_NAME}:${IMAGE_TAG} not found!"
    exit 1
fi

# 运行容器
sudo docker run --name ${CONTAINER_NAME} \
    --entrypoint bash \
    -it -d \
    --ipc=host \
    --gpus all \
    -e "ACCEPT_EULA=Y" \
    --rm \
    --network=host \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -e DISPLAY \
    -e "PRIVACY_CONSENT=Y" \
    -v /data1/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v /data1/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v /data1/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v /data1/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v /data1/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v /data1/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v /data1/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v /data1/docker/isaac-sim/documents:/root/Documents:rw \
    ${IMAGE_NAME}:${IMAGE_TAG}
