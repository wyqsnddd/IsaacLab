#!/bin/bash
set -e  # 遇到错误立即退出

# 解析命令行参数
USE_DATE_TAG=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --date-tag)
            if [[ "$2" == "true" ]]; then
                USE_DATE_TAG=true
            fi
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 检查必要文件
if [ ! -f "./container.py" ]; then
    echo "Error: container.py not found!"
    exit 1
fi

# 检查docker是否已登录
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker not running or not logged in!"
    exit 1
fi

# 配置Docker使用HTTP协议访问mirrors.tencent.com
echo "Configuring Docker to use HTTP for mirrors.tencent.com..."
if [ ! -f "/etc/docker/daemon.json" ]; then
    sudo mkdir -p /etc/docker
    echo '{"insecure-registries": ["mirrors.tencent.com"]}' | sudo tee /etc/docker/daemon.json
    sudo systemctl restart docker
    echo "Docker daemon restarted with new configuration"
fi

tag_and_push() {
    if [ $# -ne 2 ]; then
        echo "Usage: tag_and_push <repository-name> <tag>"
        return 1
    fi

    local repo=$1
    local tag=$2
    local timestamp=$(date +"%Y%m%d-%H")

    # 根据USE_DATE_TAG决定使用时间戳还是latest标签
    if [ "$USE_DATE_TAG" = true ]; then
        local target_image="mirrors.tencent.com/rl-for-control/$repo:$timestamp"
    else
        local target_image="mirrors.tencent.com/rl-for-control/$repo:latest"
    fi

    echo "Processing $repo:$tag..."
    echo "Target image will be tagged as: $target_image"

    # 获取镜像ID
    local docker_image_id=$(docker images --format "{{.ID}}\t{{.Repository}}\t{{.Tag}}" |
                           awk -v repo="$repo" -v tag="$tag" '$2 == repo && $3 == tag {print $1}')

    if [ -z "$docker_image_id" ]; then
        echo "Error: Image $repo:$tag not found!"
        return 1
    fi

    # 打标签
    echo "Tagging image as $target_image..."
    if ! docker tag "$docker_image_id" "$target_image"; then
        echo "Error: Tagging failed!"
        return 1
    fi

    # 推送镜像
    echo "Pushing image to registry..."
    if ! docker push "$target_image"; then
        echo "Error: Push failed! Ensure you are logged in (docker login)."
        return 1
    fi

    echo "Success: Pushed $target_image"
}

# 主执行流程
echo "Starting build and push process..."
echo "Using date tag: $USE_DATE_TAG"

python container.py start
tag_and_push "isaac-lab-base" "latest"

python container.py start ros2
tag_and_push "isaac-lab-ros2" "latest"

python container.py start mc_rtc
tag_and_push "isaac-lab-mc_rtc" "latest"

echo "Build and push process completed!"
