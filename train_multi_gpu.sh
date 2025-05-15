#!/bin/bash

# 设置检测间隔（秒）和日志文件前缀
MONITOR_INTERVAL=30
LOG_PREFIX="train_gpu"

# 检查yq是否安装
if ! python3 -c "import yq" &> /dev/null; then
    echo "错误: 需要安装 yq 来解析 YAML 文件"
    echo "请运行: pip install yq"
    exit 1
fi

# 解析命令行参数
WAIT_ALL_GPUS=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            WAIT_ALL_GPUS=true
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [--all]"
            exit 1
            ;;
    esac
done

# 创建数组来跟踪已使用的 GPU
declare -A used_gpus
total_gpus=1
used_count=0

# 从YAML文件读取参数
PARAM_FILE="training_params.yaml"
if [ ! -f "$PARAM_FILE" ]; then
    echo "错误: 找不到参数文件 $PARAM_FILE"
    exit 1
fi

# 获取参数数量
param_count=$(yq '.parameters | length' "$PARAM_FILE")

echo "模式: $([ "$WAIT_ALL_GPUS" = true ] && echo "等待所有GPU" || echo "仅使用空闲GPU")"

while true; do
    # 获取所有 GPU 的利用率（格式：gpu_index utilization）
    gpu_utils=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')

    # 遍历每个 GPU
    while IFS=',' read -r gpu_id utilization; do
        # 检查 GPU 是否已被使用
        if [[ -z "${used_gpus[$gpu_id]}" ]] && (( utilization < 10 )); then
            # 构建cfg-override参数
            cfg_overrides=""
            for ((i=0; i<param_count; i++)); do
                weights=(${param_weights[$i]})
                param_name=$(yq ".parameters[$i].name" "$PARAM_FILE")
                param_weight=$(yq ".parameters[$i].weights[$used_count]" "$PARAM_FILE")
                cfg_overrides="$cfg_overrides +$param_name=$param_weight"
            done

            # 生成带 GPU ID 的日志文件名
            log_file="${LOG_PREFIX}_${gpu_id}_run${used_count}.log"

            echo "[$(date +'%F %T')] GPU $gpu_id 空闲（利用率 ${utilization}%），启动训练任务..."
            echo "使用参数配置 #$used_count"

            # 启动后台任务并捕获 PID
            eval "nohup ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
                --task Isaac-Velocity-Flat-D9-v0 \
                --headless \
                --num_envs 4096 \
                --video --video_length 200 --video_interval 1000 \
                --device \"cuda:$gpu_id\" \
                $cfg_overrides > \"$log_file\" 2>&1 &"
            pid=$!

            # 记录已使用的 GPU
            used_gpus[$gpu_id]=$pid
            ((used_count++))

            echo "GPU $gpu_id 进程 PID: $pid | 日志文件: $log_file"
            echo "已使用 GPU 数量: $used_count/$total_gpus"

            # 检查是否所有 GPU 都已使用
            if [ "$WAIT_ALL_GPUS" = true ] && (( used_count >= total_gpus )); then
                echo "[$(date +'%F %T')] 所有 GPU 都已分配任务，监控结束。"
                exit 0
            fi

            # 延迟防止重复触发
            sleep 60
        fi
    done <<< "$gpu_utils"

    # 如果等待所有GPU且还有未使用的GPU，继续监控
    if [ "$WAIT_ALL_GPUS" = true ] && (( used_count < total_gpus )); then
        sleep $MONITOR_INTERVAL
    else
        # 如果不等待所有GPU，检查是否还有可用的GPU
        available_gpus=0
        while IFS=',' read -r gpu_id utilization; do
            if [[ -z "${used_gpus[$gpu_id]}" ]] && (( utilization < 10 )); then
                ((available_gpus++))
            fi
        done <<< "$gpu_utils"

        if (( available_gpus == 0 )); then
            echo "[$(date +'%F %T')] 没有更多可用的GPU，监控结束。"
            break
        fi
        sleep $MONITOR_INTERVAL
    fi
done

# 显示最终状态
echo "=== 最终状态 ==="
echo "已使用的 GPU 列表："
for gpu_id in "${!used_gpus[@]}"; do
    echo "GPU $gpu_id: PID ${used_gpus[$gpu_id]}"
done
echo "总使用 GPU 数量: $used_count/$total_gpus"
