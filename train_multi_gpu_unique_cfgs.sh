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

# 检查配置文件是否存在
for i in $(seq 0 $(($total_gpus - 1))); do
    if [ ! -f "training-${i}.yaml" ]; then
        echo "错误: 找不到配置文件 training-${i}.yaml"
        exit 1
    fi
done

echo "模式: $([ "$WAIT_ALL_GPUS" = true ] && echo "等待所有GPU" || echo "仅使用空闲GPU")"

while true; do
    # 获取所有 GPU 的利用率（格式：gpu_index utilization）
    gpu_utils=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')

    # 遍历每个 GPU
    while IFS=',' read -r gpu_id utilization; do
        # 检查 GPU 是否已被使用
        if [[ -z "${used_gpus[$gpu_id]}" ]] && (( utilization < 10 )); then
            # 生成带 GPU ID 的日志文件名
            log_file="${LOG_PREFIX}_multi_gpu_${gpu_id}_run${used_count}.log"

            echo "[$(date +'%F %T')] GPU $gpu_id 空闲（利用率 ${utilization}%），启动训练任务..."
            echo "使用参数配置 #$used_count"

            # 启动训练任务并捕获 PID
            pid=$(./modify_weight_unique_cfgs.sh "$gpu_id" "$used_count" "$log_file")

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
    elif (( used_count >= total_gpus )); then
        break
    else
        # Check for available GPUs if not waiting for all GPUs
        available_gpus=0
        while IFS=',' read -r gpu_id utilization; do
            if [[ -z "${used_gpus[$gpu_id]}" ]] && (( utilization < 10 )); then
                ((available_gpus++))
            fi
        done <<< "$gpu_utils"

        if (( available_gpus == 0 )); then
            echo "[$(date +'%F %T')] No more available GPUs, monitoring ended."
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
