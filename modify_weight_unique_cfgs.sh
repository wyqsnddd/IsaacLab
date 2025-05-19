#!/bin/bash

# 检查yq是否安装
if ! python3 -c "import yq" &> /dev/null; then
    echo "错误: 需要安装 yq 来解析 YAML 文件"
    echo "请运行: pip install yq"
    exit 1
fi

# 获取命令行参数
gpu_id=$1
run_count=$2
log_file=$3

if [ -z "$gpu_id" ] || [ -z "$run_count" ] || [ -z "$log_file" ]; then
    echo "错误: 缺少必要参数"
    echo "用法: $0 <gpu_id> <run_count> <log_file>"
    exit 1
fi

# 配置文件路径
CONFIG_FILE="source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/d9/flat_env_cfg.py"
PARAM_FILE="training-${run_count}.yaml"

# 检查参数文件是否存在
if [ ! -f "$PARAM_FILE" ]; then
    echo "错误: 找不到参数文件 $PARAM_FILE"
    exit 1
fi

echo "[$(date +'%F %T')] 开始修改配置..."

# 备份原始文件
cp "$CONFIG_FILE" "${CONFIG_FILE}.bak"

# 获取参数数量
param_count=$(yq '.parameters | length' "$PARAM_FILE")

# 遍历每个参数并修改配置文件
for ((i=0; i<param_count; i++)); do
    full_param_name=$(yq ".parameters[$i].name" "$PARAM_FILE" | tr -d '"')  # 移除引号
    param_name=${full_param_name%.weight}  # 移除 .weight 后缀
    param_weight=$(yq ".parameters[$i].weights" "$PARAM_FILE")  # 使用当前配置文件中的权重值

    echo "准备修改参数: ${param_name}, 新权重值: ${param_weight}"

    # 检查当前权重值
    echo "修改前的配置:"
    grep "self.rewards.${param_name}.weight" "$CONFIG_FILE"

    # 使用更精确的sed模式匹配，处理变量或数值的情况
    sed -i "s/\([ ]*self.rewards.${param_name}.weight = \)[a-zA-Z0-9._-]*/\1${param_weight}/" "$CONFIG_FILE"

    # 验证修改
    echo "修改后的配置:"
    grep "self.rewards.${param_name}.weight" "$CONFIG_FILE"
    echo "----------------------------------------"
done

echo "所有权重已修改完成"

# 启动训练任务
echo "[$(date +'%F %T')] 启动训练任务..."
echo "执行命令: ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-D9-v0 --headless --num_envs 4096 --video --video_length 200 --video_interval 1000 --device cuda:$gpu_id"

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-D9-v0 \
    --headless \
    --num_envs 4096 \
    --video --video_length 200 --video_interval 1000 \
    --device "cuda:$gpu_id" \
    > "$log_file" 2>&1 &

# 保存进程ID
pid=$!
echo "训练进程 PID: $pid"

# 检查进程是否存在
if ! ps -p $pid > /dev/null; then
    echo "错误: 训练进程未能成功启动"
    # 恢复原始文件
    mv "${CONFIG_FILE}.bak" "$CONFIG_FILE"
    exit 1
fi

# 等待日志文件创建
echo "等待日志文件创建..."
timeout=30
while [ ! -f "$log_file" ]; do
    if [ "$timeout" -le 0 ]; then
        echo "错误: 日志文件未能创建"
        kill $pid 2>/dev/null
        mv "${CONFIG_FILE}.bak" "$CONFIG_FILE"
        exit 1
    fi
    sleep 1
    ((timeout--))
done

# 等待环境设置完成
echo "等待环境设置完成..."
timeout=300  # 5分钟超时
while true; do
    if ! ps -p $pid > /dev/null; then
        echo "错误: 训练进程意外终止"
        cat "$log_file"  # 显示日志内容以帮助调试
        mv "${CONFIG_FILE}.bak" "$CONFIG_FILE"
        exit 1
    fi

    if grep -q "Completed setting up the environment" "$log_file"; then
        echo "[$(date +'%F %T')] 环境设置已完成"
        break
    fi

    if [ "$timeout" -le 0 ]; then
        echo "错误: 等待环境设置超时"
        echo "最后100行日志内容:"
        tail -n 100 "$log_file"
        kill $pid 2>/dev/null
        mv "${CONFIG_FILE}.bak" "$CONFIG_FILE"
        exit 1
    fi

    echo -n "."  # 显示等待进度
    sleep 1
    ((timeout--))
done

# 恢复原始文件
mv "${CONFIG_FILE}.bak" "$CONFIG_FILE"
echo "配置已恢复"

# 返回进程ID
echo $pid
