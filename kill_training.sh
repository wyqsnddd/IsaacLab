#!/bin/bash
# kill_isaac_training.sh

# 参数检查
if [ $# -eq 0 ]; then
    echo "使用方法: $0 <任务名称>"
    echo "示例: $0 Isaac-Velocity-Flat-D9-v0"
    exit 1
fi

TASK_NAME="$1"
SIGNAL="SIGTERM"

# 检查必要命令
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "提示: 命令 '$1' 未安装，部分功能可能受限"
        return 1
    fi
    return 0
}

# 查找进程
echo "扫描任务 '$TASK_NAME' 的进程..."
PIDS=$(ps aux | grep -E "[t]rain.py.*--task $TASK_NAME" | awk '{print $2}')

[ -z "$PIDS" ] && echo "无运行中任务" && exit 0

# 终止进程
for PID in $PIDS; do
    if ! ps -p $PID >/dev/null; then
        echo "进程 $PID 已消失，跳过"
        continue
    fi

    # 使用ps命令获取进程命令行，这样更可靠
    CMDLINE=$(ps -p $PID -o cmd= 2>/dev/null)
    if echo "$CMDLINE" | grep -q "train.py.*--task $TASK_NAME"; then
        echo "终止进程树 PID:$PID"

        if check_command pstree; then
            pstree -p $PID
        fi

        sudo kill -$SIGNAL -- -$(ps -o pgid= $PID | tr -d ' ') 2>/dev/null

        # 等待进程终止，最多等待5秒
        for i in {1..5}; do
            if ! ps -p $PID >/dev/null; then
                echo "进程 $PID 已成功终止"
                break
            fi
            sleep 1
        done

        # 如果进程仍然存在，使用 SIGKILL 强制终止
        if ps -p $PID >/dev/null; then
            echo "进程 $PID 未响应，正在强制终止..."
            sudo kill -9 $PID
            sleep 1
            if ps -p $PID >/dev/null; then
                echo "警告: 进程 $PID 仍然存在，可能需要手动检查"
            else
                echo "进程 $PID 已强制终止"
            fi
        fi
    else
        echo "跳过无关进程 PID:$PID"
    fi
done

# 最终检查
if ps aux | grep -q "[t]rain.py.*--task $TASK_NAME"; then
    echo "警告: 仍有残留进程，尝试手动检查:"
    ps aux | grep "[t]rain.py.*--task $TASK_NAME"
    exit 1
else
    echo "所有'$TASK_NAME'任务已终止"
    exit 0
fi
