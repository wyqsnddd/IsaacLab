#!/bin/bash
# run-container.sh

# Check if Vim is installed (critical for editing)
if ! command -v vim &> /dev/null; then
    echo "Error: Vim is required but not installed. Install with 'apt-get install vim' first!" >&2
    exit 1
fi

# 1. Run Container & Get ID (unchanged logic)
IMAGE_NAME="mirrors.tencent.com/rl-for-control/isaac-lab-mc_rtc"
IMAGE_TAG="latest"
CONTAINER_NAME="isaac-lab-mc_rtc-yuquan"

CONTAINER_ID=$(docker ps -q --filter "ancestor=${IMAGE_NAME}:${IMAGE_TAG}")
[ -z "$CONTAINER_ID" ] && { echo "Error: No running container found!"; exit 1; }

# 2. Create Temporary File for Commit Message
TMP_MSG_FILE=$(mktemp /tmp/docker-commit-msg.XXXXXX)
trap 'rm -f "$TMP_MSG_FILE"' EXIT  # Cleanup on exit

# Open Vim to edit the message (blocking process)
vim "$TMP_MSG_FILE"

# Verify message content after editing
if [ ! -s "$TMP_MSG_FILE" ]; then
    echo "Error: Empty commit message. Aborting." >&2
    exit 1
fi

# 3. Commit with Edited Message
docker commit \
  -m "$(cat "$TMP_MSG_FILE")" \
  -a "Yuquan" \
  "$CONTAINER_ID" \
  "${IMAGE_NAME}:${IMAGE_TAG}"

# Post-commit validation
if [ $? -eq 0 ]; then
    echo "Success: Image committed with message:"
    echo "---"
    cat "$TMP_MSG_FILE"
    echo "---"
else
    echo "Error: Commit failed!" >&2
    exit 1
fi
