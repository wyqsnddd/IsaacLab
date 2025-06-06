# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from source.utils.model_manager import ModelManager


def main():
    # 初始化模型管理器
    manager = ModelManager()

    # 示例：上传 IsaacLab 训练输出
    output_dir = "6-5-set-3-running-lin-vel/2025-06-05_03-41-58"
    repo_id = "your-username/d9-rough-walking"  # 替换为您的用户名和仓库名

    # 定义关键检查点
    key_checkpoints = [5000, 9999]  # 可以根据需要修改

    print("Uploading IsaacLab training output...")
    manager.upload_isaaclab_output(output_dir=output_dir, repo_id=repo_id, key_checkpoints=key_checkpoints)

    # 列出仓库中的文件
    print("\nListing repository files...")
    files = manager.list_models(repo_id)
    print("Files in repository:", files)


if __name__ == "__main__":
    main()
