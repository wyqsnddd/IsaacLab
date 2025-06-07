# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from model_manager import ModelManager


def main():
    # 初始化模型管理器
    manager = ModelManager()

    # 方法1：创建新的仓库（推荐）
    base_name = "Yuquan-Wang/pudu-d9-rough-walking"
    version = "2025-06-05-episode-650"  # 或者使用其他版本号
    repo_id = manager.create_versioned_repo(base_name, version)

    # 设置训练输出目录
    output_dir = "../data/6-5-set-4-rough-rewards/2025-06-05_07-20-04"

    # 定义关键检查点
    key_checkpoints = [2000, 5000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 14999]

    print(f"Uploading new version to repository: {repo_id}")
    manager.upload_isaaclab_output(
        output_dir=output_dir, repo_id=repo_id, key_checkpoints=key_checkpoints, create_repo=False, notes="This version limits the lateral velocity to +- 0.1 m/s."  
    )

    # 方法2：使用分支（替代方案）
    """
    # 使用现有仓库
    repo_id = "Yuquan-Wang/pudu-d9-rough-sota"
    branch_name = "v2"  # 或者使用其他分支名

    # 创建新分支
    manager.create_branch(repo_id, branch_name)

    # 上传到新分支
    manager.upload_to_branch(
        output_dir=output_dir,
        repo_id=repo_id,
        branch_name=branch_name,
        key_checkpoints=key_checkpoints
    )
    """

    # 列出仓库中的文件
    print("\nListing repository files...")
    files = manager.list_models(repo_id)
    print("Files in repository:", files)


if __name__ == "__main__":
    main()
