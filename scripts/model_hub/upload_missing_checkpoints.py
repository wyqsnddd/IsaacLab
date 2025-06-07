# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from model_manager import ModelManager


def main():
    # 初始化模型管理器
    manager = ModelManager()

    # 设置路径和仓库信息
    output_dir = "../../../data/5-30-rough-std-4/2025-05-30_15-55-48/"
    repo_id = "Yuquan-Wang/pudu-d9-rough-sota-5-30"

    # 指定要上传的检查点
    missing_checkpoints = [12000, 14000, 14999]  # 替换为您需要上传的检查点

    print("Uploading missing checkpoints...")
    manager.upload_checkpoints(
        output_dir=output_dir, repo_id=repo_id, checkpoints=missing_checkpoints, create_repo=False  # 设置为 False，因为仓库已经存在
    )

    # 列出仓库中的文件
    print("\nListing repository files...")
    files = manager.list_models(repo_id)
    print("Files in repository:", files)


if __name__ == "__main__":
    main()
