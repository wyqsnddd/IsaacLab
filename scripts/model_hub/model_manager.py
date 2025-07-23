# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import shutil
import torch
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from huggingface_hub import HfApi, create_repo, snapshot_download


class ModelManager:
    def __init__(self, cache_dir: str | None = None):
        """
        初始化模型管理器

        Args:
            cache_dir: 模型缓存目录，默认为 ~/.cache/huggingface/hub
        """
        self.api = HfApi()
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface/hub")

    def create_versioned_repo(self, base_name: str, version: str | None = None) -> str:
        """
        创建带版本号的仓库

        Args:
            base_name: 基础仓库名称
            version: 版本号，如果为 None 则使用日期

        Returns:
            完整的仓库ID
        """
        if version is None:
            version = datetime.now().strftime("%Y-%m-%d")

        repo_id = f"{base_name}-{version}"
        self.api.create_repo(repo_id, repo_type="model")
        return repo_id

    def create_branch(self, repo_id: str, branch_name: str) -> None:
        """
        在仓库中创建新分支

        Args:
            repo_id: 仓库ID
            branch_name: 分支名称
        """
        self.api.create_branch(repo_id=repo_id, branch=branch_name)

    def upload_to_branch(
        self, output_dir: str, repo_id: str, branch_name: str, key_checkpoints: list[int] = None
    ) -> None:
        """
        上传模型到指定分支

        Args:
            output_dir: 训练输出目录
            repo_id: 仓库ID
            branch_name: 分支名称
            key_checkpoints: 需要保存的关键检查点列表
        """
        # 创建临时目录
        temp_dir = Path("temp_upload")
        temp_dir.mkdir(exist_ok=True)

        try:
            # 上传检查点
            if key_checkpoints:
                for checkpoint in key_checkpoints:
                    model_file = f"model_{checkpoint}.pt"
                    src_path = os.path.join(output_dir, model_file)
                    if os.path.exists(src_path):
                        dst_path = temp_dir / model_file
                        shutil.copy2(src_path, dst_path)
                        self.api.upload_file(
                            path_or_fileobj=str(dst_path),
                            path_in_repo=f"checkpoints/{model_file}",
                            repo_id=repo_id,
                            revision=branch_name,
                        )

            # 上传配置文件
            params_dir = os.path.join(output_dir, "params")
            if os.path.exists(params_dir):
                for file in os.listdir(params_dir):
                    src_path = os.path.join(params_dir, file)
                    dst_path = temp_dir / file
                    shutil.copy2(src_path, dst_path)
                    self.api.upload_file(
                        path_or_fileobj=str(dst_path),
                        path_in_repo=f"params/{file}",
                        repo_id=repo_id,
                        revision=branch_name,
                    )

            # 创建并上传 README
            readme_content = f"""# Model Version: {branch_name}

## Model Checkpoints
{'- ' + chr(10) + '- '.join([f'Checkpoint {cp}: model_{cp}.pt' for cp in key_checkpoints]) if key_checkpoints else ''}

## Training Configuration
See the `params/` directory for training configuration files.

## Usage
1. Download the model checkpoint
2. Load using PyTorch:
```python
import torch
model = torch.load('path/to/model.pt')
```
"""

            readme_path = temp_dir / "README.md"
            with open(readme_path, "w") as f:
                f.write(readme_content)

            self.api.upload_file(
                path_or_fileobj=str(readme_path), path_in_repo="README.md", repo_id=repo_id, revision=branch_name
            )

        finally:
            shutil.rmtree(temp_dir)

    def upload_model(
        self, model_path: str, repo_id: str, model_name: str = "model.pt", create_repo: bool = True
    ) -> None:
        """
        上传模型到 Hugging Face Hub

        Args:
            model_path: 本地模型文件路径
            repo_id: 仓库ID，格式为 "username/model-name"
            model_name: 模型文件名
            create_repo: 是否创建新仓库
        """
        if create_repo:
            self.api.create_repo(repo_id, repo_type="model")

        self.api.upload_file(path_or_fileobj=model_path, path_in_repo=model_name, repo_id=repo_id)

    def download_model(
        self, repo_id: str, model_name: str = "model.pt", local_dir: str | None = None
    ) -> torch.nn.Module:
        """
        从 Hugging Face Hub 下载模型

        Args:
            repo_id: 仓库ID，格式为 "username/model-name"
            model_name: 模型文件名
            local_dir: 本地保存目录

        Returns:
            加载的模型
        """
        model_path = snapshot_download(
            repo_id=repo_id, local_dir=local_dir or self.cache_dir, local_dir_use_symlinks=False
        )

        return torch.load(f"{model_path}/{model_name}")

    def list_models(self, repo_id: str) -> list:
        """
        列出仓库中的所有模型文件

        Args:
            repo_id: 仓库ID

        Returns:
            文件列表
        """
        return self.api.list_repo_files(repo_id)

    def upload_checkpoints(
        self, output_dir: str, repo_id: str, checkpoints: list[int], create_repo: bool = False
    ) -> None:
        """
        上传指定的检查点到 Hugging Face Hub

        Args:
            output_dir: 训练输出目录
            repo_id: 仓库ID
            checkpoints: 需要上传的检查点列表
            create_repo: 是否创建新仓库
        """
        if create_repo:
            self.api.create_repo(repo_id, repo_type="model")

        # 创建临时目录
        temp_dir = Path("temp_upload")
        temp_dir.mkdir(exist_ok=True)

        try:
            # 上传指定的检查点
            for checkpoint in checkpoints:
                model_file = f"model_{checkpoint}.pt"
                src_path = os.path.join(output_dir, model_file)
                if os.path.exists(src_path):
                    print(f"Uploading checkpoint {checkpoint}...")
                    dst_path = temp_dir / model_file
                    shutil.copy2(src_path, dst_path)
                    self.api.upload_file(
                        path_or_fileobj=str(dst_path), path_in_repo=f"checkpoints/{model_file}", repo_id=repo_id
                    )
                else:
                    print(f"Warning: Checkpoint {checkpoint} not found at {src_path}")

        finally:
            # 清理临时目录
            shutil.rmtree(temp_dir)

    def upload_isaaclab_output(
        self,
        output_dir: str,
        repo_id: str,
        key_checkpoints: list[int] = None,
        create_repo: bool = True,
        notes: str = None,
    ) -> None:
        """
        上传 IsaacLab 训练输出到 Hugging Face Hub

        Args:
            output_dir: 训练输出目录
            repo_id: 仓库ID
            key_checkpoints: 需要保存的关键检查点列表，如 [5000, 9999]
            create_repo: 是否创建新仓库
            notes: 自定义笔记，将被添加到 README.md 中
        """
        if create_repo:
            self.api.create_repo(repo_id, repo_type="model")

        # 创建临时目录
        temp_dir = Path("temp_upload")
        temp_dir.mkdir(exist_ok=True)

        try:
            # 1. 保存关键检查点
            if key_checkpoints:
                for checkpoint in key_checkpoints:
                    model_file = f"model_{checkpoint}.pt"
                    src_path = os.path.join(output_dir, model_file)
                    if os.path.exists(src_path):
                        dst_path = temp_dir / model_file
                        shutil.copy2(src_path, dst_path)
                        self.api.upload_file(
                            path_or_fileobj=str(dst_path), path_in_repo=f"checkpoints/{model_file}", repo_id=repo_id
                        )

            # 2. 保存配置文件
            params_dir = os.path.join(output_dir, "params")
            if os.path.exists(params_dir):
                for file in os.listdir(params_dir):
                    src_path = os.path.join(params_dir, file)
                    dst_path = temp_dir / file
                    shutil.copy2(src_path, dst_path)
                    self.api.upload_file(path_or_fileobj=str(dst_path), path_in_repo=f"params/{file}", repo_id=repo_id)

            # 3. 保存训练日志
            for file in os.listdir(output_dir):
                if file.startswith("events.out.tfevents"):
                    src_path = os.path.join(output_dir, file)
                    dst_path = temp_dir / file
                    shutil.copy2(src_path, dst_path)
                    self.api.upload_file(path_or_fileobj=str(dst_path), path_in_repo=f"logs/{file}", repo_id=repo_id)

            # 4. 保存 exported 文件（.pt 和 .onnx）
            exported_dir = os.path.join(output_dir, "exported")
            if os.path.exists(exported_dir):
                for file in os.listdir(exported_dir):
                    if file.endswith((".pt", ".onnx")):
                        src_path = os.path.join(exported_dir, file)
                        dst_path = temp_dir / file
                        shutil.copy2(src_path, dst_path)
                        self.api.upload_file(
                            path_or_fileobj=str(dst_path), path_in_repo=f"exported/{file}", repo_id=repo_id
                        )

            # 5. 创建并上传 README
            readme_content = f"""# IsaacLab Training Output

## Model Checkpoints
- Final model: model_9999.pt
{'- ' + chr(10) + '- '.join([f'Checkpoint {cp}: model_{cp}.pt' for cp in key_checkpoints]) if key_checkpoints else ''}

## Exported Models
Optimized models for deployment are available in the `exported/` directory:
- `policy.pt`: Optimized PyTorch model
- `policy.onnx`: ONNX format model for cross-platform deployment

## Training Configuration
See the `params/` directory for training configuration files.

## Training Logs
Training logs are available in the `logs/` directory.

## Demo Videos
Demo videos are available in the `videos/` directory.

## Usage
### Using PyTorch Checkpoints
```python
import torch
model = torch.load('path/to/model.pt')
```

### Using Exported Models
```python
# PyTorch
import torch
model = torch.load('exported/policy.pt')

# ONNX (requires onnxruntime)
import onnxruntime as ort
session = ort.InferenceSession('exported/policy.onnx')
```

## Notes
{notes if notes else 'No additional notes provided.'}
"""

            readme_path = temp_dir / "README.md"
            with open(readme_path, "w") as f:
                f.write(readme_content)

            self.api.upload_file(path_or_fileobj=str(readme_path), path_in_repo="README.md", repo_id=repo_id)

        finally:
            # 清理临时目录
            shutil.rmtree(temp_dir)
