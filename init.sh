#!/bin/bash

# 设置你的环境名称
ENV_NAME="llm-from-scratch" # 请修改为你想要的名称

echo "开始创建 Conda 环境: $ENV_NAME ..."
# 1. 从 yml 文件创建基础 Conda 环境
conda env create -f environment.yml

# 2. 激活环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "开始安装 PyTorch (CUDA 13.0) ..."
# 3. 显式安装带有特殊源的 PyTorch
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

echo "开始安装其他 pip 依赖 ..."
# 4. 安装剩余的 pip 包
pip install -r requirements.txt

echo "环境 $ENV_NAME 配置完成！"