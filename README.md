# LLM from Scratch

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

从零实现的大语言模型，包含完整的数据处理、模型训练和推理管道。

</div>

## ✨ 特性

- 🔬 **从零实现** - Transformer架构的所有核心组件，包括RoPE、RMSNorm、SwiGLU等
- 📊 **完整的数据管道** - 网页抓取 → 数据清洗 → 安全过滤 → 重复检测
- 🏷️ **自定义分词器** - 基于BPE算法的高效分词器实现
- 🚀 **高效训练** - 支持GPU加速、混合精度、检查点管理
- 📈 **实验追踪** - 集成WandB进行训练监控和结果分析
- 📁 **配置驱动** - 统一使用YAML配置文件，无需环境变量

## 📋 项目结构

```
llm-from-scratch/
├── src/                    # 核心源代码
│   ├── transformer.py      # Transformer模型实现
│   ├── flashatten2.py      # FlashAttention2实现
│   ├── tokenizer.py        # BPE分词器
│   └── data.py             # 数据处理管道
├── scripts/                # 执行脚本
│   ├── data/              # 数据处理脚本
│   ├── train/             # 训练脚本
│   ├── tokenizer/         # 分词器脚本
│   ├── eval/              # 评估脚本
│   └── utils.py           # 工具函数
├── configs/               # 配置文件
│   ├── data_config.yaml   # 数据处理配置
│   ├── train_config.yaml  # 模型训练配置
│   ├── eval_config.yaml   # 模型评估配置
│   ├── tokenizer_config.yaml # 分词器配置
│   └── tokenizer_config.json # 分词器配置文件
├── data/                  # 数据目录
├── checkpoints/           # 模型检查点
└── requirements.txt       # 依赖包
```

## 🛠️ 快速开始

### 环境准备

```bash
# 克隆仓库
git clone https://github.com/yourusername/llm-from-scratch.git
cd llm-from-scratch

# 运行环境配置脚本
chmod +x init.sh
./init.sh
```

### 下载数据

- [NSFW](https://dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/jigsaw_fasttext_bigrams_nsfw_final.bin)
- [TOXIC](https://dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/jigsaw_fasttext_bigrams_hatespeech_final.bin)
- [enwiki](https://downloads.cs.stanford.edu/nlp/data/nfliu/cs336-spring-2024/assignment4/enwiki-20240420-extracted_urls.txt.gz)
- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)

**NSFW** 和 **TOXIC** 模型来自 [Soldaini et al.][2] 提供的 `fasttext` 预训练模型，用来判断文本的毒性和NSFW内容。

[**enwiki**][1] 包含了从英文维基百科提取的URL列表。

[**TinyStories**][3] 用来训练分词器。

## 📊 配置文件

修改 `configs/` 目录下的相应配置文件：

**数据配置 ([configs/data_config.yaml](configs/data_config.yaml))**:

```yaml
data:
  max_samples: 4000
  output_dir: output # 数据处理输出目录

paths:
  urls_path: path/to/enwiki-20240420-extracted_urls.txt.gz # 包含URL列表的文件路径
  toxic_path: path/to/jigsaw_fasttext_bigrams_hatespeech_final.bin # 毒性检测模型路径
  nsfw_path: path/to/jigsaw_fasttext_bigrams_nsfw_final.bin # NSFW检测模型路径

output:
  webpage_file_name: webpage_temp.txt
  webpage_cleaned_file_name: webpage_temp_cleaned.txt

clean:
  chunk_size: 1000 # 在清洗过程中处理的文本块大小
```

**训练配置 ([configs/train_config.yaml](configs/train_config.yaml))**:

```yaml
model:
  context_length: 512 # 上下文长度（最大序列长度）
  vocab_size: 65536 # 词表大小
  d_model: 128 # 模型维度
  num_layers: 2 # Transformer层数
  num_heads: 2 # 注意力头数
  d_ff: 512 # 前馈网络隐藏层维度
  theta: 10000.0 # RoPE旋转位置编码参数

data:
  train: data/train # 训练数据路径
  valid: data/valid # 验证数据路径

training:
  num_epochs: 50 # 训练轮数
  batch_size: 32 # 批次大小
  learning_rate: 6.0e-4 # 学习率
  weight_decay: 0.1 # 权重衰减系数
  warmup_steps: 1000 # 学习率预热步数
  lr_decay_steps: 100000 # 学习率衰减步数
  gradient_clip: 1.0 # 梯度裁剪阈值

checkpoint:
  save_interval: 500 # 检查点保存间隔步数
  save_dir: checkpoints # 检查点保存目录
  use_flash_attention: true # 是否使用FlashAttention

device: cuda # 训练设备 ('cuda' 或 'cpu')
dtype: float32 # 计算精度

wandb:
  project: "llm-from-scratch" # WandB项目名
  run_name: "default_run" # WandB运行名
```

**分词器配置 ([configs/tokenizer_config.yaml](configs/tokenizer_config.yaml))**:

```yaml
data:
  target_txt: "path/to/your/text_file.txt" # 待分词的文本文件路径
  train: "data/raw/train.txt" # 训练数据路径

tokenizer:
  vocab_size: 65536 # 词表大小
  special_tokens: ["|<|endoftext|>"]  # 特殊标记列表，默认包含 <|endoftext|>

output:
  train_file_path: "data/train" # 训练数据输出路径
  valid_file_path: "data/valid" # 验证数据输出路径
  tokenizer_config_path: "configs/tokenizer_config.json" # 分词器配置保存路径
```

**评估配置 ([configs/eval_config.yaml](configs/eval_config.yaml))**:

```yaml
model:
  checkpoint_path: "checkpoints/checkpoint_best.pt" # 模型检查点路径
  context_length: 512 # 上下文长度
  vocab_size: 65536 # 词表大小
  d_model: 128 # 模型维度
  num_layers: 2 # 层数
  num_heads: 2 # 注意力头数
  d_ff: 512 # 前馈网络维度
  theta: 10000.0 # RoPE参数

evaluation:
  temperature: 0.8 # 生成温度
  max_new_tokens: 100 # 最大生成长度
  top_k: 50 # Top-k采样参数
  top_p: 0.95 # Top-p采样参数
  repetition_penalty: 1.2 # 重复惩罚系数

device: cuda # 推理设备
```

若保留默认配置(`configs/**.json`),将使用 [TinyStories][3] 作为分词器训练数据，并使用 `<|endoftext|>` 作为特殊标记。

### 数据处理

```bash
# 下载和清洗数据
python -m scripts.data.runner --config configs/data_config.yaml

# 训练分词器
python -m scripts.tokenizer.runner --config configs/tokenizer_config.yaml --skip-encode

# 对数据进行编码
python -m scripts.tokenizer.runner --config configs/tokenizer_config.yaml --skip-tokenizer
```

### 模型训练

```bash
# 开始训练
python -m scripts.train.runner --config configs/train_config.yaml

# 从检查点恢复训练
python -m scripts.train.runner --config configs/train_config.yaml --resume checkpoints/checkpoint_epoch_i.pt
```

## 🔬 核心组件

### Transformer模型

实现了完整的Transformer架构，包含：

- **自定义Linear层** - 截断正态分布初始化
- **RMSNorm** - 更高效的归一化层
- **RoPE (Rotary Positional Embedding)** - 旋转位置编码
- **多头注意力** - 带因果掩码的缩放点积注意力
- **SwiGLU FFN** - $FFN(x) = W_2(SiLU(W_1x) \odot W_3x)$

### 数据处理管道

1. **网页抓取** - 从URL列表批量获取网页内容
2. **质量过滤** - Gopher规则检查
3. **安全过滤** - NSFW和毒性内容检测
4. **隐私保护** - PII信息自动掩码
5. **去重机制** - MinHash + LSH算法去重

### BPE分词器

- 完整的BPE算法实现
- 支持流式大文件处理
- 自定义特殊标记支持
- 高效的编码解码

## 🙏 鸣谢

[Stanford CS336](https://cs336.stanford.edu/)

## 📄 许可证

MIT License

[1]: https://cs336.stanford.edu/
[2]: https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge
[3]: https://arxiv.org/abs/2305.07759

