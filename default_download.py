import os
import gzip
import shutil
import urllib.request
from datasets import load_dataset
from tqdm import tqdm

# ==========================================
# 1. 创建目标文件夹结构
# ==========================================
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/tokenizer", exist_ok=True)

# ==========================================
# 2. 定义直接下载链接及目标路径
# ==========================================
download_tasks = [
    {
        "url": "https://dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/jigsaw_fasttext_bigrams_nsfw_final.bin",
        "dest": "data/raw/jigsaw_fasttext_bigrams_nsfw_final.bin"
    },
    {
        "url": "https://dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/jigsaw_fasttext_bigrams_hatespeech_final.bin",
        "dest": "data/raw/jigsaw_fasttext_bigrams_hatespeech_final.bin"
    },
    {
        "url": "https://downloads.cs.stanford.edu/nlp/data/nfliu/cs336-spring-2024/assignment4/enwiki-20240420-extracted_urls.txt.gz",
        "dest": "data/raw/enwiki-20240420-extracted_urls.txt.gz",
        "extract_to": "data/raw/enwiki-20240420-extracted_urls.txt"
    }
]

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

print("--- 开始下载静态文件 ---")
for task in download_tasks:
    dest_path = task["dest"]
    if not os.path.exists(dest_path) and not os.path.exists(task.get("extract_to", "never_matches")):
        print(f"Downloading {task['url']}...")
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(dest_path)) as t:
            urllib.request.urlretrieve(task["url"], filename=dest_path, reporthook=t.update_to)
    else:
        print(f"[{dest_path}] 已存在，跳过下载。")

    # 处理 GZ 解压
    if "extract_to" in task:
        extract_path = task["extract_to"]
        if os.path.exists(dest_path) and not os.path.exists(extract_path):
            print(f"解压 {dest_path} 到 {extract_path}...")
            with gzip.open(dest_path, 'rb') as f_in:
                with open(extract_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(dest_path) 
            print(f"解压完成。")

# ==========================================
# 3. 通过 Hugging Face 下载并导出数据集
# ==========================================
def export_dataset_to_txt(dataset_split, output_filepath):
    """将 Dataset 对象中的文本提取并写入本地 txt 文件"""
    if os.path.exists(output_filepath):
        print(f"[{output_filepath}] 已存在，跳过导出。")
        return
    
    print(f"正在将数据导出至 {output_filepath} ...")
    with open(output_filepath, "w", encoding="utf-8") as f:
        for item in tqdm(dataset_split, desc=f"Writing {os.path.basename(output_filepath)}"):
            f.write(item["text"] + "\n")

print("\n--- 开始处理 Hugging Face 数据集 ---")

# TinyStories
print("1. 加载 roneneldan/TinyStories ...")
ds_tiny = load_dataset("roneneldan/TinyStories")
export_dataset_to_txt(ds_tiny["train"], "data/tokenizer/TinyStoriesV2-GPT4-train.txt")
export_dataset_to_txt(ds_tiny["validation"], "data/tokenizer/TinyStoriesV2-GPT4-valid.txt")

# OpenWebText
print("2. 加载 Skylion007/openwebtext ...")
ds_owt = load_dataset("Skylion007/openwebtext", split="train")

print("划分 OpenWebText 训练集和验证集...")
owt_splits = ds_owt.train_test_split(test_size=0.005, seed=42)

export_dataset_to_txt(owt_splits["train"], "data/tokenizer/owt_train.txt")
export_dataset_to_txt(owt_splits["test"], "data/tokenizer/owt_valid.txt")

print("\n🎉 所有任务处理完毕！文件已全部放置在指定的 data/raw 和 data/tokenizer 目录中。")