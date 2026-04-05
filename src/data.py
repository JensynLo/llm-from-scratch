import nltk
import re
import os
import random
import fasttext
import requests
import threading
import hashlib
import unicodedata
import concurrent.futures
from collections import defaultdict
from tqdm import tqdm
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
from scripts.utils import load_config

nltk.download("punkt_tab")

configs = load_config("configs/data_config.yaml")
nsfw_path = configs["paths"]["nsfw_path"]
toxic_path = configs["paths"]["toxic_path"]
nsfw_model = fasttext.load_model(nsfw_path)
toxic_model = fasttext.load_model(toxic_path)


EMAIL_RE = re.compile(
    r"\b[a-zA-Z0-9._%+-]+@"
    r"(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b"
)
PHONE_RE = re.compile(
    r"""
    (?<!\d)                
    (?:\+?\d{1,3}[\s\-\.]?)?   
    (?:\(?\d{2,4}\)?[\s\-\.]?) 
    \d{3,4}[\s\-\.]?\d{4}      
    (?!\d)                 
    """,
    re.VERBOSE,
)
IPV4_RE = re.compile(
    r"""
    (?<!\d)                             
    (?:
        (?:25[0-5]|2[0-4]\d|1?\d?\d)\.
        (?:25[0-5]|2[0-4]\d|1?\d?\d)\.
        (?:25[0-5]|2[0-4]\d|1?\d?\d)\.
        (?:25[0-5]|2[0-4]\d|1?\d?\d)
    )
    (?!\d)                              
    """,
    re.VERBOSE,
)
NEW_EMAIL_SUB = "|||EMAIL_ADDRESS|||"
NEW_PHONE_SUB = "|||PHONE_NUMBER|||"
NEW_IP_SUB = "|||IP_ADDRESS|||"


def extract_text_from_html(html_content: bytes) -> str:
    encoding = detect_encoding(html_content)
    html_str = html_content.decode(encoding, "replace")
    text = extract_plain_text(html_str)
    return text


def _mask_email(text: str) -> tuple[str, int]:
    count = len(EMAIL_RE.findall(text))
    new_text = EMAIL_RE.sub(NEW_EMAIL_SUB, text)
    return new_text, count


def _mask_phone(text: str) -> tuple[str, int]:
    count = len(PHONE_RE.findall(text))
    new_text = PHONE_RE.sub(NEW_PHONE_SUB, text)
    return new_text, count


def _mask_ip(text: str) -> tuple[str, int]:
    count = len(IPV4_RE.findall(text))
    new_text = IPV4_RE.sub(NEW_IP_SUB, text)
    return new_text, count


def mask_pii(text: str) -> tuple[str, int, int, int]:
    string, email_count = _mask_email(text)
    string, phone_count = _mask_phone(string)
    string, ip_count = _mask_ip(string)
    return string, email_count, phone_count, ip_count


def _mask_nsfw(text: str) -> tuple[str, float]:
    labels, probs = nsfw_model.predict(text)
    label = labels[0].replace("__label__", "") # type:ignore
    return label, probs[0]


def _mask_toxic(text: str) -> tuple[str, float]:
    labels, probs = toxic_model.predict(text)
    label = labels[0].replace("__label__", "")  # type:ignore
    return label, probs[0]


def mask_harmful(text: str) -> tuple[str, str, float, float]:
    is_nsfw, nsfw_score = _mask_nsfw(text)
    is_toxic, toxic_score = _mask_toxic(text)
    return is_nsfw, is_toxic, nsfw_score, toxic_score


def gopher(text: str) -> bool:
    words = nltk.word_tokenize(text)
    # 1. Contain less than 50 or more than 100,000 words.
    counts = len(words)
    if counts < 50 or counts > 100000:
        return False
    # 2.Have a mean word length outside the range of 3 to 10 characters
    len_per_word = list(map(len, words))
    mean_word_length = sum(len_per_word) / counts
    if mean_word_length < 3 or mean_word_length > 10:
        return False
    # 3. Have more than 30% of lines ending with an ellipsis (“...”).
    lines = text.split("\n")
    ellipsis_lines = [line for line in lines if line[-3:] == "..."]
    if len(ellipsis_lines) / len(lines) > 0.3:
        return False

    # 4. Contain less than 80% of words with at least one alphabetic character
    def has_letter(s: str) -> bool:
        return any(c.isalpha() for c in s)

    words_with_letter_counts = sum(list(map(has_letter, words)))
    if words_with_letter_counts / counts < 0.8:
        return False
    return True


def extract_urls_samples(
    wiki_file_path: str, output_file: str, max_samples: int = 5000
):
    print(f"正在从 {wiki_file_path} 流式读取 URLs 并抓取样本...")

    write_lock = threading.Lock()
    samples_collected = 0

    def fetch_single_url(url: str):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            # 使用 with 语句确保连接和内存被立即释放
            with requests.get(url, headers=headers, timeout=5) as response:
                if response.status_code != 200:
                    return None
                text = extract_text_from_html(response.content)

            clean_text = text.replace("\n", " ").replace("\r", " ").strip()
            # 进行基本的长度和质量过滤，避免后续步骤的资源浪费
            if len(clean_text) < 100 or not gopher(clean_text):
                return None

            is_nsfw, is_toxic, _, _ = mask_harmful(clean_text)
            if is_nsfw != "non-nsfw":
                return None
            if is_toxic != "non-toxic":
                return None

            clean_text, _, _, _ = mask_pii(clean_text)
            return f"__label__high {clean_text}"
        except Exception:
            pass
        return None

    with (
        open(wiki_file_path, "r", encoding="utf-8") as f_in,
        open(output_file, "w", encoding="utf-8") as f_out,
        tqdm(
            total=max_samples, desc="抓取高质量样本", unit="条"
        ) as pbar, 
    ):
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = set()

            for line in f_in:
                if samples_collected >= max_samples:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

                url = line.strip()
                if not url:
                    continue

                futures.add(executor.submit(fetch_single_url, url))

                if len(futures) >= 100:
                    done, futures = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )

                    for future in done:
                        result = future.result()
                        if result:
                            with write_lock:
                                if samples_collected < max_samples:
                                    f_out.write(result + "\n")
                                    samples_collected += 1
                                    pbar.update(1)

                                    # ：每收集 100 条就强制写入磁盘，防止崩溃丢数据
                                    if samples_collected % 100 == 0:
                                        f_out.flush()

                                elif samples_collected == max_samples:
                                    # 保达到数量后立即取消后续任务
                                    executor.shutdown(wait=False, cancel_futures=True)

            # 处理池子中剩余的任务
            for future in concurrent.futures.as_completed(futures):
                if samples_collected >= max_samples:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                result = future.result()
                if result:
                    with write_lock:
                        if samples_collected < max_samples:
                            f_out.write(result + "\n")
                            samples_collected += 1
                            pbar.update(1)  

    print(
        f"\n 样本流式落盘完成，共采集 {samples_collected} 条，保存在: {output_file}"
    )
    if samples_collected < max_samples:
        print(
            "URL 列表已耗尽，但仍未达到预期的 max_samples 数量，说明过滤条件较严。"
        )


def _normalize_text(text: str) -> str:
    # 1. 应用 NFD Unicode 标准化
    text = unicodedata.normalize("NFD", text)
    # 2. 移除重音符号 (Non-spacing marks)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    # 3. 转为小写
    text = text.lower()
    # 4. 去除标点符号 (保留字母数字和空白符)
    text = re.sub(r"[^\w\s]", "", text)
    # 5. 规范化空格
    text = re.sub(r"\s+", " ", text).strip()
    return text


class MemoryCleaner:
    def __init__(self, data_samples: list[str]) -> None:
        self.samples = data_samples
        self.cleaned_samples = []

    def _clean_page(self, text: str) -> str | None:
        """对单条文本进行清洗，返回清洗后的文本或 None（如果被过滤掉）"""
        text = _normalize_text(text)

        is_nsfw, is_toxic, nsfw_score, toxic_score = mask_harmful(text)

        if is_nsfw != "non-nsfw" or nsfw_score <= 0.75:
            return None
        if is_toxic != "non-toxic" or toxic_score < 0.75:
            return None

        if not gopher(text):
            return None

        text, _, _, _ = mask_pii(text)
        return text

    def _get_ngrams(self, text: str, n: int) -> set[tuple[str, ...]]:
        """helper:从文本中提取 N-gram"""
        words = text.split()
        if len(words) < n:
            return {tuple(words)} if words else set()
        return set(tuple(words[i : i + n]) for i in range(len(words) - n + 1))

    def _deduplication_page(
        self,
        samples: list[str],
        num_hashes: int = 128,
        num_bands: int = 16,
        ngram_size: int = 4,
        jaccard_threshold: float = 0.8,
    ) -> list[str]:
        """helper:使用 MinHash LSH 对内存中的字符串列表进行模糊去重"""
        if not samples:
            return []

        doc_ngrams = []
        # 步骤 1：提取所有文档的 n-gram
        for text in tqdm(samples, desc="去重数据中[提取 N-grams 1/4]..."):
            doc_ngrams.append(self._get_ngrams(text, ngram_size))

        # 步骤 2：生成 MinHash 签名
        PRIME = 4294967311
        random.seed(42)  # 固定种子以保证每次生成的哈希函数一致
        A = [random.randint(1, PRIME - 1) for _ in range(num_hashes)]
        B = [random.randint(0, PRIME - 1) for _ in range(num_hashes)]

        doc_signatures = []
        for ngrams in tqdm(doc_ngrams, desc="去重数据中[生成签名 2/4]..."):
            sig = [float("inf")] * num_hashes
            for ngram in ngrams:
                ngram_str = " ".join(ngram)
                x = int(hashlib.md5(ngram_str.encode("utf-8")).hexdigest()[:8], 16)
                for i in range(num_hashes):
                    h_val = (A[i] * x + B[i]) % PRIME
                    if h_val < sig[i]:
                        sig[i] = h_val
            doc_signatures.append(sig)

        # 步骤 3：使用 LSH 进行分桶并寻找候选重复项
        buckets = defaultdict(list)
        rows_per_band = num_hashes // num_bands

        for doc_id, sig in enumerate(doc_signatures):
            for b in range(num_bands):
                start = b * rows_per_band
                end = start + rows_per_band
                band_tuple = tuple(sig[start:end])
                buckets[(b, band_tuple)].append(doc_id)

        candidates = set()
        for bucket_docs in buckets.values():
            if len(bucket_docs) > 1:
                for i in range(len(bucket_docs)):
                    for j in range(i + 1, len(bucket_docs)):
                        candidates.add(tuple(sorted((bucket_docs[i], bucket_docs[j]))))

        # 步骤 4：计算真实 Jaccard 相似度，并使用并查集构建聚类
        uf_parent = {i: i for i in range(len(samples))}

        def find(i: int) -> int:
            if uf_parent[i] == i:
                return i
            uf_parent[i] = find(uf_parent[i])
            return uf_parent[i]

        def union(i: int, j: int) -> None:
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                uf_parent[root_i] = root_j

        for i, j in tqdm(candidates, desc="去重数据中[计算相似度 3/4]..."):
            set1 = doc_ngrams[i]
            set2 = doc_ngrams[j]

            if not set1 and not set2:
                sim = 1.0
            elif not set1 or not set2:
                sim = 0.0
            else:
                sim = len(set1 & set2) / len(set1 | set2)

            if sim >= jaccard_threshold:
                union(i, j)

        # 步骤 5：从每个聚类中选取保留的文档
        clusters = defaultdict(list)
        for i in range(len(samples)):
            clusters[find(i)].append(i)

        docs_to_keep_indices = set()
        for root, cluster_docs in tqdm(
            clusters.items(), desc="去重数据中[抽取保留项 4/4]..."
        ):
            # 默认保留聚类桶中的第一个样本
            docs_to_keep_indices.add(cluster_docs[0])

        # 返回去重后的新列表
        return [samples[i] for i in sorted(docs_to_keep_indices)]

    def clean(self) -> list[str]:
        raw_cleaned = map(self._clean_page, self.samples)
        self.cleaned_samples = [page for page in raw_cleaned if page is not None]
        return self._deduplication_page(self.cleaned_samples)


def process_large_file(input_file: str, output_file: str, chunk_size: int = 10000):
    """
    流式读取大文件，分块清理并保存到新文件
    :param input_file: 原始数据文件路径
    :param output_file: 清理后保存的新文件路径
    :param chunk_size: 每次加载到内存中的行数
    """

    if os.path.exists(output_file):
        os.remove(output_file)

    def clean_prefix(text: str) -> str:
        """去除开头的标签"""
        text = text.strip()
        if text.startswith("__label__low "):
            return text[len("__label__low ") :]
        elif text.startswith("__label__high "):
            return text[len("__label__high ") :]
        else:
            return text

    def process_and_save_chunk(batch_data: list[str], chunk_idx: int):
        """处理单个分块并追加写入文件"""
        if not batch_data:
            return

        cleaner = MemoryCleaner(batch_data)
        cleaned_data = cleaner.clean()

        with open(output_file, "w", encoding="utf-8") as out_f:
            for line in cleaned_data:
                clean_line = line.replace("\n", " ").replace("\r", "")
                out_f.write(f"{clean_line}\n")

    batch = []
    chunk_index = 1

    with open(input_file, "r", encoding="utf-8") as in_f:
        for line in in_f:
            if not line.strip():
                continue

            # 1. 去除标签头
            processed_line = clean_prefix(line)
            batch.append(processed_line)

            # 2. 达到分块大小时，处理并保存当前分块，然后清空列表准备下一轮
            if len(batch) >= chunk_size:
                process_and_save_chunk(batch, chunk_index)
                batch = []  # 清空列表，释放内存
                chunk_index += 1

        # 3. 处理最后不足 chunk_size 的剩余数据
        if batch:
            process_and_save_chunk(batch, chunk_index)

    print(f"处理完毕！清理后的数据已保存至: {output_file}")
