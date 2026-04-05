import os
import regex as re
from tqdm import tqdm
from collections import defaultdict
import json


class BPETokenizer:
    def __init__(self):
        # 初始化基础词表: ID 0-255 对应单字节
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merges: dict[tuple[bytes, bytes], int] = {}
        self.special_tokens: dict[str, int] = {}
        # 反向映射，用于 Encode 时快速查找
        self.inverse_vocab: dict[bytes, int] = {v: k for k, v in self.vocab.items()}
        # GPT-2 的经典切词正则
        self.pat = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

    def _stream_chunks(self, input_path: str, special_tokens: list[str]):
        """流式处理输入文件，逐行读取并切分特殊标记，避免一次性加载大文件到内存"""
        if special_tokens:
            pattern = "|".join(map(re.escape, special_tokens))
            split_pat = re.compile(f"({pattern})")
        else:
            split_pat = None

        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line:
                    continue
                if split_pat:
                    parts = split_pat.split(line)
                    # 偶数索引是普通文本，奇数索引是 special_tokens
                    for i, part in enumerate(parts):
                        if i % 2 == 0 and part:
                            yield part
                else:
                    yield line

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        vocab_hex = {str(k): v.hex() for k, v in self.vocab.items()}
        merges_hex = {
            f"{k[0].hex()}|{k[1].hex()}": rank for k, rank in self.merges.items()
        }

        data = {
            "vocab": vocab_hex,
            "merges": merges_hex,
            "special_tokens": self.special_tokens,
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Tokenizer 已保存至 {save_path}")

    def load(self, load_path: str):
        """从 JSON 加载 Tokenizer 状态"""
        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab = {int(k): bytes.fromhex(v) for k, v in data["vocab"].items()}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.merges = {}
        for pair_str, rank in data["merges"].items():
            p1_hex, p2_hex = pair_str.split("|")
            self.merges[(bytes.fromhex(p1_hex), bytes.fromhex(p2_hex))] = rank

        self.special_tokens = data.get("special_tokens", {})
        print(f"Tokenizer 已成功从 {load_path} 加载")

    def train(
        self,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        max_lines: int = 1000000,
    ):
        """训练 BPE 模型"""
        current_id = max(self.vocab.keys()) + 1 if len(self.vocab) > 256 else 256
        merge_rank = (
            max(self.merges.values()) + 1
            if hasattr(self, "merges") and self.merges
            else 0
        )

        # 1. 注册特殊标记
        for token in special_tokens:
            if token not in self.special_tokens:
                self.special_tokens[token] = current_id
                self.vocab[current_id] = token.encode("utf-8")
                self.inverse_vocab[token.encode("utf-8")] = current_id
                current_id += 1

        # 2. 流式预分词并统计词频
        word_freqs = defaultdict(int)
        line_count = 0
        chunk_gen = self._stream_chunks(str(input_path), special_tokens)

        for chunk in tqdm(chunk_gen, desc="[1/2] 流式预分词计数"):
            for m in self.pat.finditer(chunk):
                word = m.group(0)
                word_bytes = tuple(bytes([b]) for b in word.encode("utf-8"))
                word_freqs[word_bytes] += 1
            line_count += 1
            if line_count > max_lines:  
                break

        # 3. BPE 迭代合并
        num_merges = vocab_size - current_id
        pbar = tqdm(total=num_merges, desc="[2/2] BPE 迭代合并")

        while len(self.vocab) < vocab_size:
            pair_counts = defaultdict(int)
            for word_tuple, freq in word_freqs.items():
                for i in range(len(word_tuple) - 1):
                    pair = (word_tuple[i], word_tuple[i + 1])
                    pair_counts[pair] += freq

            if not pair_counts:
                break

            best_pair = max(pair_counts, key=lambda x: (pair_counts[x], x))

            new_token_bytes = best_pair[0] + best_pair[1]
            self.vocab[current_id] = new_token_bytes
            self.inverse_vocab[new_token_bytes] = current_id
            self.merges[best_pair] = merge_rank

            new_word_freqs = defaultdict(int)
            for word_tuple, freq in word_freqs.items():
                if best_pair[0] in word_tuple and best_pair[1] in word_tuple:
                    new_tuple = []
                    i = 0
                    while i < len(word_tuple):
                        if (
                            i < len(word_tuple) - 1
                            and word_tuple[i] == best_pair[0]
                            and word_tuple[i + 1] == best_pair[1]
                        ):
                            new_tuple.append(new_token_bytes)
                            i += 2
                        else:
                            new_tuple.append(word_tuple[i])
                            i += 1
                    new_word_freqs[tuple(new_tuple)] += freq
                else:
                    new_word_freqs[word_tuple] += freq

            word_freqs = new_word_freqs
            current_id += 1
            merge_rank += 1
            pbar.update(1)

        pbar.close()

    def encode(self, text: str) -> list[int]:
        """将字符串编码为 Token IDs"""
        # 1. 处理特殊字符
        if self.special_tokens:
            pattern = "|".join(map(re.escape, self.special_tokens.keys()))
            parts = re.split(f"({pattern})", text)
        else:
            parts = [text]

        ids = []
        for part in parts:
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            else:
                # 2. 对普通文本进行正则预分词
                for m in self.pat.finditer(part):
                    word = m.group(0)
                    word_bytes = [bytes([b]) for b in word.encode("utf-8")]
                    # 3. 按照 self.merges 的优先级应用合并规则
                    while len(word_bytes) > 1:
                        min_rank = float("inf")
                        best_pair_idx = -1

                        for i in range(len(word_bytes) - 1):
                            pair = (word_bytes[i], word_bytes[i + 1])
                            if pair in self.merges and self.merges[pair] < min_rank:
                                min_rank = self.merges[pair]
                                best_pair_idx = i

                        # 如果找不到能合并的 pair，说明合并结束
                        if best_pair_idx == -1:
                            break
                        word_bytes[best_pair_idx] = (
                            word_bytes[best_pair_idx] + word_bytes[best_pair_idx + 1]
                        )
                        del word_bytes[best_pair_idx + 1]
                    # 4. 查表转为 IDs
                    ids.extend([self.inverse_vocab[b] for b in word_bytes])
        return ids

    def decode(self, ids: list[int]) -> str:
        """将 Token IDs 解码回字符串"""
        b_text = b"".join(self.vocab[idx] for idx in ids)
        # 用 replace 忽略无法解码的非法字节
        return b_text.decode("utf-8", errors="replace")
