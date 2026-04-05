import os
import math
import logging
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from datasets import load_from_disk
from torch.utils.data import DataLoader
from src.tokenizer import BPETokenizer
from src.transformer import TransformerLM
from scripts.utils import load_config, setup_logging
from scripts.train.runner import HFTextDataset

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, config_path: str, checkpoint_path: str):
        self.config = load_config(config_path)
        self.device = torch.device(
            self.config.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Using device: {self.device}")

        tokenizer_path = self.config["output"]["tokenizer_config_path"]
        self.tokenizer = BPETokenizer()
        self.tokenizer.load(tokenizer_path)
        logger.info(f"Tokenizer loaded from {tokenizer_path}")

        self.context_length = self.config["model"]["context_length"]
        self.vocab_size = self.config["model"]["vocab_size"]

        self.model = TransformerLM(
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            d_model=self.config["model"]["d_model"],
            num_layers=self.config["model"]["num_layers"],
            num_heads=self.config["model"]["num_heads"],
            d_ff=self.config["model"]["d_ff"],
            theta=self.config["model"]["theta"],
        ).to(self.device)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval() 
        logger.info(
            f"Model weights loaded from {checkpoint_path} (Epoch: {checkpoint.get('epoch', 'N/A')})"
        )

    # ----------------------------------------
    # 文本生成
    # ----------------------------------------
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> str:
        input_ids = self.tokenizer.encode(prompt)
        x = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        generated_ids = []

        for _ in range(max_new_tokens):
            x_cond = x[:, -self.context_length :]

            logits = self.model(x_cond)
            next_token_logits = logits[0, -1, :] / max(temperature, 1e-5)

            if top_k > 0:
                v, _ = torch.topk(
                    next_token_logits, min(top_k, next_token_logits.size(-1))
                )
                next_token_logits[next_token_logits < v[-1]] = -float("Inf")
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token.unsqueeze(0)), dim=1)
            generated_ids.append(next_token.item())

        return self.tokenizer.decode(input_ids + generated_ids)

    # ----------------------------------------
    # 验证集困惑度 (PPL) 测试
    # ----------------------------------------
    @torch.no_grad()
    def calculate_ppl(self):
        valid_data_path = self.config["data"]["valid"]
        logger.info(f"Loading valid dataset for PPL from {valid_data_path}...")

        hf_valid_dataset = load_from_disk(valid_data_path)
        hf_valid_dataset.set_format(type="torch", columns=["input_ids"])
        valid_dataset = HFTextDataset(
            hf_valid_dataset, context_length=self.context_length
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.config["training"]["batch_size"] * 2,
            shuffle=False,
        )

        criterion = nn.CrossEntropyLoss(ignore_index=0)
        total_loss = 0.0

        pbar = tqdm(valid_loader, desc="Calculating PPL")
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = criterion(logits.reshape(-1, self.vocab_size), y.reshape(-1))
            total_loss += loss.item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(valid_loader)
        ppl = math.exp(avg_loss)
        logger.info(f"✨ Validation Loss: {avg_loss:.4f} | Perplexity (PPL): {ppl:.4f}")
        return ppl

    # ----------------------------------------
    # 多轮对话 (Chat)
    # ----------------------------------------
    def chat(self, max_new_tokens: int = 100):
        print("\n" + "=" * 50)
        print("🤖 Model Chat Activated! (Type 'quit' or 'exit' to stop)")
        print("=" * 50 + "\n")

        context = ""
        while True:
            try:
                user_input = input("🧑 You: ")
                if user_input.lower() in ["quit", "exit"]:
                    print("👋 Bye!")
                    break
                if not user_input.strip():
                    continue

                context += f"\nUser: {user_input}\nBot:"

                response_full = self.generate(
                    context, max_new_tokens=max_new_tokens, temperature=0.7
                )

                new_text = response_full[len(context) :]
                print(f"🤖 Bot: {new_text.strip()}\n")

                context += f" {new_text.strip()}"

                if (
                    len(self.tokenizer.encode(context))
                    > self.context_length - max_new_tokens
                ):
                    print("[System: Context limit reached, clearing history...]")
                    context = ""

            except KeyboardInterrupt:
                print("\n👋 Bye!")
                break


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Eval Pipeline for TransformerLM")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)"
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["generate", "ppl", "chat"],
        help="Choose task: 'generate' (single prompt), 'ppl' (validate dataset), or 'chat' (interactive)",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, how are you?",
        help="Prompt for text generation",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=100, help="Max new tokens to generate"
    )

    args = parser.parse_args()

    evaluator = Evaluator(config_path=args.config, checkpoint_path=args.checkpoint)

    if args.task == "generate":
        print(f"\n--- Prompt ---\n{args.prompt}")
        output = evaluator.generate(args.prompt, max_new_tokens=args.max_tokens)
        print(f"\n--- Output ---\n{output}\n")

    elif args.task == "ppl":
        evaluator.calculate_ppl()

    elif args.task == "chat":
        evaluator.chat(max_new_tokens=args.max_tokens)
