import logging
from pathlib import Path
import argparse
from datasets import load_dataset
from src.tokenizer import BPETokenizer
from scripts.utils import load_config, setup_logging

logger = logging.getLogger(__name__)


def train_tokenizer(config: dict) -> None:
    train_file = Path(config["data"]["train"])
    output_path = Path(config["output"]["tokenizer_config_path"])
    vocab_size = config["tokenizer"]["vocab_size"]
    special_tokens = config["tokenizer"]["special_tokens"]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Training tokenizer on {train_file} with vocab_size = {vocab_size} ..."
    )
    tokenizer = BPETokenizer()
    tokenizer.train(
        input_path=str(train_file),
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    tokenizer.save(str(output_path))
    logger.info(f"Tokenizer saved to {output_path}")


def encoding(config: dict) -> None:
    target_txt = config["data"]["target_txt"]
    train_file_path = config["output"]["train_file_path"]
    valid_file_path = config["output"]["valid_file_path"]
    tokenizer_config_path = config["output"]["tokenizer_config_path"]

    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_config_path)

    dataset = load_dataset("text", data_files={"train": target_txt})

    def tokenize_function(example):
        return {"input_ids": tokenizer.encode(example["text"])}

    tokenized_dataset = dataset.map(
        tokenize_function,
        num_proc=8,  
        remove_columns=["text"],
    )
    split_dataset = tokenized_dataset["train"].train_test_split(test_size=0.1)

    train_dataset = split_dataset["train"]
    valid_dataset = split_dataset["test"]
    train_dataset.save_to_disk(train_file_path)
    valid_dataset.save_to_disk(valid_file_path)


def runner(
    config_path: str | None = None,
    skip_tokenizer: bool = False,
    skip_encode: bool = False,
):
    setup_logging()
    if config_path is None:
        raise ValueError("please add tokenizer_config_path as args")
    config = load_config(config_path)
    if not skip_tokenizer:
        train_tokenizer(config=config)
    if not skip_encode:
        encoding(config=config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenizer pipeline runner")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument(
        "--skip-tokenizer", action="store_true", help="Skip tokenizer train"
    )
    parser.add_argument("--skip-encode", action="store_true", help="Skip data encoding")
    args = parser.parse_args()
    runner(args.config, args.skip_tokenizer, args.skip_encode)
