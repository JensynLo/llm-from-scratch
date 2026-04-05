import wandb
import os
import logging
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datasets import load_from_disk
from src.transformer import TransformerLM
from scripts.utils import load_config, setup_logging

logger = logging.getLogger(__name__)


class HFTextDataset(Dataset):
    def __init__(self, hf_dataset, context_length, pad_token_id=0):
        self.hf_dataset = hf_dataset
        self.context_length = context_length
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        input_ids = self.hf_dataset[idx]["input_ids"].long()
        target_len = self.context_length + 1

        if len(input_ids) >= target_len:
            input_ids = input_ids[:target_len]
        else:
            padding_length = target_len - len(input_ids)
            padding = torch.full((padding_length,), self.pad_token_id, dtype=torch.long)
            input_ids = torch.cat([input_ids, padding])
        x = input_ids[:-1]
        y = input_ids[1:]

        return x, y


def save_checkpoint(
    model, optimizer, epoch, step, loss, checkpoint_dir: str, step_name: bool = False
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    if step_name:
        filepath = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
    else:
        filepath = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")

    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, filepath)
    logger.info(
        f"Checkpoint saved: {filepath} (epoch={epoch}, step={step}, loss={loss:.4f})"
    )


def load_checkpoint(model, optimizer, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    logger.info(f"Checkpoint loaded from {checkpoint_path}")
    return checkpoint["epoch"], checkpoint["step"]


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, vocab_size):
    model.eval()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
        total_loss += loss.item()

    model.train()
    return total_loss / len(dataloader) if len(dataloader) > 0 else 0.0


def train(config_path: str, resume_from: str | None = None):
    setup_logging()
    config = load_config(config_path)

    # wandb
    wandb_cfg = config["wandb"]
    wandb.init(
        project=wandb_cfg["project"],
        name=wandb_cfg["run_name"],
        config=config,
        resume="allow" if resume_from else None,
    )

    # 1. 设备设置
    device_config = config.get("device", "cuda")
    device = torch.device(device_config if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 2. 参数解析
    train_data_path = config["data"]["train"]
    valid_data_path = config["data"]["valid"]
    context_length = config["model"]["context_length"]
    vocab_size = config["model"]["vocab_size"]
    batch_size = config["training"]["batch_size"]

    # 3. 加载 Hugging Face 数据集并使用包装器
    logger.info(f"Loading HF dataset from {train_data_path}...")
    hf_train_dataset = load_from_disk(train_data_path)
    hf_train_dataset.set_format(type="torch", columns=["input_ids"])

    train_dataset = HFTextDataset(
        hf_train_dataset, context_length=context_length, pad_token_id=0
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    logger.info(f"Loading HF valid dataset from {valid_data_path}...")
    hf_valid_dataset = load_from_disk(valid_data_path)
    hf_valid_dataset.set_format(type="torch", columns=["input_ids"])
    valid_dataset = HFTextDataset(
        hf_valid_dataset, context_length=context_length, pad_token_id=0
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
    )

    logger.info(f"Training dataset loaded: {len(train_dataset)} samples")
    logger.info(f"Validation dataset loaded: {len(valid_dataset)} samples")

    # 4. 初始化模型
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=config["model"]["d_model"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        d_ff=config["model"]["d_ff"],
        theta=config["model"]["theta"],
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model initialized with {num_params:,} parameters")

    # 5. 优化器与调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["lr_decay_steps"],
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    wandb.watch(model, criterion, log="all", log_freq=100)

    # 6. 恢复断点
    start_epoch = 0
    global_step = 0

    if resume_from is not None and os.path.exists(resume_from):
        start_epoch, global_step = load_checkpoint(model, optimizer, resume_from)
        logger.info(f"Resumed from epoch {start_epoch}, step {global_step}")

    save_dir = config["checkpoint"]["save_dir"]
    save_interval = config["checkpoint"]["save_interval"]
    os.makedirs(save_dir, exist_ok=True) 

    # 跟踪最佳验证集 Loss
    best_val_loss = float("inf")

    # 7. 训练循环
    model.train()
    num_epochs = config["training"]["num_epochs"]

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        avg_loss = 0

        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["training"]["gradient_clip"]
            )

            optimizer.step()

            if lr_scheduler:
                lr_scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            avg_loss = epoch_loss / (batch_idx + 1)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

            if global_step % save_interval == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    global_step,
                    avg_loss,
                    save_dir,
                    step_name=True,
                )

            wandb.log(
                {
                    "train/step_loss": loss.item(),
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "global_step": global_step,
                }
            )

        # Epoch 结束后的验证与记录
        val_loss = evaluate(model, valid_loader, criterion, device, vocab_size)

        wandb.log(
            {
                "train/epoch_loss": avg_loss,
                "epoch": epoch,
                "val/epoch_loss": val_loss,
            }
        )

        save_checkpoint(
            model,
            optimizer,
            epoch,
            global_step,
            avg_loss,
            save_dir,
        )

        # 判断并保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(save_dir, "checkpoint_best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_val_loss,
                },
                best_model_path,
            )

            logger.info(
                f"✨ New best model saved! Val loss improved to: {best_val_loss:.4f}"
            )

        logger.info(
            f"Epoch {epoch} completed, Train avg loss: {avg_loss:.4f}, Val loss: {val_loss:.4f}"
        )

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training pipeline runner")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()
    train(args.config, args.resume)
