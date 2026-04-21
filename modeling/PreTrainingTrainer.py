import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


@dataclass
class PretrainingTrainerConfig:
    epochs: int = 50
    lr: float = 1e-2
    weight_decay: float = 1e-4
    momentum: float = 0.9

    grad_clip_norm: float = 1.0
    use_amp: bool = True
    log_every: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Hydra-specific
    subnet_heads: Sequence[int] = (1, 2, 4, 8)

    # Gradient accumulation
    grad_accum_steps: int = 4

    # Checkpointing
    checkpoint_dir: str = r"\checkpoints"

    # Autoregressive validation
    ar_prefix_len: int = 100  # number of initial ground-truth tokens to condition on
    max_ar_eval_batches: Optional[int] = None  # optionally limit AR eval for speed


class PretrainingTrainer:
    def __init__(
            self,
            model: nn.Module,
            train_loader: Iterable,
            val_loader: Iterable,
            config: PretrainingTrainerConfig,
    ):
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config
        self.start_epoch = 1

        assert self.cfg.grad_accum_steps >= 1

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.cfg.lr,
            momentum=self.cfg.momentum,
            weight_decay=self.cfg.weight_decay,
        )

        self.scaler = torch.amp.GradScaler(enabled=self.cfg.use_amp)

        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        self.best_metric = float("inf")  # lower loss is better

    def move_batch(self, batch):
        return batch["tokens"].to(self.cfg.device)

    def pretrain_loss(self, tokens: torch.Tensor):
        lm_logits, lm_targets = self.model(tokens, pretrain=True)
        loss = F.cross_entropy(
            lm_logits.reshape(-1, lm_logits.size(-1)),
            lm_targets.reshape(-1),
            ignore_index=int(self.model.pad_id),
        )
        return loss, lm_logits, lm_targets

    @torch.no_grad()
    def autoregressive_generate_one(
            self,
            full_tokens: torch.Tensor,
            prefix_len: int,
    ) -> Dict[str, float]:
        pad_id = self.model.pad_id

        valid_len = int((full_tokens != pad_id).sum().item())
        if valid_len <= 1:
            return {"correct": 0.0, "count": 0.0}

        prefix_len = max(1, min(prefix_len, valid_len - 1))

        # Ground-truth non-pad sequence only
        gt = full_tokens[:valid_len]  # [L]
        generated = gt[:prefix_len].clone().unsqueeze(0)  # [1, prefix_len]

        correct = 0
        count = 0

        while generated.size(1) < valid_len:
            lm_logits, _ = self.model(generated, pretrain=True)
            next_token_logits = lm_logits[:, -1, :]  # predict next token
            next_token = next_token_logits.argmax(dim=-1)  # [1]

            target_next = gt[generated.size(1)].view(1)
            correct += (next_token == target_next.to(next_token.device)).sum().item()
            count += 1

            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

        return {"correct": float(correct), "count": float(count)}

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()

        total_loss = 0.0
        total_samples = 0

        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in tqdm(enumerate(self.train_loader, start=1), total=len(self.train_loader)):
            tokens = self.move_batch(batch)

            self.model.set_active_heads(max(self.cfg.subnet_heads))

            with torch.amp.autocast(
                    enabled=self.cfg.use_amp,
                    device_type=self.cfg.device,
            ):
                loss, _, _ = self.pretrain_loss(tokens)
                loss_for_backward = loss / self.cfg.grad_accum_steps

            self.scaler.scale(loss_for_backward).backward()

            batch_size = tokens.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            should_step = (
                    step % self.cfg.grad_accum_steps == 0
                    or step == len(self.train_loader)
            )

            if should_step:
                if self.cfg.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.grad_clip_norm,
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            if step % self.cfg.log_every == 0:
                avg_loss = total_loss / max(total_samples, 1)
                print(
                    f"[train-pre] epoch={epoch} step={step} "
                    f"active_heads={max(self.cfg.subnet_heads)}, "
                    f"loss={avg_loss:.4f}"
                )

        return {
            "loss": total_loss / max(total_samples, 1),
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        self.model.set_active_heads(max(self.cfg.subnet_heads))

        total_loss = 0.0
        total_samples = 0

        ar_correct = 0.0
        ar_count = 0.0

        if self.cfg.max_ar_eval_batches is None:
            max_len = 2**32-1
        else:
            max_len = self.cfg.max_ar_eval_batches

        print("Start Validation")
        for batch_idx, batch in tqdm(enumerate(self.val_loader, start=1), total=min(max_len, len(self.val_loader))):
            if self.cfg.max_ar_eval_batches is not None and batch_idx >= self.cfg.max_ar_eval_batches:
                break

            tokens = self.move_batch(batch)

            # Teacher-forced validation loss
            with torch.amp.autocast(
                    enabled=self.cfg.use_amp,
                    device_type=self.cfg.device,
            ):
                loss, _, _ = self.pretrain_loss(tokens)

            batch_size = tokens.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Autoregressive validation
            for i in range(tokens.size(0)):
                ar_stats = self.autoregressive_generate_one(
                    tokens[i],
                    prefix_len=self.cfg.ar_prefix_len,
                )
                ar_correct += ar_stats["correct"]
                ar_count += ar_stats["count"]

        return {
            "loss": total_loss / max(total_samples, 1),
            "ar_token_acc": ar_correct / max(ar_count, 1.0),
        }

    def checkpoint_state(self, epoch: int, val_metrics: Dict[str, float]):
        return {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_metric": self.best_metric,
            "val_metrics": val_metrics,
            "config": vars(self.cfg),
        }

    def save_checkpoint(
            self,
            epoch: int,
            val_metrics: Dict[str, float],
            is_best: bool,
    ):
        state = self.checkpoint_state(epoch, val_metrics)

        last_path = os.path.join(self.cfg.checkpoint_dir, "last.pt")
        torch.save(state, last_path)

        if is_best:
            best_path = os.path.join(self.cfg.checkpoint_dir, "best.pt")
            torch.save(state, best_path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.cfg.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.best_metric = ckpt.get("best_metric", float("inf"))
        self.start_epoch = ckpt["epoch"] + 1

    def fit(self):
        start_epoch = self.start_epoch

        for epoch in range(start_epoch, self.cfg.epochs + 1):
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.evaluate()

            monitored = val_metrics["loss"]
            is_best = monitored < self.best_metric
            if is_best:
                self.best_metric = monitored

            self.save_checkpoint(epoch, val_metrics, is_best=is_best)

            print(
                f"\n[epoch {epoch}] "
                f"train_loss={train_metrics['loss']:.4f}"
            )

            print(
                f"  [val h={max(self.cfg.subnet_heads)}] "
                f"loss={val_metrics['loss']:.4f} "
                f"ar_token_acc={val_metrics['ar_token_acc']:.4f}"
            )

            print(
                f"  best_loss@h={max(self.cfg.subnet_heads)}: "
                f"{self.best_metric:.4f}\n"
            )
