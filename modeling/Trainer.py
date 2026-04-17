import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


@dataclass
class TrainerConfig:
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
    subnet_sampling_weights: Optional[Sequence[float]] = None

    # Gradient accumulation
    grad_accum_steps: int = 4

    # Checkpointing
    checkpoint_dir: str = r"\checkpoints"
    monitor_width: Optional[int] = None


class ClassificationTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: Iterable,
        val_loader: Iterable,
        config: TrainerConfig,
    ):
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config
        self.start_epoch = 1

        assert self.cfg.grad_accum_steps >= 1

        if self.cfg.monitor_width is None:
            self.cfg.monitor_width = max(self.cfg.subnet_heads)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.cfg.lr,
            momentum=self.cfg.momentum,
            weight_decay=self.cfg.weight_decay,
        )

        self.scaler = torch.amp.GradScaler(enabled=self.cfg.use_amp)

        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        self.best_metric = -float("inf")

    def move_batch(self, batch):
        tokens = batch["tokens"].to(self.cfg.device)
        labels = batch["label_id"].to(self.cfg.device)
        targets = batch["target"].to(self.cfg.device)
        return tokens, labels, targets

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()

        total_loss = 0.0
        total_samples = 0

        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in tqdm(enumerate(self.train_loader, start=1)):
            tokens, labels, targets = self.move_batch(batch)

            # Sample one Hydra subnetwork for this mini-batch
            active_h = self.model.sample_active_heads(
                weights=self.cfg.subnet_sampling_weights
            )

            with torch.amp.autocast(enabled=self.cfg.use_amp, device_type=self.cfg.device):
                logits = self.model(tokens)
                loss = F.cross_entropy(logits, targets)
                loss_for_backward = loss / self.cfg.grad_accum_steps

            self.scaler.scale(loss_for_backward).backward()

            batch_size = labels.size(0)
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
                    f"[train] epoch={epoch} step={step} "
                    f"active_heads={active_h} "
                    f"loss={avg_loss:.4f}"
                )

        return {
            "loss": total_loss / max(total_samples, 1),
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:

        self.model.eval()
        self.model.set_active_heads(self.cfg.monitor_width)

        total_loss = 0.0
        total_samples = 0

        for batch in self.val_loader:
            tokens, labels, targets = self.move_batch(batch)

            logits = self.model(tokens)
            loss = F.cross_entropy(logits, targets)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        return {
            "loss": total_loss / max(total_samples, 1),
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
        self.best_metric = ckpt.get("best_metric", -float("inf"))
        start_epoch = ckpt["epoch"] + 1
        self.start_epoch = start_epoch

    def fit(self):
        start_epoch = 1

        for epoch in range(start_epoch, self.cfg.epochs + 1):
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.evaluate()

            monitored = val_metrics["loss"]
            is_best = monitored > self.best_metric
            if is_best:
                self.best_metric = monitored

            self.save_checkpoint(epoch, val_metrics, is_best=is_best)

            print(
                f"\n[epoch {epoch}] "
                f"train_loss={train_metrics['loss']:.4f} "
            )

            print(
                f"  [val h={self.cfg.monitor_width}] "
                f"loss={val_metrics['loss']:.4f} "
            )

            print(
                f"  best_loss@h={self.cfg.monitor_width}: "
                f"{self.best_metric:.4f}\n"
            )