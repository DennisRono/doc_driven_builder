import time
import math
import torch
import logging
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from evaluation import ModelEvaluator
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from torch.cuda.amp import autocast
from torch.amp import GradScaler
from model import DocumentationModel, ModelConfig

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for training."""
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    use_mixed_precision: bool = True
    accumulation_steps: int = 1
    save_every: int = 5
    eval_every: int = 1
    early_stopping_patience: int = 5
    min_delta: float = 1e-4
    scheduler_type: str = "cosine"
    optimizer_type: str = "adamw"


class LearningRateScheduler:
    """Custom learning rate scheduler with warmup."""

    def __init__(
        self, optimizer: optim.Optimizer, config: TrainingConfig, total_steps: int
    ):
        self.optimizer = optimizer
        self.config = config
        self.total_steps = total_steps
        self.current_step = 0
        self.base_lr = config.learning_rate
        self.warmup_steps = min(config.warmup_steps, max(total_steps // 10, 1))

    def step(self):
        """Update learning rate."""
        self.current_step += 1
        lr = self._get_lr()

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _get_lr(self) -> float:
        """Calculate current learning rate - returns float, not tensor."""
        if self.current_step < self.warmup_steps:
            return self.base_lr * (self.current_step / self.warmup_steps)

        progress = (self.current_step - self.warmup_steps) / (
            self.total_steps - self.warmup_steps
        )
        progress = min(progress, 1.0)

        if self.config.scheduler_type == "cosine":
            return self.base_lr * 0.5 * (1 + math.cos(progress * math.pi))
        elif self.config.scheduler_type == "linear":
            return self.base_lr * (1 - progress)
        else:
            return self.base_lr


class ModelCheckpoint:
    """Enhanced model checkpointing with metadata."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric = float("inf")
        self.best_checkpoint_path = None

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: LearningRateScheduler,
        scaler: Optional[GradScaler],
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        model_config: ModelConfig,
        training_config: TrainingConfig,
        is_best: bool = False,
    ) -> str:
        """Save model checkpoint with full state."""
        checkpoint_data = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_step": scheduler.current_step,
            "metrics": metrics,
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "timestamp": time.time(),
        }

        if scaler is not None:
            checkpoint_data["scaler_state_dict"] = scaler.state_dict()

        checkpoint_path = (
            self.checkpoint_dir / f"checkpoint_epoch_{epoch}_step_{step}.pt"
        )
        torch.save(checkpoint_data, checkpoint_path)

        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint_data, best_path)
            self.best_checkpoint_path = best_path
            logger.info(f"New best checkpoint saved: {best_path}")

        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        torch.save(checkpoint_data, latest_path)

        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[LearningRateScheduler] = None,
        scaler: Optional[GradScaler] = None,
    ) -> Dict:
        """Load model checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")

        model.load_state_dict(checkpoint_data["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

        if scheduler is not None and "scheduler_step" in checkpoint_data:
            scheduler.current_step = checkpoint_data["scheduler_step"]

        if scaler is not None and "scaler_state_dict" in checkpoint_data:
            scaler.load_state_dict(checkpoint_data["scaler_state_dict"])

        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint_data


class Trainer:
    """Enhanced trainer with modern training techniques."""

    def __init__(
        self,
        model: DocumentationModel,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model
        self.model_config = model_config
        self.config = training_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        self.optimizer = self._create_optimizer()
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.scaler = GradScaler() if training_config.use_mixed_precision else None
        self.checkpoint_manager = ModelCheckpoint(checkpoint_dir)
        self.evaluator = ModelEvaluator()

        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        logger.info(f"Trainer initialized on {self.device}")
        logger.info(f"Model parameters: {model.get_num_trainable_params():,}")

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        if self.config.optimizer_type == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.95),
                eps=1e-8,
            )
        elif self.config.optimizer_type == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9,
            )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
    ) -> Dict[str, List[float]]:
        """Main training loop with comprehensive logging and checkpointing."""
        total_steps = len(train_loader) * self.config.epochs
        scheduler = LearningRateScheduler(self.optimizer, self.config, total_steps)

        history = {
            "train_loss": [],
            "val_loss": [],
            "val_perplexity": [],
            "learning_rate": [],
        }

        logger.info(f"Starting training for {self.config.epochs} epochs")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Warmup steps: {scheduler.warmup_steps}")

        try:
            for epoch in range(self.current_epoch, self.config.epochs):
                self.current_epoch = epoch

                train_metrics = self._train_epoch(train_loader, scheduler)
                history["train_loss"].append(train_metrics["loss"])
                
                current_lr = self.optimizer.param_groups[0]["lr"]
                history["learning_rate"].append(current_lr)

                if epoch % self.config.eval_every == 0:
                    val_metrics = self._validate_epoch(val_loader)
                    history["val_loss"].append(val_metrics["loss"])
                    history["val_perplexity"].append(val_metrics["perplexity"])

                    is_best = val_metrics["loss"] < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_metrics["loss"]
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1

                    if epoch % self.config.save_every == 0 or is_best:
                        self.checkpoint_manager.save_checkpoint(
                            self.model,
                            self.optimizer,
                            scheduler,
                            self.scaler,
                            epoch,
                            self.current_step,
                            val_metrics,
                            self.model_config,
                            self.config,
                            is_best,
                        )

                    if self.patience_counter >= self.config.early_stopping_patience:
                        logger.info(
                            f"Early stopping triggered after {epoch + 1} epochs"
                        )
                        break

                    logger.info(
                        f"Epoch {epoch + 1}/{self.config.epochs} - "
                        f"Train Loss: {train_metrics['loss']:.4f}, "
                        f"Val Loss: {val_metrics['loss']:.4f}, "
                        f"Val Perplexity: {val_metrics['perplexity']:.2f}, "
                        f"LR: {current_lr:.2e}"
                    )

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

        if test_loader is not None:
            logger.info("Running final evaluation on test set...")
            test_metrics = self._validate_epoch(test_loader)
            logger.info(
                f"Test Loss: {test_metrics['loss']:.4f}, "
                f"Test Perplexity: {test_metrics['perplexity']:.2f}"
            )

        return history

    def _train_epoch(
        self, train_loader: DataLoader, scheduler: LearningRateScheduler
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(self.device)
            target_ids = batch["target_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            if self.config.use_mixed_precision:
                with autocast():
                    outputs = self.model(input_ids, attention_mask)
                    loss = self.criterion(
                        outputs.reshape(-1, outputs.size(-1)),
                        target_ids.reshape(-1),
                    )
                    loss = loss / self.config.accumulation_steps
            else:
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(
                    outputs.reshape(-1, outputs.size(-1)), target_ids.reshape(-1)
                )
                loss = loss / self.config.accumulation_steps

            if self.config.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                if self.config.use_mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad()
                scheduler.step()
                self.current_step += 1

            total_loss += loss.item() * self.config.accumulation_steps
            num_batches += 1

        return {"loss": total_loss / max(num_batches, 1)}

    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(
                    outputs.reshape(-1, outputs.size(-1)), target_ids.reshape(-1)
                )

                non_pad_tokens = (target_ids != 0).sum().item()

                total_loss += loss.item()
                total_tokens += non_pad_tokens
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "total_tokens": total_tokens,
        }

    def resume_from_checkpoint(self, checkpoint_path: str) -> Dict:
        """Resume training from checkpoint."""
        checkpoint_data = self.checkpoint_manager.load_checkpoint(
            checkpoint_path, self.model, self.optimizer, scaler=self.scaler
        )

        self.current_epoch = checkpoint_data["epoch"] + 1
        self.current_step = checkpoint_data["step"]
        self.best_val_loss = checkpoint_data["metrics"].get("loss", float("inf"))

        return checkpoint_data

    def debug_single_batch(self, train_loader: DataLoader) -> None:
        """Debug helper - run one batch and check gradients/parameter changes."""
        logger.info("=== DEBUG: Running single batch ===")
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # Get first batch
        batch = next(iter(train_loader))
        input_ids = batch["input_ids"].to(self.device)
        target_ids = batch["target_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        
        # Store initial parameters
        initial_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Forward pass
        outputs = self.model(input_ids, attention_mask)
        loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), target_ids.reshape(-1))
        
        logger.info(f"Initial loss: {loss.item():.6f}")
        logger.info(f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.parameters())
        logger.info(f"Has gradients: {has_grads}")
        
        if has_grads:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))
            logger.info(f"Gradient norm: {grad_norm:.6f}")
        
        # Optimizer step
        self.optimizer.step()
        
        # Check parameter changes
        param_changes = []
        for name, param in self.model.named_parameters():
            if name in initial_params:
                change = (param - initial_params[name]).abs().sum().item()
                param_changes.append((name, change))
        
        # Show top 5 parameter changes
        param_changes.sort(key=lambda x: x[1], reverse=True)
        logger.info("Top parameter changes:")
        for name, change in param_changes[:5]:
            logger.info(f"  {name}: {change:.8f}")
        
        logger.info("=== DEBUG: Single batch complete ===")
