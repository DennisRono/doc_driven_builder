import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class TrainingDiagnostics:
    """Comprehensive training diagnostics and analysis."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    def overfitting_test(
        self, 
        train_loader: DataLoader, 
        num_samples: int = 10,
        max_epochs: int = 50
    ) -> Dict[str, List[float]]:
        """Test if model can overfit on small subset."""
        logger.info(f"Running overfitting test on {num_samples} samples...")
        
        # Get small subset
        subset_data = []
        for i, batch in enumerate(train_loader):
            if i >= num_samples:
                break
            subset_data.append(batch)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=0.01
        )
        
        losses = []
        accuracies = []
        
        self.model.train()
        
        for epoch in range(max_epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            total_tokens = 0
            
            for batch in subset_data:
                input_ids = batch["input_ids"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(
                    outputs.reshape(-1, outputs.size(-1)),
                    target_ids.reshape(-1)
                )
                
                loss.backward()
                optimizer.step()
                
                # Calculate accuracy
                predictions = torch.argmax(outputs, dim=-1)
                mask = target_ids != 0
                correct = (predictions == target_ids) & mask
                accuracy = correct.sum().float() / mask.sum().float()
                
                epoch_loss += loss.item()
                epoch_acc += accuracy.item()
                total_tokens += mask.sum().item()
            
            avg_loss = epoch_loss / len(subset_data)
            avg_acc = epoch_acc / len(subset_data)
            
            losses.append(avg_loss)
            accuracies.append(avg_acc)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss {avg_loss:.4f}, Acc {avg_acc:.4f}")
            
            # Early stopping if overfitting achieved
            if avg_loss < 0.1 and avg_acc > 0.9:
                logger.info(f"Overfitting achieved at epoch {epoch}")
                break
        
        return {"losses": losses, "accuracies": accuracies}
    
    def gradient_analysis(self, train_loader: DataLoader) -> Dict[str, float]:
        """Analyze gradient flow and magnitudes."""
        logger.info("Analyzing gradient flow...")
        
        self.model.train()
        
        # Get one batch
        batch = next(iter(train_loader))
        input_ids = batch["input_ids"].to(self.device)
        target_ids = batch["target_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        
        # Forward pass
        outputs = self.model(input_ids, attention_mask)
        loss = self.criterion(
            outputs.reshape(-1, outputs.size(-1)),
            target_ids.reshape(-1)
        )
        
        # Backward pass
        loss.backward()
        
        # Analyze gradients
        grad_stats = {}
        total_norm = 0.0
        param_count = 0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                grad_stats[name] = {
                    "norm": grad_norm,
                    "mean": param.grad.data.mean().item(),
                    "std": param.grad.data.std().item(),
                    "max": param.grad.data.max().item(),
                    "min": param.grad.data.min().item(),
                }
                total_norm += grad_norm ** 2
                param_count += 1
        
        total_norm = total_norm ** 0.5
        
        # Summary statistics
        summary = {
            "total_gradient_norm": total_norm,
            "avg_gradient_norm": total_norm / max(param_count, 1),
            "parameters_with_gradients": param_count,
            "loss": loss.item(),
        }
        
        logger.info(f"Total gradient norm: {total_norm:.6f}")
        logger.info(f"Average gradient norm: {summary['avg_gradient_norm']:.6f}")
        
        return {"summary": summary, "per_parameter": grad_stats}
    
    def parameter_change_analysis(
        self, 
        train_loader: DataLoader, 
        num_steps: int = 10
    ) -> Dict[str, List[float]]:
        """Analyze parameter changes over training steps."""
        logger.info(f"Analyzing parameter changes over {num_steps} steps...")
        
        # Store initial parameters
        initial_params = {}
        for name, param in self.model.named_parameters():
            initial_params[name] = param.data.clone()
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
        
        param_changes = {name: [] for name in initial_params.keys()}
        
        self.model.train()
        data_iter = iter(train_loader)
        
        for step in range(num_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            
            input_ids = batch["input_ids"].to(self.device)
            target_ids = batch["target_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(
                outputs.reshape(-1, outputs.size(-1)),
                target_ids.reshape(-1)
            )
            
            loss.backward()
            optimizer.step()
            
            # Calculate parameter changes
            for name, param in self.model.named_parameters():
                if name in initial_params:
                    change = (param.data - initial_params[name]).abs().mean().item()
                    param_changes[name].append(change)
        
        return param_changes
    
    def plot_training_curves(
        self, 
        history: Dict[str, List[float]], 
        save_dir: str = "plots"
    ):
        """Plot comprehensive training curves."""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Loss curves
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(history["train_loss"], label="Train Loss")
        if "val_loss" in history:
            plt.plot(history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        
        # Perplexity
        plt.subplot(2, 2, 2)
        if "val_perplexity" in history:
            plt.plot(history["val_perplexity"], label="Val Perplexity")
            plt.xlabel("Epoch")
            plt.ylabel("Perplexity")
            plt.title("Validation Perplexity")
            plt.legend()
            plt.grid(True)
        
        # Learning rate
        plt.subplot(2, 2, 3)
        plt.plot(history["learning_rate"], label="Learning Rate")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.yscale("log")
        plt.legend()
        plt.grid(True)
        
        # Step-wise losses if available
        plt.subplot(2, 2, 4)
        if "step_losses" in history:
            plt.plot(history["step_losses"], alpha=0.7)
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Step-wise Training Loss")
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_dir / "training_curves.png", dpi=300, bbox_inches="tight")
        plt.show()
        
        logger.info(f"Training curves saved to {save_dir / 'training_curves.png'}")
    
    def compute_bits_per_token(self, loss: float) -> float:
        """Convert loss to bits per token."""
        return loss / math.log(2)
    
    def training_summary(self, history: Dict[str, List[float]]) -> Dict[str, float]:
        """Generate comprehensive training summary."""
        summary = {}
        
        if "train_loss" in history and history["train_loss"]:
            summary["final_train_loss"] = history["train_loss"][-1]
            summary["best_train_loss"] = min(history["train_loss"])
            summary["train_loss_reduction"] = (
                history["train_loss"][0] - history["train_loss"][-1]
            ) / history["train_loss"][0]
        
        if "val_loss" in history and history["val_loss"]:
            summary["final_val_loss"] = history["val_loss"][-1]
            summary["best_val_loss"] = min(history["val_loss"])
            summary["val_loss_reduction"] = (
                history["val_loss"][0] - history["val_loss"][-1]
            ) / history["val_loss"][0]
        
        if "val_perplexity" in history and history["val_perplexity"]:
            summary["final_perplexity"] = history["val_perplexity"][-1]
            summary["best_perplexity"] = min(history["val_perplexity"])
            summary["final_bits_per_token"] = self.compute_bits_per_token(
                history["val_loss"][-1] if "val_loss" in history else 0
            )
        
        return summary
