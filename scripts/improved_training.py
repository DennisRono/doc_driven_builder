import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
import logging
import math
from dataclasses import dataclass
from training import Trainer, TrainingConfig, LearningRateScheduler
from model import DocumentationModel, ModelConfig
from dataset import MultiFormatDataProcessor, Tokenizer, DataConfig, create_data_loaders

logger = logging.getLogger(__name__)

@dataclass
class ImprovedTrainingConfig(TrainingConfig):
    """Enhanced training config with research-backed improvements."""
    # LR scheduling improvements
    lr_floor_ratio: float = 1e-3  # LR floor as ratio of base LR
    extend_training_ratio: float = 1.2  # Extend training by 20%
    
    # Optimizer improvements  
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    
    # Regularization
    label_smoothing: float = 0.0
    ema_decay: float = 0.999
    use_ema: bool = True
    
    # Evaluation
    eval_steps: int = 500  # More frequent evaluation
    log_steps: int = 100   # More frequent logging

class EMAModel:
    """Exponential Moving Average of model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self):
        """Apply EMA parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

class ImprovedTrainer(Trainer):
    """Enhanced trainer with research-backed improvements."""
    
    def __init__(
        self,
        model: DocumentationModel,
        model_config: ModelConfig,
        training_config: ImprovedTrainingConfig,
        checkpoint_dir: str = "checkpoints",
    ):
        super().__init__(model, model_config, training_config, checkpoint_dir)
        
        # Enhanced criterion with label smoothing
        if training_config.label_smoothing > 0:
            self.criterion = LabelSmoothingCrossEntropy(
                smoothing=training_config.label_smoothing,
                ignore_index=0
            )
        
        # EMA model
        if training_config.use_ema:
            self.ema_model = EMAModel(model, training_config.ema_decay)
        else:
            self.ema_model = None
            
        # Enhanced optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            betas=(training_config.beta1, training_config.beta2),
            eps=training_config.eps,
        )
        
        logger.info("Enhanced trainer initialized with research-backed improvements")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
    ) -> Dict[str, List[float]]:
        """Enhanced training loop with frequent evaluation and EMA."""
        # Extend training steps
        extended_epochs = int(self.config.epochs * self.config.extend_training_ratio)
        total_steps = len(train_loader) * extended_epochs
        
        scheduler = EnhancedLRScheduler(self.optimizer, self.config, total_steps)
        
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_perplexity": [],
            "learning_rate": [],
            "step_losses": [],  # More granular loss tracking
            "step_lrs": [],
        }
        
        logger.info(f"Starting enhanced training for {extended_epochs} epochs")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"LR floor: {scheduler.lr_floor:.2e}")
        
        global_step = 0
        
        try:
            for epoch in range(self.current_epoch, extended_epochs):
                self.current_epoch = epoch
                
                train_metrics = self._enhanced_train_epoch(
                    train_loader, scheduler, history, global_step
                )
                history["train_loss"].append(train_metrics["loss"])
                global_step = train_metrics["global_step"]
                
                current_lr = self.optimizer.param_groups[0]["lr"]
                history["learning_rate"].append(current_lr)
                
                # More frequent validation
                if epoch % self.config.eval_every == 0:
                    val_metrics = self._enhanced_validate_epoch(val_loader)
                    history["val_loss"].append(val_metrics["loss"])
                    history["val_perplexity"].append(val_metrics["perplexity"])
                    
                    # Check for improvement
                    is_best = val_metrics["loss"] < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_metrics["loss"]
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                    
                    # Save checkpoint
                    if epoch % self.config.save_every == 0 or is_best:
                        self.checkpoint_manager.save_checkpoint(
                            self.model,
                            self.optimizer,
                            scheduler,
                            self.scaler,
                            epoch,
                            global_step,
                            val_metrics,
                            self.model_config,
                            self.config,
                            is_best,
                        )
                    
                    logger.info(
                        f"Epoch {epoch + 1}/{extended_epochs} - "
                        f"Train Loss: {train_metrics['loss']:.4f}, "
                        f"Val Loss: {val_metrics['loss']:.4f}, "
                        f"Val Perplexity: {val_metrics['perplexity']:.2f}, "
                        f"LR: {current_lr:.2e}"
                    )
                    
                    # Early stopping check
                    if self.patience_counter >= self.config.early_stopping_patience:
                        logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break
                        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
        return history
    
    def _enhanced_train_epoch(
        self, 
        train_loader: DataLoader, 
        scheduler: LearningRateScheduler,
        history: Dict,
        global_step: int
    ) -> Dict[str, float]:
        """Enhanced training epoch with frequent logging."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(self.device)
            target_ids = batch["target_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(
                outputs.reshape(-1, outputs.size(-1)),
                target_ids.reshape(-1),
            )
            loss = loss / self.config.accumulation_steps
            
            # Backward pass
            loss.backward()
            
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Update EMA
                if self.ema_model:
                    self.ema_model.update()
                
                # Update scheduler
                scheduler.step()
                global_step += 1
                
                # Frequent logging
                if global_step % self.config.log_steps == 0:
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    history["step_losses"].append(loss.item() * self.config.accumulation_steps)
                    history["step_lrs"].append(current_lr)
                    
                    logger.debug(
                        f"Step {global_step}: Loss {loss.item():.4f}, LR {current_lr:.2e}"
                    )
            
            total_loss += loss.item() * self.config.accumulation_steps
            num_batches += 1
        
        return {
            "loss": total_loss / max(num_batches, 1),
            "global_step": global_step
        }
    
    def _enhanced_validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Enhanced validation with EMA model."""
        # Use EMA model for validation if available
        if self.ema_model:
            self.ema_model.apply_shadow()
        
        try:
            metrics = super()._validate_epoch(val_loader)
        finally:
            # Restore original parameters
            if self.ema_model:
                self.ema_model.restore()
        
        return metrics

class EnhancedLRScheduler(LearningRateScheduler):
    """Enhanced LR scheduler with better floor handling."""
    
    def __init__(self, optimizer, config: ImprovedTrainingConfig, total_steps: int):
        super().__init__(optimizer, config, total_steps)
        self.lr_floor = config.learning_rate * config.lr_floor_ratio
        
    def _get_lr(self) -> float:
        """Enhanced LR calculation with proper floor."""
        if self.current_step < self.warmup_steps:
            return self.base_lr * (self.current_step / self.warmup_steps)
        
        progress = (self.current_step - self.warmup_steps) / (
            self.total_steps - self.warmup_steps
        )
        progress = min(progress, 1.0)
        
        if self.config.scheduler_type == "cosine":
            lr = self.base_lr * 0.5 * (1 + math.cos(progress * math.pi))
        elif self.config.scheduler_type == "linear":
            lr = self.base_lr * (1 - progress)
        else:
            lr = self.base_lr
            
        # Apply floor
        return max(lr, self.lr_floor)

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing."""
    
    def __init__(self, smoothing: float = 0.1, ignore_index: int = -100):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass with label smoothing."""
        vocab_size = pred.size(-1)
        
        # Create smoothed labels
        smooth_target = torch.zeros_like(pred)
        smooth_target.fill_(self.smoothing / (vocab_size - 1))
        
        # Set true labels
        mask = target != self.ignore_index
        smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        # Apply mask
        smooth_target = smooth_target * mask.unsqueeze(1).float()
        
        # Compute loss
        log_prob = torch.log_softmax(pred, dim=-1)
        loss = -smooth_target * log_prob
        loss = loss.sum(dim=-1)
        
        # Average over non-ignored tokens
        return loss.sum() / mask.sum().float()

def main():
    """Main function with proper CLI interface."""
    import argparse
    import os
    import sys
    from pathlib import Path
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/improved_training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    parser = argparse.ArgumentParser(description='Enhanced model training with research-backed improvements')
    parser.add_argument('--data', type=str, required=True, help='Path to training data directory')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Output directory for models')
    parser.add_argument('--log_dir', type=str, default='logs', help='Log directory')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--lr_floor', type=float, default=1e-6, help='Learning rate floor')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--use_ema', action='store_true', help='Use EMA averaging')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing factor')
    parser.add_argument('--scheduler_type', type=str, default='cosine', choices=['cosine', 'linear'], help='LR scheduler type')
    
    args = parser.parse_args()
    
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs('plots', exist_ok=True)
        
        logger.info("Starting enhanced training with improved scheduling...")
        logger.info(f"Data directory: {args.data}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Learning rate: {args.learning_rate}")
        logger.info(f"LR floor: {args.lr_floor}")
        logger.info(f"Use EMA: {args.use_ema}")
        
        try:
            from dataset import MultiFormatDataProcessor, Tokenizer, DataConfig, create_data_loaders
            from model import DocumentationModel, ModelConfig
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            logger.error("Make sure all required Python files are in the scripts directory")
            sys.exit(1)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        data_path = Path(args.data)
        texts = []
        
        if data_path.is_file():
            # Single file
            text = MultiFormatDataProcessor.load_file(data_path)
            texts = [text] if text.strip() else []
        elif data_path.is_dir():
            # Directory of files
            for file_path in data_path.glob('**/*'):
                if file_path.is_file() and file_path.suffix.lower() in ['.txt', '.md', '.rst', '.json']:
                    try:
                        text = MultiFormatDataProcessor.load_file(file_path)
                        if text.strip():
                            texts.append(text)
                    except Exception as e:
                        logger.warning(f"Failed to load {file_path}: {e}")
        else:
            logger.error(f"Data path {args.data} does not exist")
            sys.exit(1)
        
        if not texts:
            logger.error("No valid texts found in data directory")
            sys.exit(1)
            
        logger.info(f"Loaded {len(texts)} text documents")
        
        tokenizer = Tokenizer(vocab_size=8000, use_subword=True)
        tokenizer.build_vocab(texts)
        
        data_config = DataConfig(
            max_length=256,
            batch_size=args.batch_size,
            train_split=0.8,
            val_split=0.1,
            test_split=0.1,
            min_text_length=10,
            max_text_length=2000
        )
        
        model_config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=512,
            nhead=8,
            num_layers=6,
            dim_feedforward=2048,
            max_length=data_config.max_length,
            dropout=0.1
        )
        
        training_config = ImprovedTrainingConfig(
            learning_rate=args.learning_rate,
            lr_floor_ratio=args.lr_floor / args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            use_ema=args.use_ema,
            label_smoothing=args.label_smoothing,
            scheduler_type=args.scheduler_type,
            eval_every=1,
            save_every=2,
            log_steps=50
        )
        
        try:
            train_loader, val_loader, test_loader = create_data_loaders(
                texts, tokenizer, data_config
            )
            logger.info(f"Loaded {len(train_loader)} training batches, {len(val_loader)} validation batches")
        except Exception as e:
            logger.error(f"Failed to create data loaders: {e}")
            import traceback
            logger.error(traceback.format_exc())
            sys.exit(1)
        
        model = DocumentationModel(model_config).to(device)
        logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
        
        trainer = ImprovedTrainer(
            model=model,
            model_config=model_config,
            training_config=training_config,
            checkpoint_dir=args.output_dir
        )
        
        history = trainer.train(train_loader, val_loader, test_loader)
        
        final_model_path = os.path.join(args.output_dir, 'model.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': model_config,
            'training_config': training_config,
            'tokenizer': tokenizer,
            'history': history
        }, final_model_path)
        logger.info(f"Saved final model to {final_model_path}")
        
        plot_training_curves(history, save_dir='plots')
        
        logger.info("Enhanced training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

def plot_training_curves(history: dict, save_dir: str = 'plots'):
    """Plot and save training curves."""
    import matplotlib.pyplot as plt
    
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history and history['val_loss']:
        plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    if 'val_perplexity' in history and history['val_perplexity']:
        plt.plot(history['val_perplexity'])
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.title('Validation Perplexity')
        plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(history['learning_rate'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    if 'step_losses' in history and history['step_losses']:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['step_losses'])
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Step-level Training Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(history['step_lrs'])
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.title('Step-level Learning Rate')
        plt.yscale('log')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'step_level_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Training curves saved to {save_dir}")

if __name__ == "__main__":
    main()
