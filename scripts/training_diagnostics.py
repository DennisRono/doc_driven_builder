import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import logging
import math
from pathlib import Path
import argparse
import os
import sys
import json

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
        plt.close()
        
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

def main():
    """Main function with proper CLI interface."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/training_diagnostics.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    parser = argparse.ArgumentParser(description='Training Diagnostics and Analysis')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to training data directory')
    parser.add_argument('--output_dir', type=str, default='diagnostics', help='Output directory for diagnostics')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for analysis')
    
    args = parser.parse_args()
    
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
        
        logger.info("Starting training diagnostics...")
        logger.info(f"Checkpoint: {args.checkpoint}")
        logger.info(f"Data directory: {args.data}")
        logger.info(f"Output directory: {args.output_dir}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load checkpoint
        if not os.path.exists(args.checkpoint):
            logger.error(f"Checkpoint not found: {args.checkpoint}")
            sys.exit(1)
        
        try:
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
            logger.info("Loaded checkpoint successfully")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            sys.exit(1)
        
        try:
            from dataset import create_data_loaders, Tokenizer, DataConfig
            from model import DocumentationModel
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            sys.exit(1)
        
        # Load model
        model_config = checkpoint.get('model_config')
        if model_config is None:
            logger.error("Model config not found in checkpoint")
            sys.exit(1)
        
        model = DocumentationModel(model_config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Model loaded successfully")
        
        # Load training texts
        train_file = os.path.join(args.data, 'train.txt')
        if not os.path.exists(train_file):
            logger.error(f"Training data not found: {train_file}")
            sys.exit(1)
        
        try:
            with open(train_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(texts)} training texts")
        except Exception as e:
            logger.error(f"Failed to load training texts: {e}")
            sys.exit(1)
        
        # Create tokenizer and data config
        tokenizer = Tokenizer(vocab_size=8000, use_subword=True)
        data_config = DataConfig(
            max_length=model_config.max_seq_length,
            batch_size=args.batch_size,
            train_split=0.8,
            val_split=0.1,
            test_split=0.1
        )
        
        # Create data loaders with correct API
        try:
            train_loader, val_loader, _ = create_data_loaders(texts, tokenizer, data_config)
            logger.info(f"Loaded {len(train_loader)} training batches")
        except Exception as e:
            logger.error(f"Failed to create data loaders: {e}")
            sys.exit(1)
        
        # Initialize diagnostics
        diagnostics = TrainingDiagnostics(model, device)
        
        # Run diagnostics
        logger.info("Running gradient analysis...")
        grad_analysis = diagnostics.gradient_analysis(train_loader)
        
        logger.info("Running overfitting test...")
        overfit_results = diagnostics.overfitting_test(train_loader, num_samples=5, max_epochs=20)
        
        logger.info("Running parameter change analysis...")
        param_changes = diagnostics.parameter_change_analysis(train_loader, num_steps=10)
        
        # Plot results
        logger.info("Generating diagnostic plots...")
        
        # Plot overfitting test
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(overfit_results['losses'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Overfitting Test - Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(overfit_results['accuracies'])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Overfitting Test - Accuracy')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'overfitting_test.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot parameter changes
        plt.figure(figsize=(10, 6))
        for name, changes in param_changes.items():
            if 'weight' in name:  # Only plot weight parameters
                plt.plot(changes, label=name[:20] + '...' if len(name) > 20 else name)
        plt.xlabel('Training Step')
        plt.ylabel('Parameter Change')
        plt.title('Parameter Changes During Training')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'parameter_changes.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save results
        results = {
            'gradient_analysis': grad_analysis,
            'overfitting_test': overfit_results,
            'parameter_changes': {k: v for k, v in param_changes.items() if len(v) > 0}
        }
        
        results_path = os.path.join(args.output_dir, 'diagnostics_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate summary report
        summary_path = os.path.join(args.output_dir, 'diagnostics_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Training Diagnostics Summary\n")
            f.write("============================\n\n")
            
            f.write("Gradient Analysis:\n")
            f.write(f"  Total gradient norm: {grad_analysis['summary']['total_gradient_norm']:.6f}\n")
            f.write(f"  Average gradient norm: {grad_analysis['summary']['avg_gradient_norm']:.6f}\n")
            f.write(f"  Parameters with gradients: {grad_analysis['summary']['parameters_with_gradients']}\n")
            f.write(f"  Current loss: {grad_analysis['summary']['loss']:.4f}\n\n")
            
            f.write("Overfitting Test:\n")
            f.write(f"  Final loss: {overfit_results['losses'][-1]:.4f}\n")
            f.write(f"  Final accuracy: {overfit_results['accuracies'][-1]:.4f}\n")
            f.write(f"  Epochs run: {len(overfit_results['losses'])}\n\n")
            
            f.write("Parameter Changes:\n")
            for name, changes in param_changes.items():
                if changes and 'weight' in name:
                    f.write(f"  {name}: {changes[-1]:.6f}\n")
        
        logger.info(f"Diagnostics completed successfully!")
        logger.info(f"Results saved to {args.output_dir}")
        logger.info(f"Summary: {summary_path}")
        logger.info(f"Plots: {args.output_dir}/overfitting_test.png, {args.output_dir}/parameter_changes.png")
        
    except Exception as e:
        logger.error(f"Diagnostics failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
