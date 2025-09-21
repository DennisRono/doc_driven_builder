import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
import logging
from torch.utils.data import DataLoader
import argparse
import os
import sys
from pathlib import Path
from dataset import MultiFormatDataProcessor, Tokenizer, DataConfig, create_data_loaders
from model import DocumentationModel, ModelConfig

logger = logging.getLogger(__name__)

class LRRangeFinder:
    """Learning rate range test implementation."""
    
    def __init__(self, model: nn.Module, criterion: nn.Module, device: torch.device):
        self.model = model
        self.criterion = criterion
        self.device = device
        
    def range_test(
        self,
        train_loader: DataLoader,
        start_lr: float = 1e-7,
        end_lr: float = 1e-1,
        num_iter: int = 100,
        smooth_f: float = 0.05,
        diverge_th: int = 5
    ) -> Tuple[List[float], List[float]]:
        """
        Perform learning rate range test.
        
        Args:
            train_loader: Training data loader
            start_lr: Starting learning rate
            end_lr: Ending learning rate  
            num_iter: Number of iterations
            smooth_f: Smoothing factor for loss
            diverge_th: Threshold for divergence detection
            
        Returns:
            Tuple of (learning_rates, losses)
        """
        # Save initial model state
        initial_state = self.model.state_dict()
        
        # Create optimizer with starting LR
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=start_lr,
            weight_decay=0.01,
            betas=(0.9, 0.95)
        )
        
        # Calculate LR multiplier
        lr_mult = (end_lr / start_lr) ** (1.0 / num_iter)
        
        lrs = []
        losses = []
        best_loss = float('inf')
        
        self.model.train()
        data_iter = iter(train_loader)
        
        for iteration in range(num_iter):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
                
            # Get current LR
            current_lr = optimizer.param_groups[0]['lr']
            lrs.append(current_lr)
            
            # Forward pass
            input_ids = batch["input_ids"].to(self.device)
            target_ids = batch["target_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(
                outputs.reshape(-1, outputs.size(-1)),
                target_ids.reshape(-1)
            )
            
            # Smooth loss
            if iteration == 0:
                avg_loss = loss.item()
            else:
                avg_loss = smooth_f * loss.item() + (1 - smooth_f) * avg_loss
                
            losses.append(avg_loss)
            
            # Check for divergence
            if avg_loss < best_loss:
                best_loss = avg_loss
            elif avg_loss > diverge_th * best_loss:
                logger.info(f"Stopping early at iteration {iteration} due to divergence")
                break
                
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_mult
                
        # Restore initial model state
        self.model.load_state_dict(initial_state)
        
        return lrs, losses
    
    def plot_lr_range(self, lrs: List[float], losses: List[float], save_path: str = None):
        """Plot learning rate range test results."""
        plt.figure(figsize=(10, 6))
        plt.semilogx(lrs, losses)
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Range Test')
        plt.grid(True)
        
        # Find suggested LR (steepest descent)
        gradients = np.gradient(losses)
        min_gradient_idx = np.argmin(gradients)
        suggested_lr = lrs[min_gradient_idx]
        
        plt.axvline(x=suggested_lr, color='red', linestyle='--', 
                   label=f'Suggested LR: {suggested_lr:.2e}')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close plot to prevent display issues
        
        logger.info(f"Suggested learning rate: {suggested_lr:.2e}")
        return suggested_lr

def run_lr_range_test(model, train_loader, device, save_path="lr_range_test.png"):
    """Convenience function to run LR range test."""
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    lr_finder = LRRangeFinder(model, criterion, device)
    
    logger.info("Running learning rate range test...")
    lrs, losses = lr_finder.range_test(train_loader)
    
    suggested_lr = lr_finder.plot_lr_range(lrs, losses, save_path)
    
    return suggested_lr, lrs, losses

def main():
    """Main function with proper CLI interface."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/lr_range_test.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    parser = argparse.ArgumentParser(description='Learning Rate Range Test')
    parser.add_argument('--data', type=str, required=True, help='Path to training data directory')
    parser.add_argument('--output_dir', type=str, default='plots', help='Output directory for plots')
    parser.add_argument('--min_lr', type=float, default=1e-8, help='Minimum learning rate')
    parser.add_argument('--max_lr', type=float, default=1e-2, help='Maximum learning rate')
    parser.add_argument('--num_iter', type=int, default=200, help='Number of iterations')
    
    args = parser.parse_args()
    
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        logger.info("Starting learning rate range test...")
        logger.info(f"Data directory: {args.data}")
        logger.info(f"LR range: {args.min_lr:.2e} to {args.max_lr:.2e}")
        logger.info(f"Iterations: {args.num_iter}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        data_path = Path(args.data)
        texts = []
        
        if data_path.is_file():
            text = MultiFormatDataProcessor.load_file(data_path)
            texts = [text] if text.strip() else []
        elif data_path.is_dir():
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
            batch_size=8,
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
        
        try:
            train_loader, _, _ = create_data_loaders(texts, tokenizer, data_config)
            logger.info(f"Loaded {len(train_loader)} training batches")
        except Exception as e:
            logger.error(f"Failed to create data loaders: {e}")
            import traceback
            logger.error(traceback.format_exc())
            sys.exit(1)
        
        model = DocumentationModel(model_config).to(device)
        logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
        
        save_path = os.path.join(args.output_dir, 'lr_range_test.png')
        suggested_lr, lrs, losses = run_lr_range_test(model, train_loader, device, save_path)
        
        results_path = os.path.join(args.output_dir, 'lr_range_results.txt')
        with open(results_path, 'w') as f:
            f.write(f"Learning Rate Range Test Results\n")
            f.write(f"================================\n")
            f.write(f"Suggested LR: {suggested_lr:.2e}\n")
            f.write(f"Min LR tested: {min(lrs):.2e}\n")
            f.write(f"Max LR tested: {max(lrs):.2e}\n")
            f.write(f"Min loss: {min(losses):.4f}\n")
            f.write(f"Iterations: {len(lrs)}\n")
        
        logger.info(f"LR range test completed successfully!")
        logger.info(f"Suggested learning rate: {suggested_lr:.2e}")
        logger.info(f"Results saved to {save_path} and {results_path}")
        
    except Exception as e:
        logger.error(f"LR range test failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
