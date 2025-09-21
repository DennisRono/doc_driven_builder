import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
import logging
from torch.utils.data import DataLoader

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
        plt.show()
        
        logger.info(f"Suggested learning rate: {suggested_lr:.2e}")
        return suggested_lr

def run_lr_range_test(model, train_loader, device, save_path="lr_range_test.png"):
    """Convenience function to run LR range test."""
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    lr_finder = LRRangeFinder(model, criterion, device)
    
    logger.info("Running learning rate range test...")
    lrs, losses = lr_finder.range_test(train_loader, num_iter=200)
    
    suggested_lr = lr_finder.plot_lr_range(lrs, losses, save_path)
    
    return suggested_lr, lrs, losses
