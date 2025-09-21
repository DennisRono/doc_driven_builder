.PHONY: help install train evaluate generate clean test setup-env infer package lr-test improved-train diagnostics plot-curves optimize-training quickstart dev-cycle

help:
	@echo "Documentation-Driven Builder Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make install      - Install Python dependencies"
	@echo "  make setup-env    - Create directories and setup environment"
	@echo ""
	@echo "Training:"
	@echo "  make train        - Train model with sample documentation"
	@echo "  make train-custom - Train with custom documentation (DOC_PATH=path/to/docs)"
	@echo "  make resume       - Resume training from checkpoint"
	@echo "  make improved-train - Train with enhanced scheduling and EMA"
	@echo "  make lr-test      - Run learning rate range test"
	@echo "  make diagnostics  - Run training diagnostics and analysis"
	@echo ""
	@echo "Evaluation:"
	@echo "  make evaluate     - Evaluate trained model"
	@echo "  make perplexity   - Calculate model perplexity"
	@echo "  make eval-comprehensive - Run comprehensive evaluation suite"
	@echo ""
	@echo "Generation:"
	@echo "  make generate     - Generate text with default prompt"
	@echo "  make generate-custom - Generate with custom prompt (PROMPT='your prompt')"
	@echo "  make interactive  - Start interactive generation session"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean        - Clean output and checkpoint files"
	@echo "  make test         - Run basic functionality tests"
	@echo "  make demo         - Run complete demo workflow"
	@echo "  make plot-curves  - Plot training curves and diagnostics"
	@echo "  make optimize-training - Run optimized training workflow"
	@echo "  make quickstart   - Quickstart for new users"
	@echo "  make dev-cycle    - Development cycle for iterative improvements"

# Setup commands
install:
	poetry add torch transformers sentencepiece numpy tqdm matplotlib seaborn wandb

setup-env:
	mkdir -p output checkpoints sample_docs logs plots diagnostics
	@echo "Environment setup complete"

# Training commands
train: setup-env
	python scripts/training.py \
		--data sample_docs \
		--output_dir checkpoints \
		--log_dir logs

train-custom: setup-env
	python scripts/training.py \
		--data $(DOC_PATH) \
		--output_dir checkpoints \
		--log_dir logs

resume: setup-env
	python scripts/training.py \
		--data sample_docs \
		--checkpoint checkpoints/model.pt \
		--output_dir checkpoints \
		--log_dir logs

improved-train: setup-env
	python scripts/improved_training.py \
		--data sample_docs \
		--output_dir checkpoints \
		--log_dir logs \
		--use_ema \
		--lr_floor 1e-6 \
		--scheduler_type cosine

lr-test: setup-env
	python scripts/lr_range_test.py \
		--data sample_docs \
		--output_dir diagnostics \
		--min_lr 1e-8 \
		--max_lr 1e-2

diagnostics: setup-env
	python scripts/training_diagnostics.py \
		--checkpoint checkpoints/model.pt \
		--data sample_docs \
		--output_dir diagnostics

# Evaluation commands
evaluate:
	python scripts/evaluation.py \
		--checkpoint checkpoints/model.pt \
		--data sample_docs \
		--output_dir output

perplexity:
	python scripts/evaluation.py \
		--checkpoint checkpoints/model.pt \
		--data sample_docs \
		--metric perplexity \
		--output_dir output

eval-comprehensive:
	python scripts/evaluation.py \
		--checkpoint checkpoints/model.pt \
		--data sample_docs \
		--comprehensive \
		--output_dir output

# Generation commands
generate:
	python scripts/utils.py generate \
		--checkpoint checkpoints/model.pt \
		--prompt "How to implement a neural network:" \

generate-custom:
	python scripts/utils.py generate \
		--checkpoint checkpoints/model.pt \
		--prompt "$(PROMPT)" \

interactive:
	python scripts/model.py interactive \
		--checkpoint checkpoints/model.pt

# Usage commands
infer:
	python scripts/usage.py infer

package:
	python scripts/usage.py package

plot-curves:
	@echo "Plotting training curves..."
	@if [ -f "checkpoints/model.pt" ]; then \
		python -c "\
import torch; \
import matplotlib.pyplot as plt; \
import os; \
try: \
    checkpoint = torch.load('checkpoints/model.pt', map_location='cpu', weights_only=False); \
    history = checkpoint.get('history', {}); \
    if history: \
        plt.figure(figsize=(12, 8)); \
        if 'train_loss' in history: \
            plt.subplot(2, 2, 1); \
            plt.plot(history['train_loss'], label='Train Loss'); \
            if 'val_loss' in history: plt.plot(history['val_loss'], label='Val Loss'); \
            plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training Loss'); plt.legend(); plt.grid(True); \
        if 'val_perplexity' in history: \
            plt.subplot(2, 2, 2); \
            plt.plot(history['val_perplexity']); \
            plt.xlabel('Epoch'); plt.ylabel('Perplexity'); plt.title('Validation Perplexity'); plt.grid(True); \
        if 'learning_rate' in history: \
            plt.subplot(2, 2, 3); \
            plt.plot(history['learning_rate']); \
            plt.xlabel('Epoch'); plt.ylabel('Learning Rate'); plt.title('Learning Rate Schedule'); plt.yscale('log'); plt.grid(True); \
        if 'step_losses' in history: \
            plt.subplot(2, 2, 4); \
            plt.plot(history['step_losses'], alpha=0.7); \
            plt.xlabel('Step'); plt.ylabel('Loss'); plt.title('Step-wise Training Loss'); plt.grid(True); \
        plt.tight_layout(); \
        os.makedirs('plots', exist_ok=True); \
        plt.savefig('plots/training_curves.png', dpi=300, bbox_inches='tight'); \
        print('Training curves saved to plots/training_curves.png'); \
    else: \
        print('No training history found in checkpoint'); \
except Exception as e: \
    print(f'Error plotting curves: {e}'); \
"; \
	else \
		echo "No model checkpoint found at checkpoints/model.pt"; \
	fi

# Utility commands
clean:
	rm -rf output/* checkpoints/* logs/* plots/* diagnostics/* *.zip
	@echo "Cleaned output directories"

test: setup-env
	python scripts/utils.py test

demo: clean setup-env
	@echo "Running enhanced demo workflow..."
	@echo "1. Running LR range test..."
	make lr-test
	@echo "2. Training model with improved scheduling..."
	make improved-train
	@echo "3. Running comprehensive evaluation..."
	make eval-comprehensive
	@echo "4. Generating sample text..."
	make generate
	@echo "5. Plotting training curves..."
	make plot-curves
	@echo "Enhanced demo complete!"

optimize-training: clean setup-env lr-test
	@echo "Running optimized training workflow..."
	@echo "1. LR range test completed, check diagnostics/lr_range_test.png"
	@echo "2. Starting improved training with optimal settings..."
	make improved-train
	@echo "3. Running diagnostics..."
	make diagnostics
	@echo "4. Plotting results..."
	make plot-curves
	@echo "Training optimization complete!"

# Quick start for new users
quickstart: install demo
	@echo "Quickstart complete! Try 'make generate-custom PROMPT=\"your prompt\"'"

dev-cycle: lr-test improved-train diagnostics plot-curves
	@echo "Development cycle complete - check diagnostics/ and plots/ for analysis"
