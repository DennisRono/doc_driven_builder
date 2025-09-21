.PHONY: help install train evaluate generate clean test setup-env infer package lr-test improved-train diagnostics

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
	python scripts/model.py generate \
		--checkpoint checkpoints/model.pt \
		--prompt "How to implement a neural network:" \
		--max_length 200

generate-custom:
	python scripts/model.py generate \
		--checkpoint checkpoints/model.pt \
		--prompt "$(PROMPT)" \
		--max_length 200

interactive:
	python scripts/model.py interactive \
		--checkpoint checkpoints/model.pt

# Usage commands
infer:
	python scripts/usage.py infer

package:
	python scripts/usage.py package

# plot-curves:
# 	python -c "
# 	import matplotlib.pyplot as plt
# 	import json
# 	import os
# 	if os.path.exists('logs/training_log.json'):
# 		with open('logs/training_log.json') as f:
# 			data = [json.loads(line) for line in f]
# 		epochs = [d['epoch'] for d in data]
# 		losses = [d['loss'] for d in data]
# 		plt.figure(figsize=(10, 6))
# 		plt.plot(epochs, losses)
# 		plt.title('Training Loss')
# 		plt.xlabel('Epoch')
# 		plt.ylabel('Loss')
# 		plt.savefig('plots/training_curves.png')
# 		print('Training curves saved to plots/training_curves.png')
# 	else:
# 		print('No training log found')
# "

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
