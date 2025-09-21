.PHONY: help install train evaluate generate clean test setup-env infer evaluate package

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
	@echo ""
	@echo "Evaluation:"
	@echo "  make evaluate     - Evaluate trained model"
	@echo "  make perplexity   - Calculate model perplexity"
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

# Setup commands
install:
	poetry add torch transformers sentencepiece numpy tqdm

setup-env:
	mkdir -p output checkpoints sample_docs logs
	@echo "Environment setup complete"

# Training commands
train: setup-env
	python scripts/main.py train \
		--data sample_docs

train-custom: setup-env
	python scripts/main.py train \
		--data $(DOC_PATH)

resume: setup-env
	python scripts/main.py train \
		--data sample_docs \
		--checkpoint checkpoints/model.pt

# Evaluation commands
evaluate:
	python scripts/main.py evaluate \
		--checkpoint checkpoints/model.pt \
		--data sample_docs

perplexity:
	python scripts/main.py evaluate \
		--checkpoint checkpoints/model.pt \
		--data sample_docs \
		--metric perplexity

# Generation commands
generate:
	python scripts/main.py generate \
		--checkpoint checkpoints/model.pt \
		--prompt "How to implement a neural network:" \
		--max-length 200

generate-custom:
	python scripts/main.py generate \
		--checkpoint checkpoints/model.pt \
		--prompt "$(PROMPT)" \
		--max-length 200

interactive:
	python scripts/main.py interactive \
		--checkpoint checkpoints/model.pt

# usage
infer:
	python scripts/usage.py infer

evaluate:
	python scripts/usage.py evaluate

package:
	python scripts/usage.py package

# Utility commands
clean:
	rm -rf output/* checkpoints/* logs/* *.zip
	@echo "Cleaned output directories"

test: setup-env
	python scripts/main.py test

demo: clean setup-env
	@echo "Running complete demo workflow..."
	@echo "1. Training model on sample documentation..."
	make train
	@echo "2. Evaluating model..."
	make evaluate
	@echo "3. Generating sample text..."
	make generate
	@echo "Demo complete!"

# Quick start for new users
quickstart: install demo
	@echo "Quickstart complete! Try 'make generate-custom PROMPT=\"your prompt\"'"
