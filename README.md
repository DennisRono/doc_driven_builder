# Documentation-Driven Builder

A PyTorch-based system that learns from technical documentation to generate contextually relevant outputs.

## Quick Start

```bash
# Install dependencies
make install

# Run complete demo
make demo

# Generate text with custom prompt
make generate-custom PROMPT="How to implement attention mechanism:"
```

## Features

- **Modular Architecture**: Separate components for dataset, model, training, and evaluation
- **Advanced Tokenization**: Subword tokenization with BPE-like approach
- **Flexible Training**: Mixed-precision training with gradient clipping and checkpointing
- **Comprehensive Evaluation**: Perplexity, coverage, and accuracy metrics
- **Multiple Formats**: Support for Markdown, JSON, and plain text documentation
- **CLI Interface**: Easy-to-use command-line interface with comprehensive options

## Usage Examples

### Training

```bash
# Train on sample documentation
make train

# Train on custom documentation
make train-custom DOC_PATH=path/to/your/docs

# Resume from checkpoint
make resume
```

### Generation

```bash
# Interactive mode
make interactive

# Batch generation
make generate-custom PROMPT="Explain neural networks:"
```

### Evaluation

```bash
# Full evaluation
make evaluate

# Perplexity only
make perplexity
```

## Architecture

The system uses a transformer-based architecture with:

- Multi-head self-attention
- Positional encoding
- Residual connections
- Layer normalization
- Configurable depth and width

## Sample Documentation

The `sample_docs/` directory contains example documentation covering:

- PyTorch neural network tutorials
- Transformer architecture guides
- API documentation (JSON format)
- Training procedures (plain text)

## Requirements

- Python 3.8+
- PyTorch ≥2.2.0
- Additional dependencies listed in Makefile
