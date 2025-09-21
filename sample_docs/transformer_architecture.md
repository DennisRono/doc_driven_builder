# Transformer Architecture Guide

## Overview

The Transformer architecture revolutionized natural language processing by introducing the attention mechanism as the primary building block.

## Key Components

### Self-Attention Mechanism

Self-attention allows the model to weigh the importance of different positions in the input sequence:

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

Where:

- Q: Query matrix
- K: Key matrix
- V: Value matrix
- d_k: Dimension of key vectors

### Multi-Head Attention

Multi-head attention runs multiple attention mechanisms in parallel:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
```

### Position Encoding

Since transformers don't have inherent position awareness, we add positional encodings:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

## Architecture Details

### Encoder Stack

- 6 identical layers
- Each layer has two sub-layers:
  - Multi-head self-attention
  - Position-wise feed-forward network
- Residual connections around each sub-layer
- Layer normalization

### Decoder Stack

- 6 identical layers
- Three sub-layers per layer:
  - Masked multi-head self-attention
  - Multi-head attention over encoder output
  - Position-wise feed-forward network

## Implementation Tips

1. **Scaling**: Scale attention scores by √d_k to prevent softmax saturation
2. **Masking**: Use causal masking in decoder to prevent looking ahead
3. **Dropout**: Apply dropout to attention weights and feed-forward layers
4. **Warmup**: Use learning rate warmup for stable training
