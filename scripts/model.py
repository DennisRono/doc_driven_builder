import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for the transformer model."""

    vocab_size: int = 8000
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 2048
    max_length: int = 256
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation: str = "gelu"
    layer_norm_eps: float = 1e-5
    use_residual_connections: bool = True
    use_layer_norm: bool = True


class MultiHeadAttention(nn.Module):
    """Enhanced multi-head attention with dropout and residual connections."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.nhead = config.nhead
        self.head_dim = config.d_model // config.nhead

        assert self.head_dim * config.nhead == config.d_model

        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.dropout = nn.Dropout(config.dropout)

        self.scale = math.sqrt(self.head_dim)

    def _prepare_mask_for_scores(
        self, mask: torch.Tensor, batch_size: int, seq_len: int
    ) -> torch.BoolTensor:
        """
        Normalize mask into boolean tensor of shape [batch_size, nhead, seq_len, seq_len].
        Accepts:
          - mask shape [seq_len, seq_len]
          - mask shape [batch_size, seq_len, seq_len]
          - mask shape [batch_size, 1, seq_len, seq_len]
          - mask shape [1, 1, seq_len, seq_len]
        """
        if mask is None:
            return None

        mask_bool = mask.bool()

        if mask_bool.dim() == 2:

            mask_b = (
                mask_bool.unsqueeze(0)
                .unsqueeze(0)
                .expand(batch_size, self.nhead, seq_len, seq_len)
            )
            return mask_b

        if mask_bool.dim() == 3:

            if mask_bool.shape[0] != batch_size:

                mask_b = mask_bool.unsqueeze(1).expand(
                    batch_size, self.nhead, seq_len, seq_len
                )
                return mask_b
            else:
                return mask_bool.unsqueeze(1).expand(
                    batch_size, self.nhead, seq_len, seq_len
                )

        if mask_bool.dim() == 4:

            b, h, s1, s2 = mask_bool.shape
            if s1 != seq_len or s2 != seq_len:
                raise ValueError(
                    f"Mask spatial dims ({s1},{s2}) don't match seq_len={seq_len}"
                )
            if b == batch_size and h == self.nhead:
                return mask_bool
            if b == batch_size and h == 1:
                return mask_bool.expand(batch_size, self.nhead, seq_len, seq_len)
            if b == 1 and h == 1:
                return mask_bool.expand(batch_size, self.nhead, seq_len, seq_len)

            return mask_bool.expand(batch_size, self.nhead, seq_len, seq_len)

        raise ValueError(f"Unsupported mask shape {mask.shape}")

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape

        logger.info(f"[MHA] Input x shape: {x.shape}")
        logger.info(f"[MHA] batch={batch_size}, seq_len={seq_len}, d_model={d_model}")

        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.nhead, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.nhead, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.nhead, self.head_dim)
            .transpose(1, 2)
        )

        logger.info(f"[MHA] q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        logger.info(f"[MHA] scores shape (before mask): {scores.shape}")

        mask_for_scores = None
        if mask is not None:
            try:
                mask_for_scores = self._prepare_mask_for_scores(
                    mask, batch_size, seq_len
                )
                logger.info(
                    f"[MHA] mask_for_scores shape (after prepare): {mask_for_scores.shape}"
                )
            except Exception as e:
                logger.exception("[MHA] error preparing mask for scores")
                raise

            scores = scores.masked_fill(mask_for_scores, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        logger.info(f"[MHA] attn_output shape (before transpose): {attn_output.shape}")

        attn_output_t = attn_output.transpose(1, 2).contiguous()
        logger.info(
            f"[MHA] attn_output_t shape (after transpose): {attn_output_t.shape}"
        )
        logger.info(
            f"[MHA] attn_output_t numel={attn_output_t.numel()}, expected={batch_size*seq_len*d_model}"
        )

        attn_output = attn_output_t.view(batch_size, seq_len, d_model)
        logger.info(f"[MHA] attn_output (final) shape: {attn_output.shape}")

        output = self.out_proj(attn_output)
        return self.dropout(output)


class FeedForward(nn.Module):
    """Enhanced feed-forward network with configurable activation."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.dim_feedforward)
        self.linear2 = nn.Linear(config.dim_feedforward, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        if config.activation == "gelu":
            self.activation = nn.GELU()
        elif config.activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Enhanced transformer block with residual connections and layer norm."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

        if config.use_layer_norm:
            self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
            self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.use_residual = config.use_residual_connections

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        if self.use_residual:
            attn_out = self.attention(self.norm1(x), mask)
            x = x + attn_out
        else:
            x = self.attention(self.norm1(x), mask)

        if self.use_residual:
            ff_out = self.feed_forward(self.norm2(x))
            x = x + ff_out
        else:
            x = self.feed_forward(self.norm2(x))

        return x


class PositionalEncoding(nn.Module):
    """Enhanced positional encoding with learnable option."""

    def __init__(self, config: ModelConfig, learnable: bool = False):
        super().__init__()
        self.d_model = config.d_model
        self.learnable = learnable

        if learnable:
            self.pos_embedding = nn.Embedding(config.max_length, config.d_model)
        else:
            pe = torch.zeros(config.max_length, config.d_model)
            position = torch.arange(0, config.max_length, dtype=torch.float).unsqueeze(
                1
            )
            div_term = torch.exp(
                torch.arange(0, config.d_model, 2).float()
                * (-math.log(10000.0) / config.d_model)
            )

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)

        if self.learnable:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            pos_emb = self.pos_embedding(positions)
            x = x + pos_emb
        else:
            x = x + self.pe[:seq_len, :].unsqueeze(0)

        return self.dropout(x)


class DocumentationModel(nn.Module):
    """Enhanced transformer model with modern architecture features."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config, learnable=False)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )

        if config.use_layer_norm:
            self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        else:
            self.final_norm = nn.Identity()

        self.output_projection = nn.Linear(config.d_model, config.vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using modern best practices."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for autoregressive generation."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()
        return mask

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        causal_mask = self.create_causal_mask(seq_len, device)

        mask_to_pass = causal_mask
        if attention_mask is not None:

            extended_mask = attention_mask.unsqueeze(1).unsqueeze(1)

            token_mask = extended_mask == 0

            causal_b = causal_mask.unsqueeze(0).unsqueeze(1)

            token_mask_broadcast = token_mask.expand(-1, -1, seq_len, -1)
            mask_to_pass = causal_b | token_mask_broadcast

        for block in self.blocks:
            x = block(x, mask_to_pass)

        x = self.final_norm(x)
        logits = self.output_projection(x)

        return logits

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
