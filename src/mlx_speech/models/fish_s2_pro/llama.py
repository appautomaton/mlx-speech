import mlx.nn as nn
import mlx.core as mx


class RMSNorm(nn.Module):
    """RMSNorm - Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x):
        norm = mx.sqrt(mx.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return x / norm * self.weight


class DecoderLayer(nn.Module):
    """Pre-norm transformer decoder layer with attention + FFN."""

    def __init__(self, num_heads: int, dim: int, mlp_dim: int):
        super().__init__()
        self.attn = nn.MultiHeadAttention(num_heads=num_heads, dims=dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def __call__(self, x: mx.array) -> mx.array:
        normalized = self.norm1(x)
        attn_output = self.attn(normalized, normalized, normalized)
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x


class DualARTransformer(nn.Module):
    """Fish Audio S2 Pro Dual-Autoregressive Transformer.

    Architecture:
    - vocab embedding
    - num_layers of transformer layers
    - RMSNorm
    - lm_head linear
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        num_layers: int = 30,
        dim: int = 2048,
        num_heads: int = 16,
        max_position_embeddings: int = 8192,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dim = dim
        self.num_heads = num_heads

        self.token_embed = nn.Embedding(vocab_size, dim)

        mlp_dim = dim * 4
        self.layers = [
            DecoderLayer(num_heads=num_heads, dim=dim, mlp_dim=mlp_dim)
            for _ in range(num_layers)
        ]

        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: (batch, seq) token IDs

        Returns:
            logits: (batch, seq, vocab_size)
        """
        h = self.token_embed(x)

        for layer in self.layers:
            h = layer(h)

        return self.lm_head(self.norm(h))
