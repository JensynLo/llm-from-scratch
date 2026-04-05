import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from src.transformer import (
    Linear,
    RotaryPositionalEmbedding,
    RMSNorm,
    FeedForward,
    Embedding,
)


class multihead_self_attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // num_heads
        self.d_v = self.d_model // num_heads

        # 线性变换
        self.W_q = Linear(d_model, d_model, device=device)
        self.W_k = Linear(d_model, d_model, device=device)
        self.W_v = Linear(d_model, d_model, device=device)
        self.W_o = Linear(d_model, d_model, device=device)

        # rope 位置编码
        self.rope = RotaryPositionalEmbedding(
            theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device
        )
        return

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, seq_len, d_model = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = rearrange(Q, "b s (n d_k) -> b n s d_k", n=self.num_heads)
        K = rearrange(K, "b s (n d_k) -> b n s d_k", n=self.num_heads)
        V = rearrange(V, "b s (n d_v) -> b n s d_v", n=self.num_heads)

        # 注入旋转位置编码 (RoPE)
        token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        is_causal = True

        attention_output = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=is_causal,
        )

        attention_output = rearrange(attention_output, "b n s d -> b s (n d)")

        return self.W_o(attention_output)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.Norm1 = RMSNorm(d_model=d_model, device=device)
        self.attention = multihead_self_attention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
        )
        self.Norm2 = RMSNorm(d_model=d_model, device=device)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, device=device)
        return

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self.attention(self.Norm1(x), mask=mask)
        x = x + self.ffn(self.Norm2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        # Embedding layer
        self.Embedding = Embedding(
            num_embeddings=self.vocab_size, embedding_dim=d_model, device=device
        )
        # TF blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    theta=theta,
                    max_seq_len=self.context_length,
                    device=device,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.rms = RMSNorm(d_model=d_model, device=device)
        self.lm_head = Linear(d_model, vocab_size, device=device)
        return

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.Embedding(x)
        for layer in self.layers:
            x = layer(x, mask=mask)
        x = self.rms(x)
        logits = self.lm_head(x)
        return logits
