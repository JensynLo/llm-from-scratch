import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
import math
from jaxtyping import Bool




class Linear(nn.Module):
    """
    线性层，不加偏置项，初始化的时候如果没有传入 weights，使用均值为 0，标准差为 2/(d_in+d_out) 并在[-3*std, 3*std]范围内截断的正态分布随机初始化权重。
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):

        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty(size=(out_features, in_features), **factory_kwargs)
        )
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty(size=(num_embeddings, embedding_dim), **factory_kwargs)
        )
        std = math.sqrt(2 / (num_embeddings + embedding_dim))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        return

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.ones(size=(d_model,), **factory_kwargs))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


"""$$FFN(x) = W_2(SiLU(W_1x) \odot W_3x)$$"""
class FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.weight2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.weight3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight2(F.silu(self.weight1(x)) * self.weight3(x))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert d_k % 2 == 0

        inv_freq = theta ** (-2 * torch.arange(0, d_k // 2, dtype=torch.float32) / d_k)

        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = positions[:, None] * inv_freq[None, :]

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        if device is not None:
            cos = cos.to(device)
            sin = sin.to(device)

        self.sin_cached: torch.Tensor
        self.cos_cached: torch.Tensor
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        self.d_k = d_k

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "... s (d two) -> ... s d two", two=2)
        x_even = x[..., 0]
        x_odd = x[..., 1]
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        out = torch.stack((out_even, out_odd), dim=-1)
        out = rearrange(out, "... s d two -> ... s (d two)")

        return out


def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    max_values = x.max(dim=i, keepdim=True).values
    x = x - max_values
    exps = torch.exp(x)
    s = torch.sum(exps, dim=i, keepdim=True)
    return exps / s


def scaled_dot_product(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Bool[torch.Tensor, " ... queries keys"] | None = None,
) -> torch.Tensor:
    dk = Q.size(-1)
    score = einsum(Q, K, "... q d, ... k d -> ... q k")
    score = score / dk**0.5
    if mask is not None:
        score = score.masked_fill(mask, float("-inf"))
    softed_score = softmax(score, i=-1)
    output = einsum(softed_score, V, "... q k, ... k d -> ... q d")
    return output


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
        # scaled
        self.scale = 1.0 / math.sqrt(self.d_k)
        # rope
        self.rope = RotaryPositionalEmbedding(
            theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device
        )
        return

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, seq_len, d_model = x.shape
        # 得到 QKV
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        # 转换样式
        Q = rearrange(Q, "b s (n d_k) -> b n s d_k", n=self.num_heads, d_k=self.d_k)
        K = rearrange(K, "b s (n d_k) -> b n s d_k", n=self.num_heads, d_k=self.d_k)
        V = rearrange(V, "b s (n d_v) -> b n s d_v", n=self.num_heads, d_v=self.d_v)
        # 默认 token positions: shape (1, seq_len) 以便广播到 (batch, num_heads, seq_len)
        token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        causal_mask = rearrange(causal_mask, "s_q s_k -> 1 1 s_q s_k")
        if mask is not None:
            if mask.dim() == 2:
                mask = rearrange(mask, "b s -> b 1 1 s")
                mask = mask.expand(-1, -1, seq_len, -1)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            combined_mask = causal_mask | (mask == 0)
        else:
            combined_mask = causal_mask
        attention_output = scaled_dot_product(Q, K, V, combined_mask)
        attention_output = rearrange(attention_output, "b n s d_v -> b s (n d_v)")
        output = self.W_o(attention_output)

        return output


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
        self.Embedding = Embedding(
            num_embeddings=self.vocab_size, embedding_dim=d_model, device=device
        )
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
