"""A compact bidirectional transformer for masked diffusion language modelling.

Bidirectional, not causal: a diffusion LM predicts every masked position from
the whole visible context in one pass, which is the operation the audit measures.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 50258        # GPT-2 BPE + one mask token
    max_len: int = 256
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 3072
    dropout: float = 0.0
    mask_token_id: int = 50257

    @property
    def n_params(self) -> int:
        emb = self.vocab_size * self.d_model + self.max_len * self.d_model
        per_layer = 4 * self.d_model ** 2 + 2 * self.d_model * self.d_ff
        return emb + self.n_layers * per_layer


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.n_heads = cfg.n_heads
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        shape = (B, L, self.n_heads, D // self.n_heads)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        # No attention mask: every position attends to every other.
        a = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        a = a.transpose(1, 2).contiguous().view(B, L, D)
        x = x + self.drop(self.proj(a))
        h = self.ln2(x)
        return x + self.drop(self.fc2(F.gelu(self.fc1(h))))


class MaskedDiffusionLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight        # tied
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        L = ids.shape[1]
        x = self.tok(ids) + self.pos(torch.arange(L, device=ids.device))[None]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))


def mdlm_loss(model: nn.Module, x0: torch.Tensor, mask_token_id: int,
              t_eps: float = 1e-3) -> tuple[torch.Tensor, dict]:
    """Continuous-time masked-diffusion NELBO with the absorbing schedule.

    ``t ~ U(eps, 1)`` per sequence, each token masked independently with
    probability ``t``, and the cross-entropy at masked positions reweighted by
    ``1/t`` -- the linear-schedule weight, which is what makes the objective an
    ELBO on likelihood rather than plain masked-LM training.
    """
    B, L = x0.shape
    t = torch.rand(B, 1, device=x0.device) * (1.0 - t_eps) + t_eps
    mask = torch.rand(B, L, device=x0.device) < t
    # A sequence with nothing masked contributes no gradient; force one position.
    empty = ~mask.any(dim=1)
    if empty.any():
        mask[empty, torch.randint(0, L, (int(empty.sum()),), device=x0.device)] = True

    xt = torch.where(mask, torch.full_like(x0, mask_token_id), x0)
    logits = model(xt)
    ce = F.cross_entropy(logits.view(-1, logits.size(-1)), x0.view(-1),
                         reduction="none").view(B, L)
    per_seq = (ce * mask).sum(dim=1) / t.squeeze(1)
    loss = per_seq.mean() / L
    with torch.no_grad():
        n_masked = mask.sum().clamp(min=1)
        stats = {"ce_per_masked_token": float(((ce * mask).sum() / n_masked).detach()),
                 "mask_rate": float(mask.float().mean())}
    return loss, stats
