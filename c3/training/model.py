"""A compact bidirectional transformer for masked diffusion language modelling.

Bidirectional, not causal: a diffusion LM predicts every masked position from
the whole visible context in one pass, which is the operation the audit measures.

Two details follow the MDLM recipe (Sahoo et al., 2024):

* **SUBS.**  The [MASK] token is never a valid prediction, so its logit is set
  to -inf in the output ("zero masking probabilities").  Without it the model
  leaks a little probability mass onto [MASK], which deflates every confidence
  the audit reads and can even put [MASK] in the top-1.
* **No time conditioning.**  MDLM shows the absorbing-state denoiser does not
  need t: the number of masks in the input already reveals it.

Padding (short documents, plan v2's "no packing") is handled by a key-padding
mask, so real tokens never attend to pad positions and pads never enter the loss.

``causal=True`` turns the same network into a left-to-right language model: the
autoregressive reference of plan v2 T3 ("a same-size autoregressive GPT-2 on the
same data"), trained by the same loop with ``ar_loss`` so its exact perplexity
can be set beside the diffusion model's bound.
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
    causal: bool = False           # True: the autoregressive reference model

    @property
    def n_params(self) -> int:
        emb = self.vocab_size * self.d_model + self.max_len * self.d_model
        per_layer = 4 * self.d_model ** 2 + 2 * self.d_model * self.d_ff
        return emb + self.n_layers * per_layer


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.causal = cfg.causal
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.n_heads = cfg.n_heads
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, L, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        shape = (B, L, self.n_heads, D // self.n_heads)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        # Bidirectional unless this is the AR reference; padding is hidden when present.
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                           is_causal=self.causal and attn_mask is None)
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

    def forward(self, ids: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        L = ids.shape[1]
        x = self.tok(ids) + self.pos(torch.arange(L, device=ids.device))[None]
        attn_mask = None
        if valid is not None and not bool(valid.all()):
            # (B, 1, 1, L): True = this key may be attended to.
            attn_mask = valid[:, None, None, :].to(torch.bool)
            if self.cfg.causal:
                tri = torch.ones(L, L, dtype=torch.bool, device=ids.device).tril()
                # Keep the diagonal so a pad query still attends to something.
                attn_mask = (attn_mask & tri[None, None]) | torch.eye(
                    L, dtype=torch.bool, device=ids.device)[None, None]
        for blk in self.blocks:
            x = blk(x, attn_mask)
        logits = self.head(self.ln_f(x))
        # SUBS: [MASK] is never a prediction.
        never = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
        never[self.cfg.mask_token_id] = True
        return logits.masked_fill(never, float("-inf"))


def _sample_t(B: int, device, t_eps: float, antithetic: bool, generator=None) -> torch.Tensor:
    """Per-sequence diffusion times in [t_eps, 1).

    ``antithetic`` uses MDLM's low-discrepancy sampler: one uniform offset plus
    evenly spaced strata, so a batch always covers the whole range of t and the
    1/t-weighted loss has much lower variance than with independent draws.
    """
    if antithetic:
        u = torch.rand(1, device=device, generator=generator)
        t = (u + torch.arange(B, device=device) / B) % 1.0
    else:
        t = torch.rand(B, device=device, generator=generator)
    return (t * (1.0 - t_eps) + t_eps)[:, None]


def mdlm_loss(model: nn.Module, x0: torch.Tensor, mask_token_id: int,
              valid: torch.Tensor | None = None, t_eps: float = 1e-3,
              antithetic: bool = True, keep: torch.Tensor | None = None
              ) -> tuple[torch.Tensor, dict]:
    """Continuous-time masked-diffusion NELBO with the absorbing schedule.

    ``t ~ U(eps, 1)`` per sequence, each real token masked independently with
    probability ``t``, and the cross-entropy at masked positions reweighted by
    ``1/t`` -- the linear-schedule weight, which is what makes the objective an
    ELBO on likelihood rather than plain masked-LM training.  The result is per
    real (non-pad) token.

    ``keep`` (B, L) marks positions that are never masked -- context the model is
    always given at evaluation, such as the T9 probe's rho marker.  They still
    count as real tokens, with zero loss, as every unmasked token does.
    """
    B, L = x0.shape
    if valid is None:
        valid = torch.ones_like(x0, dtype=torch.bool)
    maskable = valid if keep is None else valid & ~keep
    t = _sample_t(B, x0.device, t_eps, antithetic)
    mask = (torch.rand(B, L, device=x0.device) < t) & maskable
    # A sequence with nothing masked contributes no gradient; force one real
    # position.  (Training heuristic only: the evaluation NELBO does not do this.)
    empty = ~mask.any(dim=1) & maskable.any(dim=1)
    if empty.any():
        rows = torch.nonzero(empty, as_tuple=False)[:, 0]
        n_real = maskable[rows].sum(dim=1)
        pick = (torch.rand(rows.numel(), device=x0.device) * n_real).long()
        # the pick-th maskable position of each empty row
        cum = maskable[rows].long().cumsum(dim=1) - 1
        cols = (cum == pick[:, None]).long().argmax(dim=1)
        mask[rows, cols] = True

    xt = torch.where(mask, torch.full_like(x0, mask_token_id), x0)
    logits = model(xt, valid)
    ce = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), x0.view(-1),
                         reduction="none").view(B, L)
    per_seq = (ce * mask).sum(dim=1) / t.squeeze(1)
    loss = per_seq.sum() / valid.sum().clamp(min=1)
    with torch.no_grad():
        n_masked = mask.sum().clamp(min=1)
        stats = {"ce_per_masked_token": float(((ce * mask).sum() / n_masked).detach()),
                 "mask_rate": float(mask.sum() / valid.sum().clamp(min=1)),
                 "real_tokens": int(valid.sum())}
    return loss, stats


@torch.no_grad()
def nelbo_eval(model: nn.Module, x0: torch.Tensor, mask_token_id: int,
               valid: torch.Tensor | None = None, n_t: int = 16, seed: int = 0,
               t_eps: float = 1e-3, keep: torch.Tensor | None = None) -> tuple[float, float]:
    """(summed NELBO in nats, real tokens) for a batch: the likelihood bound.

    NELBO = integral_0^1 (1/t) E[sum over masked of -log p(x0_i | x_t)] dt, by the
    midpoint rule over ``n_t`` strata of t with one mask draw each.  Seeded, so
    the same batch always gives the same number (the G2 curve must not be noisy
    for reasons unrelated to training).  No position is forced to be masked.

    ``keep`` (B, L) marks context that is never masked and so never scored, as in
    ``mdlm_loss``: a model trained with a visible prefix is validated the same way.
    """
    B, L = x0.shape
    if valid is None:
        valid = torch.ones_like(x0, dtype=torch.bool)
    maskable = valid if keep is None else valid & ~keep
    g = torch.Generator(device=x0.device).manual_seed(int(seed))
    total = torch.zeros((), device=x0.device, dtype=torch.float64)
    grid = (torch.arange(n_t, device=x0.device, dtype=torch.float64) + 0.5) / n_t
    grid = grid * (1.0 - t_eps) + t_eps
    for t in grid:
        mask = (torch.rand(B, L, device=x0.device, generator=g) < float(t)) & maskable
        if not mask.any():
            continue
        xt = torch.where(mask, torch.full_like(x0, mask_token_id), x0)
        logits = model(xt, valid)
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), x0.view(-1),
                             reduction="none").view(B, L)
        total += (ce * mask).sum().double() / float(t)
    return float(total / n_t), float(maskable.sum())


def ar_loss(model: nn.Module, x0: torch.Tensor, valid: torch.Tensor | None = None
            ) -> tuple[torch.Tensor, dict]:
    """Next-token cross-entropy per predicted real token (the AR reference)."""
    B, L = x0.shape
    if valid is None:
        valid = torch.ones_like(x0, dtype=torch.bool)
    logits = model(x0, valid)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                         x0[:, 1:].reshape(-1), reduction="none").view(B, L - 1)
    tgt = valid[:, 1:] & valid[:, :-1]
    loss = (ce * tgt).sum() / tgt.sum().clamp(min=1)
    stats = {"ce_per_masked_token": float(loss.detach()), "mask_rate": 0.0,
             "real_tokens": int(valid.sum())}
    return loss, stats


@torch.no_grad()
def ar_nll_eval(model: nn.Module, x0: torch.Tensor, valid: torch.Tensor | None = None,
                **_) -> tuple[float, float]:
    """(summed exact NLL in nats, predicted tokens) for a batch under the AR model."""
    B, L = x0.shape
    if valid is None:
        valid = torch.ones_like(x0, dtype=torch.bool)
    logits = model(x0, valid)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                         x0[:, 1:].reshape(-1), reduction="none").view(B, L - 1)
    tgt = valid[:, 1:] & valid[:, :-1]
    return float((ce * tgt).sum()), float(tgt.sum())
