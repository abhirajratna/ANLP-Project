"""Unconditional samples from the 110M model: the "readable samples" sanity check.

MDLM's ancestral sampler for the absorbing process with a linear schedule: from
time t to s < t, each still-masked position is revealed with probability
(t - s) / t, and a revealed position takes a token drawn from the model's
distribution at that position (the mask token is excluded by SUBS).
"""

from __future__ import annotations

import torch


@torch.no_grad()
def ancestral_sample(model, n: int, length: int, mask_token_id: int, steps: int = 128,
                     device="cpu", temperature: float = 1.0, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.full((n, length), mask_token_id, dtype=torch.long, device=device)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for i in range(steps):
        t, s = float(ts[i]), float(ts[i + 1])
        masked = x == mask_token_id
        if not masked.any():
            break
        logits = model(x).float() / max(temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1)
        draw = torch.multinomial(probs.view(-1, probs.shape[-1]), 1, generator=g).view(n, length)
        p_reveal = 1.0 if s <= 0 else (t - s) / t
        reveal = masked & (torch.rand(n, length, device=device, generator=g) < p_reveal)
        x = torch.where(reveal, draw, x)
    return x
