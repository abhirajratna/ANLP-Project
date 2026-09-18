"""Model adapters.

Every audited checkpoint is reduced to one operation: given a batch of token
sequences in which some positions hold the mask token, return the logits.  That
is the single forward pass the whole design is built on.

Precision is a first-class argument because quantisation perturbs exactly the
logits under audit.  The guard below matters on this project's cluster: Ada's
u22 partition is RTX 2080 Ti (compute capability 7.5, Turing), which has no
bfloat16 support, so the planned bf16 replication cannot run there and must be
fp32 or move to an Ampere partition.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Mask token ids read from each checkpoint's own config.json on 2026-09-18, not
# copied from a paper. Getting one wrong would silently corrupt every trace: the
# model would be conditioned on an ordinary token where a mask was intended, and
# nothing downstream would flag it.
KNOWN_MODELS = {
    "llada-8b": {
        "path": "GSAI-ML/LLaDA-8B-Instruct",
        "mask_token_id": 126336,       # verified: vocab 126464, 32 layers, d_model 4096
        "local_hint": "/share1/ishita/llada_model",   # already on Ada, 15 GB, group-readable
    },
    "dream-7b": {
        "path": "Dream-org/Dream-v0-Instruct-7B",
        "mask_token_id": 151666,       # verified: vocab 152064, 28 layers, hidden 3584
        "local_hint": None,
    },
}


def verify_mask_token(model) -> None:
    """Cross-check the configured mask id against the loaded checkpoint's config.

    Cheap, and it catches the one error that would invalidate a whole run
    without producing any visible symptom.
    """
    cfg_id = getattr(getattr(model, "model", None), "config", None)
    cfg_id = getattr(cfg_id, "mask_token_id", None)
    if cfg_id is not None and int(cfg_id) != int(model.mask_token_id):
        raise RuntimeError(
            f"{model.name}: configured mask_token_id={model.mask_token_id} but the "
            f"checkpoint reports {cfg_id}; traces would be silently wrong")


@dataclass
class ModelOutputs:
    ell: np.ndarray          # confidence log-odds at masked positions
    conf: np.ndarray         # raw top-1 probability
    top1: np.ndarray         # argmax token id
    n_saturated: int = 0     # positions whose confidence hit the clip bound


def _to_logodds(conf: np.ndarray, eps: float) -> tuple[np.ndarray, int]:
    """Confidence -> log-odds, with the clipping recorded rather than hidden.

    Reported confidence saturates at 1.0 in low precision, which would send the
    log-odds to infinity.  The count of clipped positions is carried through and
    reported: it is a measurement limit of the instrument, not a nuisance.
    """
    c = np.asarray(conf, dtype=np.float64)
    n_sat = int(np.sum(c >= 1.0 - eps) + np.sum(c <= eps))
    c = np.clip(c, eps, 1.0 - eps)
    return np.log(c) - np.log1p(-c), n_sat


class DiffusionLM:
    """Interface every adapter implements."""

    mask_token_id: int
    vocab_size: int
    name: str

    def logits(self, input_ids):  # pragma: no cover - interface
        raise NotImplementedError

    def score_masked(self, input_ids, mask, eps: float = 1e-6) -> ModelOutputs:
        """One forward pass; read off confidence and top-1 at masked positions."""
        import torch

        with torch.no_grad():
            lg = self.logits(input_ids)
            sel = lg[0][torch.as_tensor(np.flatnonzero(mask), device=lg.device)]
            probs = torch.softmax(sel.float(), dim=-1)
            conf, top1 = probs.max(dim=-1)
            conf = conf.cpu().numpy()
            top1 = top1.cpu().numpy()
        ell, n_sat = _to_logodds(conf, eps)
        return ModelOutputs(ell=ell, conf=conf, top1=top1, n_saturated=n_sat)


@dataclass
class HFDiffusionLM(DiffusionLM):
    """Adapter for LLaDA / Dream style masked diffusion checkpoints."""

    name: str
    path: str
    mask_token_id: int
    precision: str = "4bit"      # "4bit" | "fp16" | "fp32" | "bf16"
    device: str = "cuda"
    model: object = field(default=None, repr=False)
    tokenizer: object = field(default=None, repr=False)
    vocab_size: int = 0

    def load(self) -> "HFDiffusionLM":
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self.device.startswith("cuda") and torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            cap, gpu = major + minor / 10.0, torch.cuda.get_device_name()
            if self.precision == "bf16" and major < 8:
                raise RuntimeError(
                    f"bfloat16 needs an Ampere-or-newer GPU; this device is compute "
                    f"capability {cap} ({gpu}). On Ada's u22 partition (RTX 2080 Ti, "
                    f"7.5) use precision='fp32' for the precision control, or request "
                    f"an Ampere partition."
                )
            if self.precision == "4bit" and cap < 7.5:
                raise RuntimeError(
                    f"bitsandbytes NF4 needs compute capability >= 7.5; this device is "
                    f"{cap} ({gpu}). u22 mixes GTX 1080 Ti (6.1) with RTX 2080 Ti (7.5) "
                    f"-- submit with '--constraint=2080ti'."
                )
            if cap < 7.5:
                print(f"[model] WARNING: running on {gpu} (compute {cap}); results are "
                      f"not comparable with 2080 Ti runs", flush=True)

        kwargs: dict = {"trust_remote_code": True}
        if self.precision == "4bit":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = {"": 0}
        else:
            kwargs["torch_dtype"] = {
                "fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16
            }[self.precision]

        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.path, **kwargs)
        if self.precision != "4bit":
            self.model = self.model.to(self.device)
        self.model.eval()
        self.vocab_size = int(getattr(self.model.config, "vocab_size", 0))
        return self

    def logits(self, input_ids):
        out = self.model(input_ids)
        return out.logits if hasattr(out, "logits") else out[0]


@dataclass
class ToyDiffusionLM(DiffusionLM):
    """A deterministic stand-in used to test the harness without a GPU.

    Confidence is a smooth function of a fixed per-token propensity plus
    position-specific noise, so the harness can be exercised end to end on CPU.
    """

    name: str = "toy"
    vocab_size: int = 64
    mask_token_id: int = 63
    seed: int = 0
    noise_scale: float = 1.0
    noise_dispersion: float = 0.0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.token_skill = rng.normal(0.0, 1.5, self.vocab_size)
        self.pos_log_sigma = rng.normal(0.0, self.noise_dispersion, 4096)

    def logits(self, input_ids):
        import torch

        ids = input_ids.detach().cpu().numpy()
        B, L = ids.shape
        rng = np.random.default_rng(self.seed + int(ids.sum()) % 100_000)
        out = rng.normal(0.0, 0.3, size=(B, L, self.vocab_size))
        sig = self.noise_scale * np.exp(self.pos_log_sigma[:L])
        for b in range(B):
            for pos in range(L):
                true_tok = int(ids[b, pos]) % self.vocab_size
                out[b, pos, true_tok] += self.token_skill[true_tok] + rng.normal(0, sig[pos])
        return torch.as_tensor(out, dtype=torch.float32)


def build_model(name: str, precision: str = "4bit", path: str | None = None,
                device: str = "cuda") -> DiffusionLM:
    if name == "toy":
        return ToyDiffusionLM()
    if name not in KNOWN_MODELS:
        raise KeyError(f"unknown model {name!r}; known: {sorted(KNOWN_MODELS)} or 'toy'")
    spec = KNOWN_MODELS[name]
    model = HFDiffusionLM(name=name, path=path or spec["path"],
                          mask_token_id=spec["mask_token_id"],
                          precision=precision, device=device).load()
    verify_mask_token(model)
    return model
