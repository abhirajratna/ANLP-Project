"""Model adapters.

Every audited checkpoint is reduced to one operation: given a batch of token
sequences in which some positions hold the mask token, return the logits.  That
is the single forward pass the whole design is built on.

Precision is a first-class argument because quantisation perturbs exactly the
logits under audit.  The guard below matters on this project's cluster: Ada's
u22 partition is RTX 2080 Ti (compute capability 7.5, Turing), which has no
bfloat16 support, so the planned bf16 replication cannot run there and must be
fp32 or move to an Ampere partition.

An 8B model does not fit one 11 GB card in fp16 (~16 GB of weights), so the
non-quantised paths accept ``device_map="auto"`` with a ``max_memory`` budget:
split across two cards, or offload to CPU for fp32.  fp16 has a narrow range,
and an overflow shows up as inf/NaN logits that would silently become NaN
confidences, so every forward pass is checked (:func:`check_finite_logits`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np


def set_deterministic(seed: int) -> dict:
    """Fix every seed and turn on PyTorch's deterministic settings (T1 step 4).

    Must run before the first CUDA call: cuBLAS reads its workspace setting when
    the context is created.  ``warn_only`` because some kernels (bitsandbytes)
    have no deterministic variant; the non-determinism floor measures what is
    left.  Returns the settings, for the trace meta.
    """
    import random

    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {"seed": int(seed), "use_deterministic_algorithms": True, "warn_only": True,
            "cudnn_benchmark": False,
            "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"]}


def check_finite_logits(logits, name: str = "model", precision: str = "na") -> None:
    """Refuse inf/NaN logits instead of letting them become NaN confidences."""
    import torch

    if not bool(torch.isfinite(logits).all()):
        n_bad = int((~torch.isfinite(logits)).sum())
        hint = (" fp16 overflows above 65504; use fp32 (with CPU offload if needed)"
                if precision == "fp16" else "")
        raise FloatingPointError(f"{name} ({precision}): {n_bad} non-finite logits.{hint}")


def parse_max_memory(spec: str | None) -> dict | None:
    """'0=10GiB,1=10GiB,cpu=60GiB' -> {0: '10GiB', 1: '10GiB', 'cpu': '60GiB'}."""
    if not spec:
        return None
    out: dict = {}
    for part in spec.split(","):
        dev, _, amount = part.partition("=")
        dev = dev.strip()
        out[int(dev) if dev.isdigit() else dev] = amount.strip()
    return out

# Mask token ids read from each checkpoint's own config.json (Instruct on
# 2026-09-18, Base on 2026-09-26), not copied from a paper. Getting one wrong would
# silently corrupt every trace: the model would be conditioned on an ordinary token
# where a mask was intended, and nothing downstream would flag it.
#
# ``revision`` pins the HF commit (checked 2026-09-26), so trust_remote_code runs
# the code the traces were validated against.  ``logit_shift`` is 1 for Dream:
# it keeps the autoregressive alignment of its Qwen initialisation, so the logits
# at position i predict token i+1, and its reference sampler
# (generation_utils.py) shifts them right by one before reading position i.
# Without the shift every Dream trace scores the wrong position.  ``variant`` is
# recorded in every trace; plan v2 uses Base on raw text and Instruct on tasks.
KNOWN_MODELS = {
    "llada-8b-base": {
        "path": "GSAI-ML/LLaDA-8B-Base",
        "revision": "0f2787f2d87eac5eed8a087d5ecd24277e6255b2",
        "variant": "base",
        "mask_token_id": 126336,       # verified: vocab 126464, 32 layers, d_model 4096
        "logit_shift": 0,
        "local_hint": None,
    },
    "llada-8b-instruct": {
        "path": "GSAI-ML/LLaDA-8B-Instruct",
        "revision": "08b83a6feb34df1a6011b80c3c00c7563e963b07",
        "variant": "instruct",
        "mask_token_id": 126336,
        "logit_shift": 0,
        # On Ada, 15 GB, group-readable -- but which variant it holds is not
        # recorded; detect_variant() checks before it is used.
        "local_hint": "/share1/ishita/llada_model",
    },
    "dream-7b-base": {
        "path": "Dream-org/Dream-v0-Base-7B",
        "revision": "6572adb5535263e4d1a337b56942ba48b6dee2a9",
        "variant": "base",
        "mask_token_id": 151666,       # verified: vocab 152064, 28 layers, hidden 3584
        "logit_shift": 1,
        "local_hint": None,
    },
    "dream-7b-instruct": {
        "path": "Dream-org/Dream-v0-Instruct-7B",
        "revision": "05334cb9faaf763692dcf9d8737c642be2b2a6ae",
        "variant": "instruct",
        "mask_token_id": 151666,
        "logit_shift": 1,
        "local_hint": None,
    },
}

# Proposal-era names.  They meant the Instruct checkpoints; kept as aliases so old
# job scripts keep working, but traces are labelled with the explicit key.
MODEL_ALIASES = {"llada-8b": "llada-8b-instruct", "dream-7b": "dream-7b-instruct"}


def detect_variant(path: str) -> str | None:
    """'instruct' if a local checkpoint ships a chat template, 'base' if not.

    Returns None when ``path`` is not a local directory (a hub id), since then
    the registry's pinned revision already fixes the variant.
    """
    import json
    from pathlib import Path

    d = Path(path)
    cfg = d / "tokenizer_config.json"
    if not d.is_dir() or not cfg.exists():
        return None
    try:
        has_template = bool(json.loads(cfg.read_text()).get("chat_template"))
    except (OSError, ValueError):
        return None
    return "instruct" if has_template else "base"


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
    ell: np.ndarray          # confidence log-odds at masked positions (exact, unclipped)
    conf: np.ndarray         # raw top-1 probability
    top1: np.ndarray         # argmax token id
    n_saturated: int = 0     # positions whose confidence rounds to 1 (or 0) at eps


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

    @property
    def input_device(self):
        """Where input ids must live for a forward pass."""
        import torch

        m = getattr(self, "model", None)
        if m is None:
            return torch.device("cpu")
        dev = getattr(m, "device", None)
        if dev is not None:
            return dev
        try:
            return next(m.parameters()).device
        except (AttributeError, StopIteration):
            return torch.device("cpu")

    def score_masked(self, input_ids, mask, eps: float = 1e-6) -> ModelOutputs:
        """One forward pass; read off confidence and top-1 at masked positions.

        The mask token itself is excluded from the prediction (its logit is set
        to -inf, the SUBS parameterisation): a masked slot can never be filled
        with [MASK].  The log-odds are computed exactly as
        ell = z_top1 - logsumexp(z_others) in float32 on the logits, so confidence
        that rounds to 1.0 in low precision still gets a finite, ordered ell;
        ``n_saturated`` counts those positions for the record.
        """
        import torch

        with torch.no_grad():
            lg = self.logits(input_ids)
            sel = lg[0][torch.as_tensor(np.flatnonzero(mask), device=lg.device)].float()
            if 0 <= int(self.mask_token_id) < sel.shape[-1]:
                sel[:, int(self.mask_token_id)] = float("-inf")
            z1, top1 = sel.max(dim=-1)
            others = sel.scatter(1, top1[:, None], float("-inf"))
            ell_t = z1 - torch.logsumexp(others, dim=-1)
            conf_t = torch.softmax(sel, dim=-1).gather(1, top1[:, None])[:, 0]
            ell = ell_t.double().cpu().numpy()
            conf = conf_t.cpu().numpy()
            top1 = top1.cpu().numpy()
        n_sat = int(np.sum(conf >= 1.0 - eps) + np.sum(conf <= eps))
        return ModelOutputs(ell=ell, conf=conf, top1=top1, n_saturated=n_sat)


@dataclass
class HFDiffusionLM(DiffusionLM):
    """Adapter for LLaDA / Dream style masked diffusion checkpoints."""

    name: str
    path: str
    mask_token_id: int
    precision: str = "4bit"      # "4bit" | "fp16" | "fp32" | "bf16"
    device: str = "cuda"
    revision: str | None = None
    variant: str | None = None
    logit_shift: int = 0
    device_map: str | None = None     # "auto": split across GPUs / offload to CPU
    max_memory: dict | None = None    # per-device budget for device_map="auto"
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
            if self.device_map:
                # Both LLaDA and Dream declare _no_split_modules, which is what
                # device_map="auto" needs to place whole blocks per device.
                kwargs["device_map"] = self.device_map
                if self.max_memory:
                    kwargs["max_memory"] = self.max_memory

        found = detect_variant(self.path)
        if found is not None and self.variant is not None and found != self.variant:
            raise RuntimeError(
                f"{self.name}: registry says variant={self.variant!r} but the checkpoint at "
                f"{self.path} looks like {found!r} (chat template "
                f"{'present' if found == 'instruct' else 'absent'}); pick the matching model key")
        if found is not None:
            self.variant = found
        # A local directory is whatever was copied there; the pinned revision
        # applies only to hub downloads.
        rev = None if found is not None else self.revision
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True,
                                                       revision=rev)
        self.model = AutoModel.from_pretrained(self.path, revision=rev, **kwargs)
        if self.precision != "4bit" and not self.device_map:
            self.model = self.model.to(self.device)
        self.model.eval()
        self.vocab_size = int(getattr(self.model.config, "vocab_size", 0))
        placement = getattr(self.model, "hf_device_map", None)
        if placement:
            devs = sorted({str(v) for v in placement.values()})
            print(f"[model] {self.name}: {self.precision} over {devs}", flush=True)
        return self

    def logits(self, input_ids):
        out = self.model(input_ids)
        lg = out.logits if hasattr(out, "logits") else out[0]
        if self.logit_shift:
            import torch
            lg = torch.cat([lg[:, :1], lg[:, :-1]], dim=1)
        return lg


@dataclass
class LocalMDLM(DiffusionLM):
    """The 110M model this project trains: the controlled instrument.

    Unlike the 8B checkpoints, its training set is known and the calibration
    corpus was removed from it, which is what lets the audit make claims about
    the *level* of calibration error rather than only its gradient.
    """

    name: str = "local-110m"
    path: str = ""                 # final.pt written by c3.training.train
    device: str = "cuda"
    precision: str = "fp32"        # small enough that fp32 costs nothing
    variant: str = "base"
    revision: str | None = None
    use_ema: bool = True           # MDLM evaluates with the EMA weights
    model: object = field(default=None, repr=False)
    tokenizer: object = field(default=None, repr=False)
    mask_token_id: int = 50257
    vocab_size: int = 50258

    def load(self) -> "LocalMDLM":
        import torch
        from transformers import AutoTokenizer

        from ..training.model import MaskedDiffusionLM, ModelConfig

        state = torch.load(self.path, map_location="cpu", weights_only=False)
        # The checkpoint stores "model_cfg"; train_summary.json spells the same
        # thing "model_config". Accept either rather than depending on which
        # artefact the caller happened to look at.
        raw_cfg = state.get("model_cfg") or state.get("model_config")
        if raw_cfg is None:
            raise KeyError(f"{self.path} has no model config (keys: {sorted(state)})")
        cfg = ModelConfig(**raw_cfg)
        self.mask_token_id = cfg.mask_token_id
        self.vocab_size = cfg.vocab_size
        model = MaskedDiffusionLM(cfg)
        weights = state.get("model_ema") if self.use_ema and state.get("model_ema") else state["model"]
        self.weights = "ema" if weights is not state["model"] else "raw"
        model.load_state_dict(weights)
        dtype = {"fp16": torch.float16, "fp32": torch.float32,
                 "bf16": torch.bfloat16}[self.precision]
        self.model = model.to(self.device, dtype=dtype).eval()
        # The 110M model is trained on GPT-2 BPE; the mask id sits one past the
        # tokenizer's vocabulary and is never produced by tokenisation.
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.revision = state.get("config_hash")
        print(f"[model] local-110m: {state['step']} steps, "
              f"{state['tokens_seen']/1e9:.2f}B tokens, {cfg.n_params/1e6:.0f}M params, "
              f"{self.weights} weights", flush=True)
        return self

    def logits(self, input_ids):
        return self.model(input_ids)


@dataclass
class ToyDiffusionLM(DiffusionLM):
    """A deterministic stand-in used to test the harness without a GPU.

    At a masked position the toy boosts the logit of the *true* token by a fixed
    per-token skill plus position-specific noise, so its top-1 is right with a
    probability that tracks its confidence -- the harness tests then exercise
    agreement and calibration, not just shapes.  It learns the true tokens through
    :meth:`set_reference`, which ``collect_document`` calls for models that have
    it (only this toy).  Each row's noise is seeded by hashing the row, so a
    row's output does not depend on what else is in the batch.
    """

    name: str = "toy"
    vocab_size: int = 64
    mask_token_id: int = 63
    seed: int = 0
    noise_scale: float = 1.0
    noise_dispersion: float = 0.0
    precision: str = "fp32"       # fp16/bf16 round the logits, as a half-precision head would
    variant: str = "toy"

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.token_skill = rng.normal(1.0, 1.5, self.vocab_size)
        self.pos_log_sigma = rng.normal(0.0, self.noise_dispersion, 4096)
        self.reference: np.ndarray | None = None

    def set_reference(self, x0: np.ndarray) -> None:
        self.reference = np.asarray(x0, dtype=np.int64) % self.vocab_size

    def logits(self, input_ids):
        import hashlib

        import torch

        ids = input_ids.detach().cpu().numpy()
        B, L = ids.shape
        sig = self.noise_scale * np.exp(self.pos_log_sigma[:L])
        out = np.empty((B, L, self.vocab_size))
        for b in range(B):
            h = int.from_bytes(hashlib.blake2b(ids[b].tobytes(), digest_size=8).digest(), "little")
            rng = np.random.default_rng([self.seed, h])
            out[b] = rng.normal(0.0, 0.3, size=(L, self.vocab_size))
            for pos in range(L):
                tok = int(ids[b, pos])
                if (tok == self.mask_token_id and self.reference is not None
                        and pos < self.reference.size):
                    tok = int(self.reference[pos])
                tok %= self.vocab_size
                out[b, pos, tok] += self.token_skill[tok] + rng.normal(0, sig[pos])
        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        return torch.as_tensor(out, dtype=torch.float32).to(dtype[self.precision])


def build_model(name: str, precision: str = "4bit", path: str | None = None,
                device: str = "cuda", device_map: str | None = None,
                max_memory: dict | None = None) -> DiffusionLM:
    if name == "toy":
        # 4-bit is meaningless for the toy; it stands in as full precision.
        return ToyDiffusionLM(precision="fp32" if precision == "4bit" else precision)
    name = MODEL_ALIASES.get(name, name)
    if name == "local-110m":
        if not path:
            raise ValueError("local-110m needs --model-path pointing at final.pt")
        return LocalMDLM(path=path, device=device,
                         precision=("fp32" if precision == "4bit" else precision)).load()
    if name not in KNOWN_MODELS:
        raise KeyError(f"unknown model {name!r}; known: {sorted(KNOWN_MODELS)} or 'toy'")
    spec = KNOWN_MODELS[name]
    if device_map and precision == "4bit":
        raise ValueError("device_map applies to fp16/fp32/bf16; 4-bit loads on one GPU")
    model = HFDiffusionLM(name=name, path=path or spec["path"],
                          mask_token_id=spec["mask_token_id"],
                          precision=precision, device=device,
                          revision=spec.get("revision"), variant=spec.get("variant"),
                          logit_shift=int(spec.get("logit_shift", 0)),
                          device_map=device_map, max_memory=max_memory).load()
    verify_mask_token(model)
    return model
