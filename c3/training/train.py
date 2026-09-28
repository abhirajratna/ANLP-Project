"""Training loop for the 110M model, built to be interrupted.

Ada's ``medium`` QOS caps a job at 4 days, so a run that needs longer is not one
long job but a chain of requeued ones.  Every piece of state that would change
the result if lost -- weights, EMA weights, optimizer moments, the AMP scaler,
the step counter -- is checkpointed together and reloaded on start, and
checkpoints are written to a temporary name and renamed, so a job killed
mid-write leaves the previous checkpoint intact rather than a truncated one.

Randomness on resume is re-derived, not restored: every rank reseeds its data
sampler and torch generator from (seed, rank, step), so ranks never draw the
same batches and a resumed run does not replay the start of the mask stream.
A checkpoint also records a hash of the configuration and of the shard index; a
job that finds a checkpoint from a *different* run refuses to resume it instead
of silently continuing someone's trial weights.

Precision is fp16 with a gradient scaler, not bf16: the 2080 Ti is Turing and
has no bf16 support.  That is a hardware constraint, not a preference.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .data import DocWindowDataset
from .model import (MaskedDiffusionLM, ModelConfig, ar_loss, ar_nll_eval, mdlm_loss,
                    nelbo_eval)

# Fields that may differ between the job that wrote a checkpoint and the job
# resuming it without making it a different run.
_RESUMABLE_FIELDS = {"out_dir", "log_every", "ckpt_every", "wandb_project", "wandb_entity", "eval_every",
                     "eval_batches", "index_path"}


@dataclass
class TrainConfig:
    index_path: str
    out_dir: str
    seq_len: int = 256
    batch_size: int = 32              # per GPU; 11 GB at seq_len 256
    grad_accum: int = 4
    lr: float = 3e-4
    min_lr_frac: float = 0.1
    warmup_steps: int = 2000
    max_tokens: int = 10_000_000_000  # real (non-pad) tokens; plan v2: 10-20B
    weight_decay: float = 0.02
    grad_clip: float = 1.0
    log_every: int = 50
    ckpt_every: int = 1000
    eval_every: int = 1000
    eval_batches: int = 16
    ema_decay: float = 0.9999
    window_mode: str = "chunks"       # "chunks" (whole corpus) | "prefix" (first 256 only)
    objective: str = "mdlm"           # "mdlm" | "ar" (the autoregressive reference)
    seed: int = 20260917
    init_from: str | None = None      # weights-only start (fine-tuning, T9)
    run_name: str | None = None       # W&B id; default: out_dir name + config hash
    wandb_project: str | None = None
    # The team (wandb "entity") the run belongs to. Left unset, runs land in the
    # submitter's personal namespace, which is not where a shared project's runs
    # should live.
    wandb_entity: str | None = None


def config_hash(cfg: TrainConfig, model_cfg: ModelConfig, index_path: str | None = None) -> str:
    """Hash of everything that defines the run (not where it is written)."""
    d = {k: v for k, v in asdict(cfg).items() if k not in _RESUMABLE_FIELDS}
    d["model"] = asdict(model_cfg)
    if index_path and Path(index_path).exists():
        idx = json.loads(Path(index_path).read_text())
        d["data"] = {k: idx.get(k) for k in ("total_tokens", "documents", "format", "val_fraction")}
    return hashlib.blake2b(json.dumps(d, sort_keys=True).encode(), digest_size=8).hexdigest()


def _lr_at(step: int, total_steps: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    prog = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    prog = min(1.0, max(0.0, prog))
    return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


def _save(path: Path, sidecar: dict | None = None, **state) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)          # atomic: a killed job never truncates the live ckpt
    if sidecar is not None:
        # Small JSON next to the checkpoint, so the job script can compare the
        # step of two copies without loading 1.5 GB (see slurm/train_110m.sbatch).
        side = path.with_suffix(".json")
        tmp_side = side.with_suffix(".json.tmp")
        tmp_side.write_text(json.dumps(sidecar))
        os.replace(tmp_side, side)


def _seed_everything(cfg: TrainConfig, rank: int, step: int, data) -> None:
    key = [cfg.seed, rank, step]
    data.reseed(*key)
    torch.manual_seed(int(np.random.SeedSequence(key).generate_state(1)[0]))


class EMA:
    """Exponential moving average of the weights (MDLM evaluates with EMA)."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().float().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in self.shadow.items()})


def evaluate(model: torch.nn.Module, batches, mask_token_id: int, device,
             objective: str = "mdlm") -> dict:
    """Validation NELBO per real token and the perplexity bound exp(NELBO).

    For the AR reference the same keys hold its exact NLL and perplexity.
    """
    model.eval()
    tot = tok = 0.0
    for i, (ids, valid) in enumerate(batches):
        x = torch.from_numpy(ids).to(device)
        v = torch.from_numpy(valid).to(device)
        if objective == "ar":
            s, n = ar_nll_eval(model, x, v)
        else:
            s, n = nelbo_eval(model, x, mask_token_id, v, seed=i)
        tot += s
        tok += n
    model.train()
    nelbo = tot / max(tok, 1.0)
    return {"val_nelbo": nelbo, "val_ppl_bound": float(math.exp(min(nelbo, 50.0))),
            "val_tokens": tok}


def train(cfg: TrainConfig, model_cfg: ModelConfig | None = None) -> Path:
    model_cfg = model_cfg or ModelConfig(max_len=cfg.seq_len, causal=cfg.objective == "ar")
    if cfg.objective not in {"mdlm", "ar"}:
        raise ValueError(f"objective must be 'mdlm' or 'ar', got {cfg.objective!r}")
    if (cfg.objective == "ar") != model_cfg.causal:
        raise ValueError("objective 'ar' needs ModelConfig(causal=True), and 'mdlm' causal=False")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint.pt"
    chash = config_hash(cfg, model_cfg, cfg.index_path)

    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        torch.distributed.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        rank, local_rank, world = 0, 0, 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
    is_main = rank == 0
    use_cuda = device.startswith("cuda")

    if use_cuda:
        major, _ = torch.cuda.get_device_capability(local_rank)
        if is_main and major < 8:
            print(f"[train] {torch.cuda.get_device_name(local_rank)} is compute {major}.x: "
                  f"no bf16, using fp16 + GradScaler", flush=True)

    torch.manual_seed(cfg.seed + rank)
    model = MaskedDiffusionLM(model_cfg).to(device)
    raw = model
    if cfg.init_from and not ckpt_path.exists():
        init = torch.load(cfg.init_from, map_location="cpu", weights_only=False)
        raw.load_state_dict(init.get("model_ema") or init["model"])
        if is_main:
            print(f"[train] initialised weights from {cfg.init_from} "
                  f"({'EMA' if init.get('model_ema') else 'raw'} weights)", flush=True)
        del init
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    decay = [p for n, p in raw.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in raw.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    data = DocWindowDataset(cfg.index_path, cfg.seq_len, seed=cfg.seed + rank,
                            mode=cfg.window_mode, pad_id=50256)
    # Steps are planned from the expected real tokens per window, so the token
    # budget counts text, not padding.
    mean_real = data.real_tokens / data.n_windows
    tokens_per_step_est = cfg.batch_size * cfg.grad_accum * mean_real * world
    total_steps = max(1, int(cfg.max_tokens // tokens_per_step_est))

    val_batches = None
    if is_main and cfg.eval_every:
        try:
            val = DocWindowDataset(cfg.index_path, cfg.seq_len, mode="prefix", split="val",
                                   pad_id=50256)
            val_batches = list(val.fixed_batches(cfg.eval_batches, cfg.batch_size, seed=cfg.seed))
        except ValueError as exc:
            print(f"[train] no validation shards ({exc}); validation disabled", flush=True)

    ema = EMA(raw, cfg.ema_decay) if is_main and cfg.ema_decay else None

    step, tokens_seen = 0, 0
    if ckpt_path.exists():
        # map_location="cpu", not the GPU. The checkpoint carries fp32 optimizer
        # moments as well as weights (~1.5 GB here); loading it straight to the
        # device holds that on top of the model, optimizer and activations, and
        # the resumed job dies in backward with a CUDA OOM while the identical
        # fresh run trains fine. load_state_dict copies CPU tensors into the
        # existing device tensors, and the optimizer state follows its params.
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        found = state.get("config_hash")
        if found is None and os.environ.get("C3_ALLOW_LEGACY_RESUME") != "1":
            raise RuntimeError(
                f"{ckpt_path} was written by the proposal-era trainer (no config hash), e.g. a "
                f"trial run.  Start the production run in a new --out-dir; set "
                f"C3_ALLOW_LEGACY_RESUME=1 only to deliberately continue that checkpoint.")
        if found is not None and found != chash:
            raise RuntimeError(
                f"{ckpt_path} belongs to a different run (config hash {found} vs {chash}).  "
                f"Use a new --out-dir for a new run; resuming it would continue another "
                f"configuration's weights.")
        raw.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        scaler.load_state_dict(state["scaler"])
        if ema is not None and state.get("model_ema") is not None:
            # The state was loaded to the CPU (see above), and unlike the model and
            # optimizer, nothing copies the EMA back: move it to the model's device,
            # or the first ema.update() after a resume mixes cuda and cpu tensors.
            ema.shadow = {k: v.to(device, dtype=torch.float32)
                          for k, v in state["model_ema"].items()}
        step, tokens_seen = state["step"], state["tokens_seen"]
        total_steps = state.get("total_steps", total_steps)
        del state
        if use_cuda:
            torch.cuda.empty_cache()
        if is_main:
            print(f"[train] resumed from {ckpt_path} at step {step} "
                  f"({tokens_seen/1e9:.2f}B tokens)", flush=True)
    elif is_main:
        print(f"[train] fresh start: {total_steps} steps x ~{tokens_per_step_est:.0f} real tokens/step "
              f"= {cfg.max_tokens/1e9:.1f}B tokens; window mode {cfg.window_mode}, "
              f"{data.n_windows:,} windows; config {chash}", flush=True)
    _seed_everything(cfg, rank, step, data)

    run = None
    if is_main and cfg.wandb_project:
        try:
            import wandb

            from ..provenance import git_commit
            run = wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity,
                             config={**asdict(cfg), **asdict(model_cfg), "config_hash": chash,
                                     "git_commit": git_commit()[0]},
                             resume="allow", id=cfg.run_name or f"{out_dir.name}-{chash}")
        except Exception as exc:                      # wandb must never kill a run
            print(f"[train] wandb disabled: {exc}", flush=True)

    history = out_dir / "history.jsonl"

    def log_row(row: dict) -> None:
        if is_main:
            with open(history, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            if run:
                run.log(row, step=row["step"])

    def checkpoint(final: bool = False) -> None:
        state = dict(model=raw.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(),
                     model_ema=(ema.shadow if ema is not None else None), step=step,
                     tokens_seen=tokens_seen, total_steps=total_steps, config_hash=chash,
                     cfg=asdict(cfg), model_cfg=asdict(model_cfg))
        _save(ckpt_path, sidecar={"step": step, "tokens_seen": tokens_seen,
                                  "config_hash": chash}, **state)
        print(f"[train] checkpoint at step {step}", flush=True)

    model.train()
    t0 = time.time()
    tokens_at_start = tokens_seen   # throughput must exclude tokens from a prior job
    while step < total_steps:
        lr = _lr_at(step, total_steps, cfg)
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad(set_to_none=True)

        agg_loss, real = 0.0, 0
        for micro in range(cfg.grad_accum):
            ids, valid = data.batch(cfg.batch_size)
            x = torch.from_numpy(ids).to(device, non_blocking=True)
            v = torch.from_numpy(valid).to(device, non_blocking=True)
            sync = (micro == cfg.grad_accum - 1)
            ctx = model.no_sync() if (ddp and not sync) else torch.enable_grad()
            with ctx:
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_cuda):
                    if cfg.objective == "ar":
                        loss, stats = ar_loss(model, x, v)
                    else:
                        loss, stats = mdlm_loss(model, x, model_cfg.mask_token_id, v)
                scaler.scale(loss / cfg.grad_accum).backward()
            agg_loss += float(loss.detach()) / cfg.grad_accum
            real += stats["real_tokens"]

        scaler.unscale_(opt)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(raw.parameters(), cfg.grad_clip))
        scaler.step(opt)
        scaler.update()
        if ema is not None:
            ema.update(raw)
        step += 1
        tokens_seen += real * world

        if is_main and step % cfg.log_every == 0:
            # Measure against tokens produced by *this* process: dividing the
            # cumulative count by this process's elapsed time reports a throughput
            # several times too high after a resume.
            tps = (tokens_seen - tokens_at_start) / max(1e-9, time.time() - t0)
            print(f"[train] step {step}/{total_steps} loss {agg_loss:.4f} "
                  f"ce/tok {stats['ce_per_masked_token']:.3f} lr {lr:.2e} gnorm {grad_norm:.2f} "
                  f"{tokens_seen/1e9:.3f}B tok {tps/1e3:.1f}k tok/s", flush=True)
            log_row({"step": step, "loss": agg_loss, "ce_per_masked_token": stats["ce_per_masked_token"],
                     "lr": lr, "grad_norm": grad_norm, "loss_scale": float(scaler.get_scale()),
                     "tokens": tokens_seen, "tokens_per_s": tps})

        if is_main and val_batches and (step % cfg.eval_every == 0 or step == total_steps):
            ev = evaluate(raw, val_batches, model_cfg.mask_token_id, device, cfg.objective)
            if ema is not None:
                live = copy.deepcopy(raw.state_dict())
                ema.copy_to(raw)
                ev_ema = evaluate(raw, val_batches, model_cfg.mask_token_id, device, cfg.objective)
                raw.load_state_dict(live)
                ev.update({f"ema_{k}": v for k, v in ev_ema.items() if k != "val_tokens"})
            print(f"[train] step {step} val NELBO {ev['val_nelbo']:.4f} "
                  f"(ppl <= {ev['val_ppl_bound']:.1f})"
                  + (f", EMA {ev['ema_val_nelbo']:.4f}" if "ema_val_nelbo" in ev else ""), flush=True)
            log_row({"step": step, "tokens": tokens_seen, **ev})

        if is_main and (step % cfg.ckpt_every == 0 or step == total_steps):
            checkpoint()

    if is_main:
        final = out_dir / "final.pt"
        _save(final, model=raw.state_dict(),
              model_ema=(ema.shadow if ema is not None else None), step=step,
              tokens_seen=tokens_seen, config_hash=chash, cfg=asdict(cfg),
              model_cfg=asdict(model_cfg))
        (out_dir / "train_summary.json").write_text(json.dumps(
            {"steps": step, "tokens_seen": tokens_seen, "config_hash": chash,
             "config": asdict(cfg), "model_config": asdict(model_cfg)}, indent=2))
        print(f"[train] done -> {final}", flush=True)
    if ddp:
        # NCCL teardown can fail on its own -- this run lost its last 43 steps to
        #   DistBackendError ... ncclUnhandledCudaError ... Cuda failure 'out of memory'
        # raised inside _shutdown_backend after training had finished. Letting a
        # shutdown fault propagate throws away a completed run, so it is logged
        # and swallowed: by this point the weights are already saved.
        try:
            torch.cuda.empty_cache()
            torch.distributed.destroy_process_group()
        except Exception as exc:
            print(f"[train] non-fatal error during NCCL teardown: {exc}", flush=True)
    return out_dir / "final.pt"
