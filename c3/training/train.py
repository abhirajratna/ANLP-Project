"""Training loop for the 110M model, built to be interrupted.

Ada's ``medium`` QOS caps a job at 4 days, so a run that needs longer is not one
long job but a chain of requeued ones.  Every piece of state that would change
the result if lost -- weights, optimizer moments, the AMP scaler, the data
generator, the step counter -- is checkpointed together and reloaded on start,
and checkpoints are written to a temporary name and renamed, so a job killed
mid-write leaves the previous checkpoint intact rather than a truncated one.

Precision is fp16 with a gradient scaler, not bf16: the 2080 Ti is Turing and
has no bf16 support.  That is a hardware constraint, not a preference.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .data import TokenShardDataset
from .model import MaskedDiffusionLM, ModelConfig, mdlm_loss


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
    max_tokens: int = 6_000_000_000
    weight_decay: float = 0.02
    grad_clip: float = 1.0
    log_every: int = 50
    ckpt_every: int = 1000
    seed: int = 20260917
    wandb_project: str | None = None


def _lr_at(step: int, total_steps: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    prog = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    prog = min(1.0, max(0.0, prog))
    return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


def _save(path: Path, **state) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)          # atomic: a killed job never truncates the live ckpt


def train(cfg: TrainConfig, model_cfg: ModelConfig | None = None) -> Path:
    model_cfg = model_cfg or ModelConfig(max_len=cfg.seq_len)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint.pt"

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

    if device.startswith("cuda"):
        major, _ = torch.cuda.get_device_capability(local_rank)
        if is_main and major < 8:
            print(f"[train] {torch.cuda.get_device_name(local_rank)} is compute {major}.x: "
                  f"no bf16, using fp16 + GradScaler", flush=True)

    torch.manual_seed(cfg.seed + rank)
    model = MaskedDiffusionLM(model_cfg).to(device)
    raw = model
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    decay = [p for n, p in raw.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in raw.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))

    tokens_per_step = cfg.batch_size * cfg.grad_accum * cfg.seq_len * world
    total_steps = max(1, cfg.max_tokens // tokens_per_step)
    data = TokenShardDataset(cfg.index_path, cfg.seq_len, seed=cfg.seed + rank)

    step, tokens_seen = 0, 0
    if ckpt_path.exists():
        # map_location="cpu", not the GPU. The checkpoint carries fp32 optimizer
        # moments as well as weights (~1.5 GB here); loading it straight to the
        # device holds that on top of the model, optimizer and activations, and
        # the resumed job dies in backward with a CUDA OOM while the identical
        # fresh run trains fine. load_state_dict copies CPU tensors into the
        # existing device tensors, and the optimizer state follows its params.
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        raw.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        scaler.load_state_dict(state["scaler"])
        step, tokens_seen = state["step"], state["tokens_seen"]
        data.rng = np.random.default_rng(state["data_seed"] + step)
        del state
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        if is_main:
            print(f"[train] resumed from {ckpt_path} at step {step} "
                  f"({tokens_seen/1e9:.2f}B tokens)", flush=True)
    elif is_main:
        print(f"[train] fresh start: {total_steps} steps x {tokens_per_step} tokens/step "
              f"= {cfg.max_tokens/1e9:.1f}B tokens", flush=True)

    run = None
    if is_main and cfg.wandb_project:
        try:
            import wandb
            run = wandb.init(project=cfg.wandb_project, config={**asdict(cfg), **asdict(model_cfg)},
                             resume="allow", id=out_dir.name)
        except Exception as exc:                      # wandb must never kill a run
            print(f"[train] wandb disabled: {exc}", flush=True)

    model.train()
    t0 = time.time()
    tokens_at_start = tokens_seen   # throughput must exclude tokens from a prior job
    while step < total_steps:
        lr = _lr_at(step, total_steps, cfg)
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad(set_to_none=True)

        agg_loss = 0.0
        for micro in range(cfg.grad_accum):
            x = torch.from_numpy(data.batch(cfg.batch_size)).to(device, non_blocking=True)
            sync = (micro == cfg.grad_accum - 1)
            ctx = model.no_sync() if (ddp and not sync) else torch.enable_grad()
            with ctx:
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                    loss, stats = mdlm_loss(model, x, model_cfg.mask_token_id)
                scaler.scale(loss / cfg.grad_accum).backward()
            agg_loss += float(loss.detach()) / cfg.grad_accum

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(raw.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        step += 1
        tokens_seen += tokens_per_step

        if is_main and step % cfg.log_every == 0:
            # Measure against tokens produced by *this* process: dividing the
            # cumulative count by this process's elapsed time reports a throughput
            # several times too high after a resume.
            tps = (tokens_seen - tokens_at_start) / max(1e-9, time.time() - t0)
            msg = (f"[train] step {step}/{total_steps} loss {agg_loss:.4f} "
                   f"ce/tok {stats['ce_per_masked_token']:.3f} lr {lr:.2e} "
                   f"{tokens_seen/1e9:.3f}B tok {tps/1e3:.1f}k tok/s")
            print(msg, flush=True)
            if run:
                run.log({"loss": agg_loss, "ce_per_masked_token": stats["ce_per_masked_token"],
                         "lr": lr, "tokens": tokens_seen, "tokens_per_s": tps}, step=step)

        if is_main and (step % cfg.ckpt_every == 0 or step == total_steps):
            _save(ckpt_path, model=raw.state_dict(), opt=opt.state_dict(),
                  scaler=scaler.state_dict(), step=step, tokens_seen=tokens_seen,
                  data_seed=cfg.seed + rank, cfg=asdict(cfg), model_cfg=asdict(model_cfg))
            print(f"[train] checkpoint at step {step}", flush=True)

    if is_main:
        final = out_dir / "final.pt"
        _save(final, model=raw.state_dict(), step=step, tokens_seen=tokens_seen,
              cfg=asdict(cfg), model_cfg=asdict(model_cfg))
        (out_dir / "train_summary.json").write_text(json.dumps(
            {"steps": step, "tokens_seen": tokens_seen, "config": asdict(cfg),
             "model_config": asdict(model_cfg)}, indent=2))
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
