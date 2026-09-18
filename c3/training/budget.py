"""Compute budgeting, in the units this project's cluster actually has.

Ada's ``u22`` partition is RTX 2080 Ti: Turing, 11 GB, no bfloat16.  The
``medium`` QOS allows 4 GPUs and a 4-day walltime.  Those numbers, not a
generic A100 figure, decide what the 110M arm can be.

The published MDLM recipe is roughly 5.2e11 training tokens.  At the throughput
below that is about two months of wall-clock on this allocation, so it is not
available and the token budget has to be chosen deliberately rather than
inherited.  What *is* available is one to three passes over the 10BT subset,
which is the range these helpers are for.
"""

from __future__ import annotations

from dataclasses import dataclass

SECONDS_PER_DAY = 86_400

# Measured, not assumed.  Benchmarked 2026-09-18 on Ada u22: 124M params,
# seq_len 256, batch 32 x grad_accum 4 per GPU, fp16 AMP, 2x RTX 2080 Ti ->
# 74.7k tokens/s sustained, i.e. 37.3k tok/s/GPU.  With FLOPs ~= 6*N*D that is
# 6 * 124e6 * 37.3e3 = 2.77e13 = 27.7 TFLOP/s per GPU (about 52% of the 53.8
# TFLOP/s fp16 tensor-core peak, which is a plausible MFU for this shape).
#
# An earlier planning figure of 15 TFLOP/s was a guess and was too pessimistic by
# ~1.8x.  A first smoke test appeared to confirm it at 13.7k tok/s, but that run
# checkpointed every 10 steps and was measuring ~1.4 GB of disk writes rather
# than training -- worth remembering before trusting a throughput number taken
# from a run that was doing anything else.
TFLOPS_2080TI_FP16_SUSTAINED = 27.7
MEASURED_TOKENS_PER_S_PER_GPU = 37_300


@dataclass
class TrainBudget:
    n_params: float
    n_tokens: float
    n_gpus: int
    tflops_per_gpu: float = TFLOPS_2080TI_FP16_SUSTAINED

    @property
    def flops(self) -> float:
        """Forward+backward ~= 6 * params * tokens."""
        return 6.0 * self.n_params * self.n_tokens

    @property
    def gpu_seconds(self) -> float:
        return self.flops / (self.tflops_per_gpu * 1e12)

    @property
    def wall_seconds(self) -> float:
        return self.gpu_seconds / max(1, self.n_gpus)

    @property
    def wall_days(self) -> float:
        return self.wall_seconds / SECONDS_PER_DAY

    @property
    def slurm_jobs_at_4day(self) -> int:
        import math
        return max(1, math.ceil(self.wall_days / 4.0))

    def report(self) -> str:
        return (
            f"params={self.n_params/1e6:.0f}M  tokens={self.n_tokens/1e9:.1f}B  "
            f"gpus={self.n_gpus}\n"
            f"  {self.flops:.3e} FLOPs -> {self.gpu_seconds/3600:.0f} GPU-hours"
            f" -> {self.wall_days:.2f} days wall-clock\n"
            f"  needs {self.slurm_jobs_at_4day} SLURM job(s) at the 4-day medium-QOS limit"
        )


def compare_budgets(n_params: float = 124e6, n_gpus: int = 4) -> str:
    rows = [("MDLM published recipe", 5.2e11), ("3 epochs of 10BT", 3.0e10),
            ("1 epoch of 10BT", 1.0e10), ("project default", 6.0e9),
            ("smoke test", 5.0e8)]
    out = [f"Budget on {n_gpus} x RTX 2080 Ti at {TFLOPS_2080TI_FP16_SUSTAINED} TFLOP/s sustained:"]
    for name, toks in rows:
        b = TrainBudget(n_params, toks, n_gpus)
        out.append(f"  {name:24s} {toks/1e9:6.1f}B tok  {b.wall_days:6.2f} days  "
                   f"{b.slurm_jobs_at_4day:2d} job(s)")
    return "\n".join(out)
