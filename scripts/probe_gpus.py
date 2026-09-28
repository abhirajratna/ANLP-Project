#!/usr/bin/env python3
"""Check that this job's GPUs actually work, before a long job depends on them.

A node can schedule GPU jobs while its GPUs are unusable: on 2026-09-28
gnode076 failed several users' jobs with ``CUDA unknown error`` and torch saw
zero devices.  Run plainly, this checks that torch sees at least N GPUs and
runs a matmul on each.  Run under torchrun, each rank also joins an NCCL
all-reduce, the collective DDP training starts with.  Exits non-zero on failure.

    python scripts/probe_gpus.py 2
    torchrun --nproc_per_node=2 scripts/probe_gpus.py 2
"""
from __future__ import annotations

import os
import socket
import sys


def main() -> int:
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    import torch

    host = socket.gethostname()
    n = torch.cuda.device_count()
    if n < want:
        print(f"[probe] FAIL on {host}: torch sees {n} GPU(s), need {want} "
              f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})", flush=True)
        return 1

    if "WORLD_SIZE" not in os.environ:
        for i in range(n):
            a = torch.randn(1024, 1024, device=f"cuda:{i}")
            val = float((a @ a).sum())
            if val != val:                                   # NaN
                print(f"[probe] FAIL on {host}: GPU {i} returned NaN", flush=True)
                return 1
            print(f"[probe] GPU {i} ok: {torch.cuda.get_device_name(i)}", flush=True)
        return 0

    import torch.distributed as dist

    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    x = torch.full((1,), float(rank + 1), device="cuda")
    dist.all_reduce(x)
    expect = world * (world + 1) / 2
    ok = abs(float(x) - expect) < 1e-6
    print(f"[probe] rank {rank}/{world} NCCL all-reduce {'ok' if ok else 'WRONG'} "
          f"({float(x)} vs {expect}) on {host} GPU {local}", flush=True)
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
