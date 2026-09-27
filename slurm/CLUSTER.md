# Ada: what this project actually has

Measured on 2026-09-18 from the `ojas.k` account. These numbers, not generic
A100 assumptions, drive every budget decision in the repo.

| Fact | Value | Consequence |
|---|---|---|
| Account / QOS | `research` / `medium` | 4 GPUs, 40 CPUs, 125 GB RAM, **4-day walltime**, 8 queued jobs |
| Partition available | `u22` only | `ihub`, `plafnet2`, `rrc` need other accounts |
| GPU | **u22 is MIXED**: nodes with feature `2080ti` are RTX 2080 Ti (11 GB, compute **7.5**, Turing); nodes advertising **no feature are GTX 1080 Ti** (11 GB, compute **6.1**, Pascal) | **always submit `--constraint=2080ti`**; on 2080 Ti there is still **no bfloat16** — use fp16 + GradScaler, or fp32 |
| Node-local scratch | `/scratch`, 1.8 TB | where token shards and HF caches belong |
| **`/share1` is login-node-only** | not *mounted* on compute nodes, but reachable from them by `ssh`/`rsync` back to `ada` | job StdOut and anything opened by path must be on `/home2`; large blobs live on `/share1` and are **pulled to node-local SSD at job start** (the pattern in `run-llama.sh`) |
| `/share1/$USER` free space | ~76 GB, 3,000-inode quota | the right home for tarballs, checkpoints and model weights — few large files, never many small ones |
| Node-local SSD | `/ssd_scratch`, 880 GB | staging target; persists on a node between jobs, so only the first job on a node pays the transfer |
| Compute-visible persistent | `/home2/$USER` (NFS), 30 GB quota, 25.7 GB used | **~5 GB free** — the binding storage constraint; `~/.cache` holds 5.3 GB of reclaimable pip/HF cache |
| Inode quota on `/share1` | 3,000 files | the built env is **47,361 files**, so it ships as a single tarball |
| `/share7`, `/cvit_contrib_packages` | NFS, compute-visible, **read-only** | not usable for output |
| Compute-node internet | yes (`huggingface.co` reachable) | stream/tokenise directly on the node |
| Python / CUDA | `module load u22/python/3.12.4 u22/cuda/12.4` | torch cu124 wheels include sm_75 |
| LLaDA-8B at `/share1/ishita/llada_model` | 15 GB, group-readable | `rsync` it from `ada` to `/ssd_scratch` at job start — a LAN copy, not a 15 GB HuggingFace download, and free against the `/home2` quota |
| Interactive `srun` | capped at 6 h | batch jobs get the full 4 days |

## The other trap: you will silently get a 1080 Ti

`sinfo -p u22` shows most nodes with `AVAIL_FEATURES=(null)`. Those are **GTX
1080 Ti**, compute capability 6.1. Nothing warns you: a job without
`--constraint=2080ti` schedules there happily and then either fails deep inside a
library or, worse, succeeds while producing numbers that are not comparable with
2080 Ti runs. Concretely on a 1080 Ti:

- **bitsandbytes NF4 4-bit does not work** — it needs compute >= 7.5.
- **No fp16 tensor cores** (needs >= 7.0), so the throughput budget is wrong by
  several times.

Every GPU job in `slurm/` pins `--constraint=2080ti`, and
`c3/traces/models.py` raises at load time if it finds itself on anything below
7.5 with 4-bit requested, rather than letting a mis-scheduled job quietly produce
incomparable traces.

## The trap that costs a day

`sbatch` jobs whose `--output` path is on `/share1` fail instantly with
`Reason=JobLaunchFailure`, exit code 53, elapsed `00:00:00` and **no log file at
all** — because SLURM cannot create the log. There is nothing to read and nothing
to debug. If a job dies this way, check `scontrol show job <id> | grep StdOut`
and confirm the path is on `/home2`.

The distinction that matters: `/share1` is not *mounted* on compute nodes, so
nothing can `open()` it there — but the node can still `ssh`/`rsync` from the
login node, which is how `run-llama.sh` fetches its 25 GB model. So `/share1` is
a fine place to *store* large files and a useless place to *read* them from.

## Consequences worth stating plainly

1. **bf16 is unavailable.** The proposal's W6 control is "re-run a 200-document
   subsample of LLaDA-8B in bfloat16". On Turing that cannot run. Either do the
   precision control in **fp32** (the harness supports it and will refuse bf16
   with an explanatory error), or get access to an Ampere partition.

2. **The published MDLM recipe is out of reach.** ~5.2e11 tokens is ~75 days of
   wall-clock on 4x2080Ti. One epoch of the 10BT subset is ~1.4 days and fits in
   a single job. `scripts/train_110m.py --budget-only` prints the full table.

3. **A requeued job can land on a different node**, so `/ssd_scratch` is not
   durable. Shards and checkpoints are kept on `/share1` and staged to the node
   at job start. This is what makes the 110M run viable despite `/home2` having
   only ~5 GB free: checkpoints at ~1.5 GB each go to `/share1`, not home.

4. **The Python environment ships as a tarball.** It is built on node-local SSD
   inside a job (the `python/3.12.4` module only resolves on compute nodes),
   tarred to `/share1` as one ~3 GB file, and extracted back to `/ssd_scratch` by
   each job that needs it (`slurm/activate_env.sh`).

## Trace collection, schema v2 (`slurm/collect_traces.sbatch`)

The sbatch is a SLURM array: task *I* collects the documents with
`doc_id % C3_NUM_SHARDS == I`, so pass `--array=0-(N-1)` with `C3_NUM_SHARDS=N`
(QOS: at most 4 GPUs at once, so `%2` throttling for 2-GPU tasks). Parts go to
`/ssd_scratch/$USER/traces/$C3_RUN/`, are pushed to `/share1/$USER/traces/$C3_RUN/`
every 30 minutes and on exit, and a requeued or resubmitted task pulls its shard's
parts back and resumes (`--resume`). A killed task loses at most one part
(50 documents). Merge with `scripts/merge_traces.py --stem <dir>/<run> --out ...`;
a one-shard run merges itself.

Cost per 256-token document under the plan v2 design (Sec. 3.3: 9 rates x 2
uniform masks, top-k trajectories for k = 2, 4, 8 recording every state, and a
matched uniform state for each decoder state): **466 forward passes and ~60k
rows**. Without the matched arm: 242 passes, ~31k rows. The T2 pilot design
(5 rates x 2, k = 4 only): 74 passes, ~9.6k rows. Disk: see
`artifacts/trace_size_estimate.json`.

**8B precision on 2080 Ti (no bf16).** fp16 weights are ~15 GiB, so one 11 GB
card is not enough:

| option | how | cost |
|---|---|---|
| fp16 over 2 GPUs | `--gres=gpu:2`, `C3_DEVICE_MAP=auto`, `C3_MAX_MEMORY=0=9GiB;1=9GiB` | layers run in sequence, so ~1 GPU of throughput for 2 GPUs of QOS; fp16 range, so every pass is checked for inf/NaN logits |
| fp32 + CPU offload | `C3_DEVICE_MAP=auto`, `C3_MAX_MEMORY=0=9GiB;cpu=60GiB`, `--mem=96G` | ~32 GB of weights, most streamed over PCIe every pass: far too slow for a sweep, usable for a 2-document spot check |
| 4-bit (NF4) | one GPU | an ablation only in plan v2 |

**Measured throughput:** not yet measured for schema v2. The log prints
docs/hour every 25 documents; record it here after the pilot re-run.
