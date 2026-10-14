# Ada: what this project actually has

## Changed in the October 2026 update (checked 2026-10-09)

The wiki's user guide mixes current and outdated facts; these were checked on
the cluster itself.

| Fact | Value | Consequence |
|---|---|---|
| Login | `ssh <user>@ada-gw1.iiit.ac.in` (or `ada-gw2`), Rocky Linux 9 gateways | `ada.iiit.ac.in` / `ada` refuse SSH (GSSAPI only), so jobs reach `/share1` via `$USER@$C3_LOGIN_HOST`, picked by `slurm/login_host.sh` (first of ada-gw1, ada-gw2 that answers; set `C3_LOGIN_HOST` to override) |
| Partition | still `u22` (not the wiki's `short`/`long`); `2080ti` feature unchanged | no change to `--partition` / `--constraint` |
| Memory cap | `u22` has `MaxMemPerCPU=3000` (MB) | every job asks for at most CPUs x 3000 MB, written in `M`: `G` means 1024 MB, so `24G` on 8 CPUs is over |
| QoS `medium` | 4 days, 4 GPUs, 40 CPUs, 125,000 MB, 20 jobs per user | the wiki's "4 running / 8 submitted" is outdated |
| Account | jobs name `--account=research`; associations were rebuilt in the update | a user missing from the new database gets `Invalid account or account/partition combination` on every partition, and only `hpc.admin@iiit.ac.in` can restore it |
| `/ssd_scratch`, `/scratch` | files deleted 7 days after creation (ctime) | `activate_env.sh` re-unpacks an env older than 48 h, so nothing a 4-day job uses is older than 6 days |
| `/share1` quota | 100 GB, 3,000 files | T3 uses ~51 GB (env 3.3, FineWeb 28.5, shards ~20) plus ~2 GB per run |
| Job history | `sacct` lost everything before the update | old job IDs exist only in `logs/` and `logs/t3_jobs.txt` |

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
| **ssh from compute nodes (since the Oct 2026 reset)** | `ada` refuses it ("no longer available for SSH access"); the gateways `ada-gw1` / `ada-gw2` accept key auth and see `/share1` | every job reaches `/share1` through `$C3_LOGIN_HOST` (probed by `login_host.sh`, sourced from `activate_env.sh`); a job that still says `$USER@ada:` fails to stage or push |
| **CPUs raised to fit memory (since the Oct 2026 reset, Slurm 26.05)** | more than ~3 GB per CPU raises cpus-per-task (64G with 16 CPUs -> 22), and `srun` inside the job then fails: `cpus-per-task set by two different environment variables` | `activate_env.sh` re-exports `SLURM_TRES_PER_TASK` to match; count the raised CPUs against the 40-CPU QOS cap |
| Node-local SSD | `/ssd_scratch`, 880 GB | staging target; persists on a node between jobs, so only the first job on a node pays the transfer |
| Compute-visible persistent | `/home2/$USER` (NFS), 30 GB quota, 25.7 GB used | **~5 GB free** — the binding storage constraint; `~/.cache` holds 5.3 GB of reclaimable pip/HF cache |
| Inode quota on `/share1` | 3,000 files | the built env is **47,361 files**, so it ships as a single tarball |
| `/share7`, `/cvit_contrib_packages` | NFS, compute-visible, **read-only** | not usable for output |
| Compute-node internet | yes (`huggingface.co` reachable), but downloads can stall (PyPI ran at 43 kB/s on 2026-09-27) | download data once, verified, and read it locally (`slurm/fetch_fineweb.sbatch`); don't stream it through a multi-hour job |
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

**Measured throughput (schema v2, 2026-09-29):**

| job | model / placement | rate |
|---|---|---|
| T2 pilot, 2722140 | LLaDA-8B-Base fp16, `device_map=auto` over 2 x 2080 Ti (9 GiB each) | **7.6 forward passes/s** of 256 tokens; 74 passes/doc -> 370 docs/hour. The full plan v2 design (466 passes/doc) -> ~59 docs/hour, ~34 h per 2,000-doc arm |
| contamination, 2722128 | same | 731 docs/hour (50 passes of ~100 tokens each) |
| 110M training, 2722281 | 2 x 2080 Ti, batch 16 x accum 8 | 43.4k tok/s -> 10B tokens in ~2.7 days. Batch 32 x accum 4 OOMed on gnode047, whose GPU 1 carried ~670 MB of foreign processes |
| 110M training, 17431 (2026-10-12) | 4 x 2080 Ti on one node (gnode060), batch 16 x accum 4 (same global batch) | 93.5-96k tok/s -> 10B tokens in ~30 h |
| T9 probe grid, 17340 | 110M fp32, 1 x 2080 Ti | 1,000 instances x 8 cells (46,480 decoding steps) in 201 s |
| T9 fine-tune, 17349 | 110M, 3 x 2080 Ti, batch 16 x accum 2, one instance per 256-token window | 23.9k real tok/s (instances average 85 tokens): 100M tokens in 1 h 10 min |

Five repeats of the pilot were bitwise identical under fp16 over two cards
(`artifacts/t2_pilot/floor_llada_base_fp16.json`).

**Model staging** lives in `slurm/stage_model.sh`, sourced by `collect_traces.sbatch`
and `contamination.sbatch`. The shared LLaDA copy is used only if it exists and its
tokenizer config matches the requested variant (a chat template means Instruct);
otherwise the pinned hub revision is downloaded to `/ssd_scratch/$USER/hf`.

**Contamination screen** (`slurm/contamination.sbatch`): ~50 forward passes of
~100 tokens per document, one job per model; needs `C3_THRESHOLD` until
`c3.CONTAM_THRESHOLD` is fixed, and writes `$PERSIST/contamination_<model>.json`.

## October 2026: login gateways, accounts, and the C1 sweep

**`ada.iiit.ac.in` no longer accepts SSH.** Its banner points at `ada-gw1` / `ada-gw2`,
and from the gateway `ssh ada` is refused. Jobs reach `/share1` through
`slurm/login_host.sh` (sourced by `activate_env.sh` and `build_env.sbatch`): `C3_LOGIN_HOST`
if set, otherwise the first of `ada-gw1 ada-gw2 ada` that accepts a batch-mode login. The
110M scripts (`train_110m`, `build_shards`, `fetch_fineweb`, `verify_110m`) still say
`$USER@ada` and belong to the T3 branch. The gateway has git 2.52 and python 3.9 (no
project environment). A fresh account had a 50 GB `/home2` quota, and `/share1` allowed
100,000 MB and 3,000 inodes.

**Every user needs a SLURM association.** `AccountingStorageEnforce=associations`, so a
user with none (`sacctmgr show assoc where user=$USER` prints nothing) gets
`Invalid account or account/partition combination specified` from every `sbatch` and
`srun`. On 12 Oct only `ojas.k` had one (`research`, QOS `medium`). Only the Ada admins
can add it.

**The C1 sweep** (`slurm/submit_c1.sh`, every submission logged to `logs/c1_jobs.txt`):

```bash
bash slurm/submit_c1.sh env                  # once: the env tarball (~30 min)
bash slurm/submit_c1.sh smoke                # 6 runs x 5 documents, full design
bash slurm/submit_c1.sh floor dream-7b-base  # the T1 floor for Dream
bash slurm/submit_c1.sh main                 # (a)-(d) in priority order
bash slurm/submit_c1.sh precision            # (f) 4-bit, 200 documents, 1 GPU
bash slurm/submit_c1.sh merge c1-llada-base  # per finished run: zero-copy merge + validation
```

Expected documents after the 256-token filter: LLaDA-8B 1,979 of 2,000 and Dream-7B
1,976 (the same counts as the contamination screens), 198 / 197 of the 200 precision
documents, and all 1,319 / 500 task documents (a few answers are truncated at 256). At
186 B/row the full design writes ~11 MB per FineWeb document, so ~22 GB per FineWeb arm
and ~64 GB for the whole sweep. That leaves no room on `/share1` for a second copy of
every run, so the merge is zero-copy: `<run>.dataset.json` lists the validated parts
(path, sha256, rows), and `c3.assemble.load_traces` reads it. Single-shard runs are
collected with `--merge-as none`, and `merge_validate.sbatch` merges every run the same
way.

**Memory buys CPUs.** `u22` sets `MaxMemPerCPU=3000` MB, so SLURM raises a job's CPU
count to its memory / 3000 MB. A 2-GPU trace task asking for 8 CPUs and 32 GB is
allocated 12 CPUs. With the 40-CPU QOS cap and two such tasks running, a CPU job can
have at most ~40 GB (14 CPUs). A larger request pends with `QOSMaxCpuPerUserLimit` even
though few CPUs are visibly in use (dev job 18932: 56 GB -> 20 CPUs; lowered with
`scontrol update JobId=... MinMemoryNode=40960 NumCPUs=14`).

## Set-level C3 on train110m-v4 (October 2026; `prereg/deviations.md` items 7-13)

Development runs from a second checkout, `/home2/$USER/c3-dev` (pass `C3_REPO=/home2/$USER/c3-dev`
and submit its copy of the script), so the runner checkout `/home2/$USER/c3` stays fixed while
its jobs are queued. Results land in `/home2/$USER/c3-runs/c3set/{dev,report}/`; the fit-half
dev copy of the trace and the on-policy step logs live on `/share1/$USER/c3set/`.

| Job | Resources | What |
|---|---|---|
| `c3set_analysis.sbatch` | CPU 14 / 42000M | `C3_STEPS` from `floor subset fieldtest setcal onpolicy fitcal infill`; `C3_MODE=dev` reads the fit half only |
| `infill_eval.sbatch` | 1 GPU, 6 / 18000M, array 0-3 | FineWeb infilling over a knob grid (`C3_GRID`), batched 32 documents per forward pass; resumable |
| `probe_c3set.sbatch` | 1 GPU, 8 / 24000M | C on the JSON probe with `ft110m-v4`; exit 1 if G6 fails |
| `wandb_log.sbatch` | CPU 2 / 4000M | result JSONs to W&B group `c3-setlevel-110m-v4`; never fails the job |
| `run_tests.sbatch` | CPU 4 / 12000M | the test suite on Linux (the bash-driven tests only run here) |

Batched infilling is the same algorithm as `c3.decode.loop.decode` on each document alone
(`tests/test_infill.py` holds tokens and step counts equal on the CPU toy, whose rows do not
depend on the batch). On a GPU, matrix kernels can round differently with batch shape, so
batches are fixed chunks of the sorted documents and a rerun reproduces them exactly; the dev
smoke run checks that two runs agree token for token.
The final report-half run is `prereg/c3set_commands.sh`, from the freeze tag only.
